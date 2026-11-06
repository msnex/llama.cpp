# llama.cpp 推理引擎工程师手册

## 从会跑，到会测，到改得动

> 基于 llama.cpp **0.9.0**（commit `cd959bae`，2026-11-04）编写。
> 主力机：**Apple M5 Pro，24GB 统一内存**，macOS + Metal 后端。
> 备用机：**Intel CPU + Vulkan 后端**（核显或 Arc 独显，两种情况的差别单列）。
> 读者画像：有 C/C++ 后端经验，熟悉内存、线程、性能分析，但没有 LLM 推理基础。
> 目标：8 到 12 周内建立“使用 -> 测量 -> 原理 -> 改造”的闭环，而不是背 API。

---

## 目录

- 第 0 章 先建立一张地图
- 第 1 章 把闭环跑通
- 第 2 章 够用的 Transformer 数学
- 第 3 章 llama.cpp 的骨架
- 第 4 章 ggml：计算图、内存与后端
- 第 5 章 性能评估与优化
- 第 6 章 量化：把权重压小，而不把模型压坏
- 第 7 章 适配：模型、后端、服务
- 附录 A：术语表
- 附录 B：命令速查
- 附录 C：源码地图
- 附录 D：性能报告模板
- 附录 E：自检题参考答案
- 附录 F：推荐阅读
- 附录 G：版本提示
- 最后一页：三句话

---

## 写在前面：这本书不做什么

llama.cpp 很大：0.9.0 的 `src/models/` 下有 173 个模型实现，工具链覆盖 CLI、server、多模态、语法约束、训练、RPC、Web UI。逐文件阅读会迅速迷路。

这本书只围绕推理核心：

- 一个 token 如何变成下一个 token；
- 权重、KV cache、激活值如何占用内存；
- 计算图如何映射到 Metal / Vulkan / CPU 后端；
- 如何测量 prefill 与 decode 的性能并定位瓶颈；
- 量化如何改变模型大小、速度和精度；
- 如何把一个新模型或新硬件接进来。

以下内容只“知道存在”即可，不在主线展开：训练/微调、Web UI、MCP、语法采样、多模态编码器细节、RPC、路由器模式、扩散模型、各种 speculative decoding 变体、173 个模型架构的差异。

### 两台机器怎么分工

| | 主力机 M5 Pro | 备用机 Intel + Vulkan |
|---|---|---|
| 角色 | 日常开发、调参、写报告 | 复现“没有统一内存”的世界、验证后端可移植性 |
| 强项 | 带宽高、单机能跑 14B 级模型、工具链省心 | 验证 x86 代码路径、Vulkan kernel、CPU 回退 |
| 弱项 | 内存焊死，24GB 是硬上限 | 带宽低（核显）或 VRAM 小（独显），iGPU 与 CPU 抢内存 |
| 必须学会的差异 | 统一内存的预算方式、Metal 的图与融合 | Vulkan 的设备/队列/显存类型、驱动差异 |

两台机器都要跑同一套实验。只在一台机器上得到的结论，通常会在另一台上被推翻，这本身是最有价值的一课。

### 约定

- 命令默认在 llama.cpp 仓库根目录执行，二进制在 `build/bin/`；构建目录不同就替换路径。
- macOS 命令用 `$(sysctl -n hw.ncpu)` 取核数，Linux 用 `$(nproc)`，文中会分别标注。
- 文中的 tok/s、内存、耗时都是**数量级示意**，必须用你自己的机器复测。标了“量级参考”的表格不是规格书。
- P0/P1/P2 表示学习优先级：P0 必学，P1 应学，P2 选学/了解。
- 代码片段为了突出主线省略了错误处理、资源释放和边界检查，不要直接复制到生产。
- API 以 0.9.0 的 `include/llama.h` 为准。升级后先看头文件和工具 README，再改代码。
- 文中的 “Metal” 指 `ggml/src/ggml-metal/`，“Vulkan” 指 `ggml/src/ggml-vulkan/`，两者都是 ggml 的后端，不是 llama.cpp 的某一层。

---

## 第 0 章 先建立一张地图

### 0.1 一句话描述 LLM 推理

LLM 推理就是反复做同一件事：

```text
文本 -> tokenizer -> token id 序列 -> embedding
     -> N 层 Transformer（Attention + FFN）
     -> logits -> sampler -> 下一个 token
     -> 把新 token 追加到序列，重复
```

推理引擎的职责，是把这条链路高效地映射到真实硬件上：

1. **模型表示**：把 Hugging Face 权重转成 GGUF，读进内存/显存；
2. **计算图**：把每一层的矩阵乘、注意力、归一化、激活函数组织成 ggml 图；
3. **内存管理**：权重、KV cache、临时激活值、多序列状态；
4. **调度**：batch 切分、CPU/GPU 分工、多后端图切分、多序列并发；
5. **采样与服务**：logits 变成 token，再变成文本；HTTP/CLI 接入。

### 0.2 两个阶段：prefill 和 decode

这是全书最重要的区分。

| | Prefill（预填充） | Decode（解码） |
|---|---|---|
| 输入 | 整段 prompt，T 个 token | 上一步生成的 1 个 token（batch 可为多序列） |
| 矩阵形状 | `[T, H] x [H, H]` | `[1, H] x [H, H]`（或 `[B, H]`） |
| 主要瓶颈 | 算力（compute-bound） | 内存带宽（memory-bound） |
| 关键指标 | pp t/s、TTFT | tg t/s、TPOT/ITL |
| 优化重点 | 大 batch、FlashAttention、GEMM、并行 | 量化、带宽、KV cache、offload、投机解码 |

同一个模型，prefill 的 t/s 通常是 decode 的几倍到上百倍（取决于硬件、batch 和上下文）。这不是实现问题，而是矩阵乘的算术强度不同。

在 M5 Pro 上，这个差距尤其大：Metal 的 GEMM 能吃到 GPU 算力，pp 可以比 tg 高一个数量级；在 Intel 核显上差距会小一些，因为两边都被同一根内存带宽卡住。

### 0.3 三个必须记住的数字关系

设模型参数量为 P，平均每个权重占 b bit：

- 权重字节数：`P * b / 8`
- decode 每生成 1 个 token 的计算量：约 `2P` FLOPs
- decode 每生成 1 个 token 至少要读一遍权重：约 `P * b / 8` 字节
- 算术强度：`2P / (P*b/8) = 16/b` FLOP/byte

4-bit 权重的算术强度只有约 4 FLOP/byte。CPU 的“脊点”大约在几到几十 FLOP/byte，GPU 更高。所以单序列 decode 几乎必然被内存带宽限制。

一个 8B 模型，Q4_K_M 约 4.6 GiB（约 4.9 GB）。把参照硬件的带宽代进去：

| 硬件 | 峰值带宽 | decode 理论上限（4.9GB 权重） | 实测常见区间 |
|---|---:|---:|---:|
| M5 Pro（统一内存） | 307 GB/s | 约 63 tok/s | 30 到 45 |
| Intel 核显 + DDR5-5600 双通道 | 约 90 GB/s | 约 18 tok/s | 7 到 12 |
| Intel Arc B580（独显） | 456 GB/s | 约 93 tok/s | 45 到 65 |
| Intel CPU 纯 CPU + DDR5-5600 | 约 90 GB/s | 约 18 tok/s | 5 到 9 |

实际值通常是理论值的 40% 到 70%。这几个数字能解释大部分“为什么我的 tok/s 只有这么点”的问题：**先算理论上限，再看实测落在百分之几，最后才谈优化**。

这个上限的算法只有两步：

1. **取权重字节数**：直接看 GGUF 文件大小最准（8B 的 Q4_K_M 约 4.9 GB）。用 `P * b / 8` 估算时，b 要取 4.8 到 4.9，不能取 4：Q4_K_M 会把 output 等张量留在更高比特，8B 模型的平均 bpw 约为 4.9。
2. **除以带宽**：`4.9 GB / 307 GB/s = 16.0 ms/token`，即约 63 tok/s；换成 90 GB/s 是 54.4 ms，约 18 tok/s。表里每一行都是这一步。

两个边界条件：

- 这是**短上下文**的界。decode 每步还要读 KV cache：32k 上下文、每 token 128 KiB 的模型，KV 读取约 4 GiB，和权重同一量级，上限会从 63 降到约 33 到 34 tok/s。
- GPU 后端的实测通常落在上限的 40% 到 70%；纯 CPU 往往只有 30% 到 50%，因为反量化和线程开销占掉了带宽预算。

### 0.4 四遍学习法

不要从第一页读到最后一页。按“四遍”推进：

| 遍数 | 读什么 | 做什么 | 产出 |
|---|---|---|---|
| 第一遍 | 第 0、1 章 | 两台机器都构建、跑模型、跑 benchmark | 两张机器基线表 |
| 第二遍 | 第 2、3 章 | 手算 KV/FLOPs，跟一遍 decode 调用链 | 能画出推理数据流 |
| 第三遍 | 第 4、5 章 | 看 ggml 图、后端切分、做性能实验 | 一份性能分析报告 |
| 第四遍 | 第 6、7 章 | 量化、imatrix、接一个小模型/算子 | 一个可复现的改造项目 |

### 0.5 主线与支线

| 优先级 | 内容 | 原因 |
|---|---|---|
| P0 必学 | prefill/decode、KV cache、GGUF、llama_context 生命周期、llama-bench | 日常工作的 80% |
| P0 必学 | 量化基本盘（Q4_K_M、Q8_0、imatrix、KV 量化） | 部署和性能的核心杠杆 |
| P0 必学 | 统一内存预算（M5 Pro）与 Vulkan 设备/显存类型（Intel） | 两台机器的“物理约束” |
| P1 应学 | ggml 图、backend scheduler、CPU/GPU offload | 定位“为什么慢/为什么回退” |
| P1 应学 | Metal 的命令缓冲/融合、Vulkan 的 shader/coopmat | 读懂后端日志 |
| P1 应学 | 模型转换与新增架构流程 | 适配工作的入口 |
| P2 选学 | 多 GPU、投机解码、多模态、server 并发 | 按岗位需要深入 |
| P2 了解 | 语法采样、UI、RPC、训练、扩散 | 知道边界即可 |

### 0.6 0.9.0 的 API 现状

llama.cpp 迭代很快。0.9.0 这个快照上，有三件事必须先知道，否则会照抄旧教程写出编译不过的代码：

1. **新的 batch API 已经成为示例的默认写法**：`llama_batch_ext` + `llama_process()`，`examples/simple/simple.cpp` 已经全部切换。旧的 `llama_batch` + `llama_decode()` 仍然保留，也仍然好用，两者数据流相同。
2. **内存（KV）抽象是 `llama_memory`**：`llama_get_memory(ctx)`、`llama_memory_clear()`、`llama_memory_seq_rm/cp/keep/add/div()`。旧的 `llama_kv_cache_*` 系列接口已经不在公开头文件里，旧博客里的代码需要改。
3. **一批默认值变了**（这是最容易踩的坑）：

| 参数 | 0.9.0 默认 | 旧教程常见写法 |
|---|---|---|
| `-ngl` | `auto`（尽量全放） | `0` |
| `-fit` | `on`（自动调整未指定的参数以放下模型） | 不存在 |
| `--jinja` | `on` | 需要显式打开 |
| server 端口 | `9931` | `8080` |
| `-fa` | `auto` | `off` |

书以**稳定的概念主线**为主：`model -> context -> batch -> memory -> graph -> backend -> sampler`。示例代码优先用新 API；需要和旧代码对照时，会给出 `llama_decode` 的等价写法。

遇到不确定的字段，以 `include/llama.h` 和对应工具的 `README.md` 为准。

### 0.7 两台机器的画像

#### 0.7.1 硬件与内存模型

| | M5 Pro 24GB | Intel + Vulkan |
|---|---|---|
| CPU | 最高 18 核（6 个 super core + 12 个性能核），实际核数以 `sysctl -n hw.ncpu` 为准 | 视型号，AVX2 或 AVX-512，可能有 P/E 混合核 |
| GPU | 最高 20 核，Metal 4，每核带 Neural Accelerator | 核显（UMA）或 Arc 独显（独立 VRAM） |
| 内存 | 24GB 统一内存，最高 307 GB/s | 系统内存 DDR5 双通道，约 64 到 100 GB/s；独显另有 8 到 12GB |
| 后端 | Metal，macOS 默认开启 | Vulkan，需要 `-DGGML_VULKAN=ON` |
| 内存模型 | CPU/GPU 同一块物理内存 | 核显 UMA 共享；独显分离 |
| offload 语义 | `-ngl` **不增加**可用内存 | 核显同左；独显才真正“增加容量” |
| 典型瓶颈 | 带宽、Metal 工作集上限、swap | 带宽（核显）、VRAM（独显）、驱动/kernel 覆盖 |

先在自己的机器上把下面几条命令的输出抄进报告，后面所有分析都要用到：

```bash
# macOS
sysctl -n machdep.cpu.brand_string
sysctl -n hw.ncpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu
sysctl -n hw.memsize
sw_vers
system_profiler SPDisplaysDataType | head -20

# Linux（Intel）
lscpu | head -20
free -h
vulkaninfo --summary
```

#### 0.7.2 统一内存三定律

这是主力机最重要的一节。三条定律决定了 M5 Pro 上所有调参的边界。

**第一定律：offload 不增加内存。**
在 CUDA 机器上，`-ngl` 把层从系统内存搬到显存，等于“扩容”。在 Apple silicon 上，Metal buffer 和 CPU 访问的是同一块物理内存，`-ngl` 只改变谁来计算，不改变总量。所以“先全放 GPU，放不下再降 ngl”这套 CUDA 直觉在这里失效：降 `-ngl` 不会腾出内存，只会把计算挪回 CPU。

**第二定律：Metal 的有效预算不是 24GB。**
Metal 设备报告的 `recommendedMaxWorkingSetSize` 是系统建议的工作集上限，通常明显小于物理内存。启动日志会打印它，`--list-devices` 也会显示：

```text
  MTL0: Apple M5 Pro (17xxx MiB, 17xxx MiB free)
```

预算要按这个数算，而不是按 24GB 算。超出它不一定会立刻失败，但会进入内存压力，触发压缩和 swap，tok/s 断崖式下跌。

**第三定律：权重是 mmap 进来的，不是复制进来的。**
Metal 后端声明了 `buffer_from_host_ptr` 能力，模型加载器在默认 buffer 类型上会直接把 mmap 的权重页交给 GPU 使用。这意味着：

- 权重只占一份物理内存，大小约等于 GGUF 文件大小；
- 用 `-lm none` 会放弃 mmap 映射，把权重读进 Metal buffer（一份拷贝，加载更慢）；
- 用 `-lm mlock` 会把文件页钉住，避免被换出，但也会压缩可用余量。

#### 0.7.3 24GB 的预算表

一台干净的 M5 Pro，24GB 要这样分：

| 项目 | 量级 | 说明 |
|---|---:|---|
| macOS + 常驻应用 | 3 到 6GB | 关掉浏览器和 IDE 可以压到 3GB 左右 |
| Metal 驱动、命令缓冲、图内存 | 0.5 到 1.5GB | 随 `n_ubatch` 增大 |
| 权重 | 约等于文件大小 | mmap + host ptr，不复制 |
| KV cache | 按公式算 | 启动日志里的 `KV self size` 是准的 |
| 激活峰值 | 0.3 到 2GB | 随 `n_ubatch` 近似线性增长 |
| 安全余量 | 1 到 2GB | 不要用满，prefill 峰值比 decode 高 |

由此得到几条实用结论：

- **8B Q4_K_M（4.9GB）+ 32k f16 KV（约 4GB）+ 余量 = 约 10GB**，很舒服；
- **8B Q4_K_M + 128k f16 KV（约 16GB）** 已经超过 Metal 工作集，必须 `-ctk q8_0 -ctv q8_0` 或降上下文；
- **14B Q4_K_M（约 9GB）+ 16k KV（约 2.5GB）** 可行，32k 就要看模型层数和 KV 头数；
- **30B 级稠密模型**放不下，30B 级 MoE 要用 Q3 或更激进的量化，并接受更短的上下文。

#### 0.7.4 备用机的三种情况

“Intel CPU + Vulkan”不是一种机器，而是三种，性能差一个数量级。第一件事是确认自己是哪一种：

```bash
./build/bin/llama-cli --list-devices
```

| 情况 | 日志特征 | 内存模型 | 8B Q4 的 tg 量级 |
|---|---|---|---|
| 核显（Iris Xe / Arc iGPU） | `uma: 1`，total 约等于系统内存的一半 | 共享系统内存 | 7 到 12 tok/s |
| Arc 独显（A750/B580 等） | `uma: 0`，total 为 VRAM | 独立 VRAM | 45 到 65 tok/s |
| 纯 CPU（Vulkan 设备不可用） | 只有 CPU 设备 | 系统内存 | 5 到 9 tok/s |

核显的关键认知：**Vulkan 提速的是 prefill，不是 decode**。decode 仍然被系统内存带宽限制，和纯 CPU 差不多；Vulkan 的价值是把 prefill 从 CPU 手里拿走，并把 CPU 核空出来。如果你的任务是长 prompt、短输出，核显收益明显；如果是长输出，收益有限。

#### 0.7.5 推荐起手配置

下面是两台机器的“第一组参数”。它们是起点，不是结论；先用它跑通、跑出基线，再按第 5 章的方法做单变量实验。

M5 Pro 24GB：

| 目标 | 模型/量化 | 上下文 | 关键参数 | tg 量级 |
|---|---|---|---:|---:|
| 跑通闭环 | 0.5B 到 1.5B，Q4 | 8k | `-ngl 99 -fa on` | 100 以上 |
| 日常主力 | 8B，Q4_K_M | 16k | `-fa on -ctk q8_0 -ctv q8_0` | 30 到 45 |
| 长文档 | 8B，Q4_K_M | 64k | 同上 | 20 到 30 |
| 质量优先 | 14B，Q4_K_M | 8k | 同上 | 15 到 20 |
| 不建议 | 30B 级稠密 | - | - | 放不下 |

Intel + Vulkan：

| 目标 | 模型/量化 | 上下文 | 关键参数 | tg 量级 |
|---|---|---|---:|---:|
| 核显日常 | 4B 到 8B，Q4 | 8k | `-ngl 99 -fa on -ub 512` | 7 到 12 |
| Arc 独显 | 8B，Q4_K_M | 16k | `-ngl 99 -fa on -ctk q8_0` | 45 到 65 |
| 纯 CPU 兜底 | 4B，Q4 | 4k | `-ngl 0 -t <P 核数>` | 10 到 20 |

三条选择规则：

1. 权重 + KV + 1.5GB 必须小于 Metal 工作集（Mac）或 VRAM / 总内存（Intel）；
2. 质量优先时先加比特，再加上下文；内存不够时先砍上下文，再砍比特；
3. 任何“起手配置”都要用 `llama-bench` 复测一遍，再写进报告。

---

## 第 1 章 把闭环跑通

**本章目标**：在两台机器上完成“构建 -> 推理 -> 基准测试 -> 质量冒烟”，并知道启动日志里哪些数字值得看。

**先记结论**：

- 先确保你跑的是 Release 构建，不是 Debug。
- 先跑 `llama-bench`，不要凭聊天窗口的体感判断性能。
- 第一次运行包含模型下载、mmap 缺页、后端初始化，数据不能当基线。
- 任何性能结论都必须带上：模型、量化、上下文长度、batch、后端、线程数、GPU 层数、系统版本。

### 1.1 构建

#### 1.1.1 macOS / Metal（主力机）

前置条件：

```bash
xcode-select --install                        # 命令行工具
xcodebuild -downloadComponent MetalToolchain  # 较新的 Xcode 需要单独下载 Metal 工具链
brew install cmake                            # 或者用官方 pkg
```

取 0.9.0 的快照并构建：

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout cd959bae99ff4bed22bef2604f45241c09bbb8c8

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j "$(sysctl -n hw.ncpu)"
```

说明：

- macOS 上 Metal 默认开启，不需要 `-DGGML_METAL=ON`；要关掉才写 `-DGGML_METAL=OFF`。
- Metal 着色器默认编进二进制（`GGML_METAL_EMBED_LIBRARY=ON`），所以构建慢、运行快，不会在每次启动时编译 kernel。
- Accelerate 框架默认开启（`GGML_ACCELERATE=ON`），它主要帮 CPU 侧的 prefill。
- 不写 `CMAKE_BUILD_TYPE` 时 CMake 也会默认 Release，但显式写出来可以避免误建 Debug。
- 如果构建报找不到 `metal` 编译器，先确认 Metal Toolchain 已经下载，再确认 Xcode 版本与系统匹配。

#### 1.1.2 Intel / Vulkan（备用机）

Linux 依赖：

```bash
sudo apt-get install libvulkan-dev glslc spirv-headers glslang-dev glslang-tools vulkan-tools
vulkaninfo --summary
```

Windows 用 Vulkan SDK（LunarG），把 `glslc.exe` 放进 PATH；MSYS2 用 `mingw-w64-ucrt-x86_64-vulkan-devel` 等包。细节见 `docs/build.md` 的 Vulkan 一节。

构建：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build --config Release -j "$(nproc)"
```

Windows（Visual Studio 生成器）注意两点：`-j` 要写成具体数字，产物在 `build/bin/Release/`。

```bat
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j 8
build\bin\Release\llama-cli.exe --list-devices
```

说明：

- Vulkan 后端在构建期用 `glslc` 把上百个 shader 编译成 SPIR-V，第一次构建明显比 CPU-only 慢，属正常。
- 构建完先跑 `vulkaninfo` 和 `--list-devices`。如果 `--list-devices` 只显示 CPU，问题在 ICD/驱动，不在 llama.cpp。
- 纯 CPU 兜底构建（排查用）：`cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release`，不带任何 GPU 后端。
- 想同时保留两套，用不同构建目录：`build`（Vulkan）和 `build-cpu`（纯 CPU）。

#### 1.1.3 验证构建

```bash
./build/bin/llama-cli --version
./build/bin/llama-cli --list-devices
```

两条输出都要存档：

- `--version` 会打印版本号、commit、编译器、后端编译选项。0.9.0 的源码树如果不是 release tag，会显示 `0.9.0-dev`，这是正常的。
- `--list-devices` 会打印设备名和可用内存。macOS 上这个“可用内存”就是 Metal 的 `recommendedMaxWorkingSetSize`，是你做预算的依据。

产物在 `build/bin/`。常用：

| 程序 | 用途 |
|---|---|
| `llama` | 统一入口：`llama serve`、`llama cli`、`llama download` 等 |
| `llama-cli` | 传统 CLI，功能最全 |
| `llama-server` | OpenAI 兼容 HTTP 服务 |
| `llama-bench` | pp/tg 基准测试 |
| `llama-batched-bench` | 多 batch 吞吐测试 |
| `llama-perplexity` | 困惑度/KL 评估 |
| `llama-imatrix` | 采集重要性矩阵 |
| `llama-quantize` | 量化 |
| `llama-fit-params` | 估算并打印“能放下的参数组合” |
| `ggml-metal-tuning` | 仅在 macOS + Metal 构建中生成，离线扫 kernel 配置（可选） |

> 注意：仓库里如果存在 `build-x64-linux-gcc-debug` 之类的目录，不要拿它做性能测试。Debug 构建可能慢数倍。

### 1.2 第一次推理

最省事的方式是从 Hugging Face 直接拉一个 GGUF：

```bash
./build/bin/llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF
```

或者用本地文件：

```bash
./build/bin/llama-cli -m /path/to/model.gguf -p "用一句话解释 KV cache" -n 128
```

启动服务：

```bash
./build/bin/llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

默认监听 `127.0.0.1:9931`（不是 8080）。以启动日志里的地址为准：

```bash
curl http://127.0.0.1:9931/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"你好"}]}'
```

第一次只验证“能跑通”，不要调参。

### 1.3 启动日志读什么

不同后端打印的字段不同，但核心信息稳定。重点找：

```text
general.architecture = llama        # 模型架构
n_layer / n_embd / n_head / n_head_kv
n_ctx_train                         # 训练上下文
model size / model params           # 权重大小与参数量
n_ctx                               # 你实际分配的上下文
n_batch / n_ubatch                  # 逻辑 batch / 物理 batch
flash_attn                          # 是否启用
type_k / type_v                     # KV cache 类型
KV self size                        # KV 占用
graph nodes / graph splits          # 图节点数与后端切分数
```

macOS 上还会多一段 Metal 设备信息，这是主力机的“体检报告”：

```text
ggml_metal_device_init: GPU name:   MTL0 (Apple M5 Pro)
ggml_metal_device_init: GPU family: MTLGPUFamilyAppleN  (N)
ggml_metal_device_init: simdgroup reduction   = true
ggml_metal_device_init: simdgroup matrix mul. = true
ggml_metal_device_init: has unified memory    = true
ggml_metal_device_init: has bfloat            = true
ggml_metal_device_init: has tensor            = true
ggml_metal_device_init: use residency sets    = true
ggml_metal_device_init: use shared buffers    = true
ggml_metal_device_init: recommendedMaxWorkingSetSize  = xxxxx.xx MB
```

其中两个字段要特别看：

- `has tensor = true`：M5 系列默认启用 Metal Tensor API（`mpp::tensor_ops` 的 matmul）。这个路径在 M5 之前默认关闭，因为旧芯片上不划算。它是 M5 上 prefill 变快的主要原因之一。
- `recommendedMaxWorkingSetSize`：你的真实预算。24GB 机器上它明显小于 24GB。

Intel 机器看 Vulkan 设备行：

```text
ggml_vulkan: 0 = Intel(R) Graphics (ADL GT2) | uma: 1 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
```

- `uma: 1` 表示核显共享系统内存；
- `matrix cores` 为 `none` 表示没有 coopmat 路径，GEMM 走普通 shader；
- `fp4: 0` 表示不支持 FP4，NVFP4 模型不要选。

三个数字最容易被忽略：

1. `n_ctx`：KV cache 按它分配，直接决定内存占用；
2. `n_ubatch`：真正送进计算图的 token 数上限，影响 GEMM 形状和内存峰值；
3. `graph splits`：图被切成几段、跨几个后端；切分越多，拷贝和同步越多。

### 1.4 建立基线：llama-bench

最常用的命令：

```bash
./build/bin/llama-bench -m model.gguf -p 512 -n 128 -r 5 -o md
```

含义：

- `-p 512`：prefill 512 个 token，测 pp；
- `-n 128`：生成 128 个 token，测 tg；
- `-r 5`：重复 5 次取平均；
- `-o md`：输出 Markdown 表格，方便贴进报告。

也可以指定上下文深度：

```bash
./build/bin/llama-bench -m model.gguf -p 512 -n 128 -d 4096
```

`-d` 会先把 KV cache 填到 4096 深度，再测 pp/tg。长上下文性能必须带 `-d` 测，否则会高估。

0.9.0 里有两个对本机很有用的选项：

- `-fitt`（目标余量）和 `-fitc`（最小上下文）：让 `llama-bench` 自己把 `-ngl` 和 `-c` 调到放得下，适合 24GB 这种“差一点就放不下”的场景。注意 `llama-bench` 没有 `-fit` 这个开关，`-fit` 属于 `llama-cli`/`llama-server`；
- `--list-devices`：先确认测试用的是哪个设备；多设备时用 `-dev Vulkan0`（或 `-dev MTL0`）明确指定，避免测到别的卡。

输出表的关键列：

| 列 | 含义 |
|---|---|
| `model` | 模型与量化类型 |
| `size` | 模型文件大小 |
| `params` | 参数量 |
| `backend` | CPU/Metal/Vulkan 等 |
| `ngl` | offload 到 GPU 的层数 |
| `test` | `pp512`、`tg128` 等 |
| `t/s` | 平均吞吐，正负标准差 |

> `llama-bench` 的测量不包含 tokenization 和 sampling 时间。它测的是纯模型计算。真实端到端延迟还要加上这两部分，尤其是长 prompt 和复杂 sampler。

### 1.5 质量冒烟

性能之前先确认输出没坏。做三件事：

1. 用固定 prompt 问 3 个问题，确认不是乱码、重复、答非所问；
2. 用 `llama-perplexity` 在 Wikitext-2 上算一个 PPL；
3. 如果模型带 chat template，确认多轮对话正常。

```bash
./scripts/get-wikitext-2.sh
./build/bin/llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw -c 512 -ngl 99
```

注意文件名是 `wikitext-2-raw/wiki.test.raw`，不是 `wikitext-2/`。

PPL 只用于**同一模型、同一 tokenizer、同一数据**下的相对比较（例如量化前后）。不要拿它跨模型比较，也不要用它判断聊天质量。

### 1.6 实验 1：两台机器的基线表

至少在主力机上跑下面四组，在备用机上跑对应组：

```bash
# A. CPU，全部层（先用 4 线程，再按 P 核数扫 1/2/4/6/8）
llama-bench -m model.gguf -p 512 -n 128 -ngl 0 -t 4

# B. GPU，全部层
llama-bench -m model.gguf -p 512 -n 128 -ngl 99

# C. GPU + FlashAttention
llama-bench -m model.gguf -p 512 -n 128 -ngl 99 -fa on

# D. GPU + FA + q8_0 KV
llama-bench -m model.gguf -p 512 -n 128 -ngl 99 -fa on -ctk q8_0 -ctv q8_0
```

macOS 上线程数用 `$(sysctl -n hw.perflevel0.physicalcpu)`，Intel 上先用物理核数，再扫 1/2/4/8/全部。

记录：

| 配置 | pp512 t/s | tg128 t/s | 峰值内存 | 备注 |
|---|---:|---:|---:|---|
| A CPU |  |  |  |  |
| B GPU |  |  |  |  |
| C GPU+FA |  |  |  |  |
| D +KV q8 |  |  |  |  |

观察重点：

- B 比 A 快多少？M5 Pro 上 pp 应该快一个数量级，tg 快 2 到 4 倍；Intel 核显上 tg 可能几乎不变。
- C 对 pp 和 tg 分别影响多少？长上下文时 FA 通常更明显。
- D 的 tg 是否变快、内存是否下降？如果变慢，可能是 KV 量化 kernel 不支持或回退。
- 同一组参数在 M5 Pro 和 Intel 上的比值，是否接近两者带宽的比值？接近说明都受带宽限制，不接近说明有一台被别的东西卡住了。

### 1.7 常见坑

| 现象 | 常见原因 | 处理 |
|---|---|---|
| 启动就 OOM | `n_ctx` 太大、并行数太多、量化不够狠 | 降 `-c`、降 `-np`、换更低比特 |
| macOS 风扇狂转、tok/s 掉一半 | 内存压力、swap、后台应用 | 关应用、`vm_stat` 看 pageout、降 `-c` |
| Metal 没被用上 | 用了 CPU-only 构建，或 `-ngl 0` | 重构建，`--list-devices` 确认，去掉 `-ngl 0` |
| Vulkan 只看到 CPU | ICD 未安装、`VK_ICD_FILENAMES` 不对、驱动太旧 | `vulkaninfo`，装驱动/ICD |
| 输出乱码/重复 | chat template、BOS/EOS、量化过狠 | 换官方模板、换 Q5/Q6、检查 tokenizer |
| 第一次很慢 | 下载、mmap 缺页、后端初始化、shader 加载 | 先跑一次 warmup，再测 |
| 多线程越多越慢 | 混合核（P/E）、带宽饱和 | 用 P 核数，测 `-t` 曲线；绑核只在 Linux/Windows 有效 |
| 同一命令两次差异大 | 温度墙、后台负载、页缓存、内存压缩 | 固定负载，多次取中位数 |
| 构建时找不到 metal | Xcode 未装 Metal Toolchain | `xcodebuild -downloadComponent MetalToolchain` |
| 跑的是 Rosetta 版本 | Homebrew 在 `/usr/local`，二进制是 x86_64 | `file build/bin/llama-cli`，换成 arm64 构建 |

### 1.8 本章自检

1. 为什么 `llama-bench` 的 tg 数值不能直接当成用户看到的 tok/s？
2. `n_ctx`、`n_batch`、`n_ubatch` 分别控制什么？
3. 你的 M5 Pro 上，8B Q4 模型的理论 decode 上限是多少？实测是多少？差在哪？
4. 为什么在 M5 Pro 上“降 `-ngl` 腾内存”是错的，而在 CUDA 机器上是对的？
5. 为什么必须先跑 Release 构建？

## 第 2 章 够用的 Transformer 数学

**本章目标**：不推导反向传播，不做训练，只建立“每个张量是什么形状、每一步花多少计算和内存”的直觉。够你读懂 llama.cpp 的图和后端日志即可。

**先记结论**：

- 推理的主体是矩阵乘，不是“神经网络魔法”。
- Attention 负责 token 之间的信息交换；FFN 负责逐 token 的特征变换。
- KV cache 用空间换时间：避免 decode 时重算历史 K/V。
- GQA/MQA 减少 KV 头数，直接决定 KV cache 大小。
- MoE 减少“每 token 激活的参数量”，但不减少“必须存下的参数量”。
- 统一内存机器上，“显存”和“内存”是同一个预算。

### 2.1 形状约定：ggml 和 PyTorch 是反的

ggml 张量有 4 个维度 `ne[0..3]`，`ne[0]` 是内存中最连续的方向。线性层的权重在 ggml 中通常写成：

```text
ffn_up.weight: {n_embd, n_ff}
```

对应 PyTorch 的 `[n_ff, n_embd]`。也就是说，**ggml 的维度顺序基本是 PyTorch 的逆序**。

读源码时看到：

```cpp
create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, n_ff}, 0);
```

要知道它表示“输入 n_embd，输出 n_ff”的矩阵，计算时是 `x[n_embd] @ W[n_embd, n_ff]`。

另一个关键点是 `nb`（stride，字节步长）：

- `nb[0]` 是一个块/元素的字节数；
- `nb[1]` 是沿第 1 维走一步的字节数；
- 对量化张量，`ne[0]` 必须是 block size 的整数倍。

这解释了为什么很多算子要求“最后一维连续”，以及为什么转置、view、contiguous 在 ggml 里是需要认真对待的事。

在 Metal 和 Vulkan 上，这个约定直接决定 kernel 是否高效：`ne[0]` 连续意味着沿 K 维可以向量化读取，量化 block 也能对齐。

### 2.2 Tokenizer 与 embedding

Tokenizer 把文本切成 token，并映射为整数 id。常见类型：

| 类型 | 代表模型 | 特点 |
|---|---|---|
| SPM | LLaMA 系列 | byte-level BPE + byte fallback |
| BPE | GPT-2/Qwen 等 | byte-level BPE |
| WPM | BERT | WordPiece |
| UGM | T5 | Unigram |

llama.cpp 里 `llama_vocab` 负责：

- `llama_tokenize()`：文本 -> token id；
- `llama_token_to_piece()`：token id -> 文本片段；
- BOS/EOS/EOT、特殊 token、FIM token 等元数据。

Embedding 是一个查表：

```text
x = token_embd.weight[token_id]    // [n_embd]
```

所以 `token_embd.weight` 的形状是 `{n_embd, n_vocab}`。对 128k 词表、4096 维的模型，这个矩阵本身就有 5 亿参数，不能忽略。在 24GB 机器上，词表大的模型光 embedding 就占 0.5 到 1GB。

### 2.3 归一化与线性层

现代 LLM 常用 RMSNorm：

```text
y = x / sqrt(mean(x^2) + eps) * gamma
```

它比 LayerNorm 少一个减均值，计算更简单。llama.cpp 的 `build_norm()` 会根据模型选择 `LLM_NORM`、`LLM_NORM_RMS` 或 `LLM_NORM_GROUP`。

线性层就是矩阵乘：

```text
y[M, N] = x[M, K] @ W[K, N]
```

- prefill 时 M = T（几百到几万），是 GEMM，容易打满算力；
- decode 时 M = 1（或 batch B），退化为 GEMV，主要受内存带宽限制。

这就是“同一个 matmul，prefill 和 decode 性能差几十倍”的根本原因。M5 Pro 的 Tensor API 优化的正是 prefill 的 GEMM；decode 的 GEMV 仍然由带宽决定。

### 2.4 Attention：token 之间交换信息

对每个 head，设 head_dim 为 d：

```text
Q = x @ Wq    // [T, d]
K = x @ Wk    // [T, d]
V = x @ Wv    // [T, d]

scores = Q @ K^T / sqrt(d)     // [T, T]
scores = causal_mask(scores)   // 只能看自己和左边
P = softmax(scores)            // [T, T]
O = P @ V                      // [T, d]
```

多个 head 各自做一遍，拼接后再过 `Wo`。

几个必须理解的点：

1. **Causal mask**：第 i 个 token 只能 attend 到 `<= i` 的位置。prefill 时是一次性算完整个 T x T 矩阵；decode 时 Q 只有 1 行，K/V 是历史所有 token。
2. **GQA/MQA**：query head 数 `n_head` 通常大于 KV head 数 `n_head_kv`。多个 query head 共享同一组 K/V。这直接减少了 KV cache 和 K/V 投影的计算。
3. **RoPE**：把位置信息编码成对 Q/K 的旋转。它让 attention 分数依赖相对位置。RoPE 的 base、scaling 类型、YaRN 参数都会影响长上下文外推。
4. **FlashAttention**：不显式物化 T x T 的 scores 矩阵，而是分块计算 QK^T、softmax、PV，减少内存读写。长上下文时尤其重要，也是量化 V cache 的前提。

Attention 的复杂度：

- prefill：`O(T^2 * d)` 每层，T 大时是平方级；
- decode：每 token `O(T * d)` 每层，T 是当前上下文长度。

所以长上下文不仅吃 KV 内存，也让 decode 每步的 attention 变慢。只是当 T 还不够大时，权重读取仍然占主导。

#### 2.4.1 两个后端对 FlashAttention 的支持边界

这是“照抄 CUDA 参数”最容易翻车的地方。0.9.0 的源码里可以读到确切的限制。

Metal（`ggml-metal-device.m` 的 `GGML_OP_FLASH_ATTN_EXT` 分支）：

- 支持的 K/V 类型：`f32`、`f16`、`q8_0`、`q4_0`、`q4_1`、`q5_0`、`q5_1`，以及设备支持 bfloat 时的 `bf16`；
- 支持的 head_dim：32、40、48、64、72、80、96、112、128、192、256、320、512、576；
- K 和 V 的类型必须相同；
- 除 K = V 外，只支持少数 K > V 的组合：96/64、128/96、192/128、320/256、576/512。

Vulkan（`ggml_backend_vk_device_supports_op`）：

- 支持的 K/V 类型：`f32`、`f16`、`bf16`、`q8_0`、`q5_1`、`q5_0`、`q4_1`、`q4_0`、`iq4_nl`；
- 要求 head_dim 是 8 的倍数；
- 不支持 K/V 行列表（`src[5]` 非空时直接回退）。

对照表：

| KV 类型 | Metal | Vulkan |
|---|---|---|
| f16 / f32 | 支持 | 支持 |
| bf16 | 需要 bfloat 支持（M5 支持） | 支持 |
| q8_0 | 支持 | 支持 |
| q4_0 / q4_1 / q5_0 / q5_1 | 支持 | 支持 |
| iq4_nl | 不支持（回退） | 支持 |
| 其他 K-quant（q4_K 等） | 不支持 | 不支持 |

严格说，`-ctk`/`-ctv` 的允许值本来就只有 `f32`、`f16`、`bf16`、`q8_0`、`q4_0`、`q4_1`、`iq4_nl`、`q5_0`、`q5_1`，K-quant 不能作为 KV 类型。

所以“KV 量化”在 llama.cpp 里指的是 `-ctk` / `-ctv` 的那几种 `ggml_type`，不是把 K-quant 塞进 KV。选 `q8_0` 或 `q4_0` 之前，先确认你的 head_dim 在支持列表里，否则会静默回退，变慢且不报错。

### 2.5 FFN：逐 token 的特征变换

Llama 类模型常用 SwiGLU：

```text
gate = silu(x @ Wgate)     // [T, n_ff]
up   = x @ Wup             // [T, n_ff]
y    = (gate * up) @ Wdown // [T, n_embd]
```

其中 `n_ff` 通常是 `n_embd` 的 2.5 到 4 倍。FFN 的参数量往往占全模型的一半以上，是 decode 时读取权重的大头。

### 2.6 KV cache：decode 的空间换时间

没有 KV cache 时，生成第 t 个 token 要重新算前 t 个 token 的 K/V，复杂度是平方级。有了 KV cache：

- prefill 阶段：算出所有 prompt token 的 K/V，写入 cache；
- decode 阶段：只算新 token 的 Q/K/V，K/V 追加到 cache，attention 直接读历史 K/V。

每个 token 的 KV cache 大小：

```text
bytes/token = 2 * n_layer * n_head_kv * head_dim * bytes_per_element
```

前面的 2 是 K 和 V。

以 Llama 3 8B 为例：`n_layer=32`，`n_head_kv=8`，`head_dim=128`，f16：

```text
2 * 32 * 8 * 128 * 2 = 131072 bytes = 128 KiB/token
```

- 8k 上下文：约 1 GiB；
- 32k 上下文：约 4 GiB；
- 128k 上下文：约 16 GiB。

这还没算上对齐、多序列、临时 buffer。很多“模型明明能放下，一开长上下文就 OOM”的问题，都是 KV cache 造成的。

在 24GB 的 M5 Pro 上，这张表要背下来：

| 每 token KV | 32k 上下文 f16 | 32k 上下文 q8_0 | 128k 上下文 q8_0 |
|---|---:|---:|---:|
| 64 KiB（小模型 / 少 KV 头） | 2 GiB | 1 GiB | 4 GiB |
| 128 KiB（8B 级） | 4 GiB | 2 GiB | 8 GiB |
| 256 KiB（32B 级） | 8 GiB | 4 GiB | 16 GiB |

配上 8B Q4 的 4.9GB 权重：`4.9 + 8 + 1.5 = 14.4GB`，128k + q8_0 KV 是可行的；f16 KV 就是 21GB，必然进入内存压力。这就是为什么在 Mac 上 `-ctk q8_0 -ctv q8_0` 经常不是“优化”，而是“能不能跑”的前提。

KV cache 的布局、量化、共享、回收，是推理引擎最重要的内存管理问题之一。llama.cpp 用 `llama_memory` 抽象它，默认实现是 `llama_kv_cache`，此外还有 SWA、recurrent、hybrid、DSA 等变体。

### 2.7 Logits 与采样

最后一层输出 `hidden[n_embd]`，经过 `output_norm` 和 `output.weight` 得到：

```text
logits[n_vocab] = hidden @ output.weight^T
```

logits 是未归一化的分数。采样器把它变成下一个 token：

| 采样器 | 作用 |
|---|---|
| greedy | 取 argmax，确定性 |
| temperature | `logits / T`，T 越大越随机 |
| top-k | 只保留概率最高的 k 个 |
| top-p | 保留累计概率达到 p 的最小集合 |
| min-p | 保留概率不低于 max_p * p 的 token |
| repetition/frequency/presence penalty | 惩罚重复 token |
| grammar | 用语法约束候选集（支线） |

llama.cpp 的采样器是链式的：

```cpp
llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
```

顺序会影响结果。通常先做惩罚/过滤，再 temperature，最后 dist。greedy 不需要 dist。

### 2.8 MoE：稀疏激活

MoE 层把 FFN 换成多个 expert，加一个 router：

```text
scores = router(x)             // [n_expert]
topk = 选前 k 个 expert
y = sum_{e in topk} gate_e * expert_e(x)
```

特点：

- 每 token 只算 k 个 expert，计算量接近 k/n_expert；
- 但所有 expert 权重都要存，内存占用不降；
- decode 时 expert 权重的读取仍然是大头，且访问模式不规则；
- 常见优化是把 expert 放 CPU/分层 offload，或者做 expert cache。

llama.cpp 用 `mul_mat_id` 之类的算子处理 MoE，支持 `-ncmoe`、`--cpu-moe`、`--moe-cache-mib` 等选项。

**在统一内存机器上，MoE 的 offload 逻辑要重新理解**：把 expert 放 CPU 不节省物理内存，只是换计算单元。真正省内存的手段只有：更低比特、更少 expert 层、更小上下文。`--moe-cache-mib` 在 Mac 上基本没有意义，因为不存在“从系统内存搬到 VRAM”这回事。

主线只需要理解“稀疏计算、稠密存储”。

### 2.9 变体地图：知道它们改了什么

| 变体 | 与标准 Transformer 的差异 | 对推理的影响 |
|---|---|---|
| SWA | 只看最近 W 个 token | KV cache 可只保留窗口，长上下文省内存 |
| MLA/DSA | 压缩/稀疏的 KV 表示 | KV 布局和 attention 实现不同 |
| Recurrent/Hybrid | 用循环状态替代不断增长的 KV | 状态大小固定，但并行/回滚复杂 |
| MoE | FFN 稀疏激活 | 内存大、计算少、访问不规则 |
| Diffusion | 非自回归去噪 | 不是逐 token decode，调度不同 |

这些变体在 llama.cpp 里对应不同的 `llama_memory` 和 graph 实现。第一遍学习只读标准 KV cache，遇到其他模型知道“内存模型变了”即可。

### 2.10 成本模型：roofline

把一次计算抽象成：

```text
时间 >= max(内存字节数 / 带宽, FLOPs / 峰值算力)
```

两者的比值就是算术强度（FLOP/byte）。硬件的“脊点”是峰值算力/带宽：

- 低于脊点：memory-bound；
- 高于脊点：compute-bound。

对 decode（batch=1）：

```text
AI = 2P / (P*b/8) = 16/b
```

- b=4：约 4 FLOP/byte；
- b=8：约 2 FLOP/byte；
- b=16：约 1 FLOP/byte。

CPU 的脊点通常是几到几十，GPU 是几十到几百。所以 decode 基本是 memory-bound。量化降低 b，直接降低字节数，通常提高 decode 速度。

对 prefill（T 个 token）：

```text
AI = 2PT / (P*b/8) = 16T/b
```

T 越大，AI 越高，越接近 compute-bound。所以 prefill 优化要看 GEMM 效率、batch、FA、BLAS/GPU，而不是只看权重字节数。

三台参照硬件的脊点（数量级）：

| 硬件 | 峰值算力（FP16，量级） | 带宽 | 脊点（FLOP/byte） |
|---|---:|---:|---:|
| M5 Pro（20 核 GPU） | 数十 TFLOPS | 307 GB/s | 数十到上百 |
| Intel 核显 | 数 TFLOPS | 90 GB/s | 数十 |
| Intel CPU（AVX-512） | 1 到 3 TFLOPS | 90 GB/s | 十几到几十 |

注意 M5 Pro 的 GPU 算力相对带宽更高，所以它的 decode 更容易被带宽打满，而 prefill 更容易吃到算力。这解释了为什么 Mac 上 pp 提升明显、tg 提升有限。

一个粗略的例子（数量级，不是精确值）：

| 场景 | 主要成本 | 优化抓手 |
|---|---|---|
| 8B Q4，单序列 decode | 读约 4.9 GB 权重 | 量化、带宽、KV |
| 8B Q4，512 token prefill | 约 8.2 TFLOP | GPU、GEMM、batch、FA |
| 32k 上下文 decode | 读权重 + 读 KV | KV 量化、FA、上下文管理 |
| 多序列 server | 权重可复用，KV 增大 | continuous batching、内存容量 |

### 2.11 实验 2：手算你的模型

1. 从启动日志或 GGUF 元数据读出 `n_layer`、`n_head_kv`、`head_dim`、`n_ctx`；
2. 用公式算每 token KV 字节数和总 KV；
3. 用模型文件大小估算 decode 理论 tok/s（文件大小 / 有效带宽）；
4. 跑 `llama-bench`，比较实测/理论；
5. 把上下文从 4k 调到 32k，观察 tg 和内存变化。

有效带宽的估算方法：取一个能完全放进内存的模型，用 `llama-bench -ngl 0` 测 tg，反推 `文件大小 / (1/tg)`。M5 Pro 上通常落在 180 到 230 GB/s，约为峰值的 60% 到 75%。

### 2.12 本章自检

1. 为什么 decode 的算术强度与参数量 P 无关？
2. GQA 如何影响 KV cache 大小？
3. prefill 和 decode 的 attention 复杂度分别是什么？
4. 量化为什么通常能提高 decode 速度，却不一定提高 prefill 速度？
5. MoE 为什么“计算少”但“内存大”？
6. Metal 和 Vulkan 的 FA 支持哪些 KV 类型？你的模型 head_dim 在 Metal 的支持列表里吗？

## 第 3 章 llama.cpp 的骨架

**本章目标**：建立对象模型和调用链。遇到问题时，知道该看哪个文件，而不是全局搜索。

**先记结论**：

- `llama_model` 是不可变权重和元数据；`llama_context` 是可变运行时状态。
- `llama_batch` / `llama_batch_ext` 是“输入描述”，不是模型输入张量本身。
- `n_batch` 是逻辑上限，`n_ubatch` 是每次图计算的物理上限。
- `llama_process` 的核心步骤：batch 规范化 -> KV slot -> 构图 -> 分配 -> 计算 -> 输出 logits。
- 图会复用；shape/topology 变化会触发重建和重新分配。

### 3.1 目录地图

| 路径 | 作用 | 主线优先级 |
|---|---|---|
| `include/llama.h` | 公开 C API、结构体、枚举 | P0 |
| `src/llama.cpp` | API 包装、后端初始化 | P0 |
| `src/llama-context.cpp` | decode/encode、图计算、输出 | P0 |
| `src/llama-model.cpp` | 模型加载、张量分配、build_graph | P0 |
| `src/models/*.cpp` | 各架构的 hparams/tensors/graph（173 个） | P0 只读 llama |
| `src/llama-graph.cpp/.h` | 通用图构建辅助（attention/norm/ffn） | P1 |
| `src/llama-batch.cpp/.h` | batch 分配、ubatch 切分 | P0 |
| `src/llama-memory.*`、`llama-kv-cache*.cpp` | KV/记忆管理 | P0 |
| `src/llama-sampler.cpp` | 采样器实现 | P1 |
| `src/llama-vocab.cpp` | tokenizer | P1 |
| `src/llama-quant.cpp` | 量化 | P1 |
| `ggml/include/ggml.h` | 张量、图、算子 | P0 |
| `ggml/include/ggml-backend.h` | 后端、buffer、scheduler | P1 |
| `ggml/src/ggml-cpu/` | CPU 内核 | P1 |
| `ggml/src/ggml-metal/` | Metal 后端（主力机） | P1 |
| `ggml/src/ggml-vulkan/` | Vulkan 后端（备用机） | P1 |
| `common/` | CLI/server 共用逻辑 | P2 |
| `tools/` | 各工具入口 | P1 |
| `app/` | 统一入口二进制 `llama` | P2 |
| `conversion/`、`gguf-py/` | HF -> GGUF 转换 | P1 |

### 3.2 对象模型

```text
llama_model
  ├─ hparams          模型超参（层数、头数、维度、RoPE、SWA...）
  ├─ vocab            tokenizer / special tokens
  ├─ tensors          权重（可能在 CPU/GPU 的不同 buffer）
  └─ arch             架构类型

llama_context
  ├─ model            引用上面的 model
  ├─ cparams          运行时参数（n_ctx/n_batch/n_ubatch/threads/FA/KV 类型）
  ├─ memory           KV cache 或其他记忆（llama_memory）
  ├─ sched            ggml backend scheduler
  ├─ backend_cpu      CPU 后端
  ├─ threadpools      计算线程池
  ├─ logits/embd      输出缓冲
  └─ samplers         可选的后端采样器
```

对应关系：

- 一个 `llama_model` 可以被多个 `llama_context` 共享（只读权重）；
- 每个 context 有自己的 KV cache、batch、输出；
- 多序列（server 的 slots）通常共享一个 context 和一份 KV cache，通过 `seq_id` 区分；
- batch 里的 `seq_id` 决定 token 属于哪个序列。

### 3.3 加载路径

高层流程：

```text
GGUF 文件
  -> gguf_init_from_file()          读元数据 + 张量索引
  -> llama_model_loader             校验、映射、拆分文件
  -> 解析 general.architecture
  -> load_arch_hparams()            读超参
  -> load_arch_tensors()            创建 ggml_tensor，绑定权重
  -> 分配 backend buffer            CPU/GPU/宿主内存
  -> mmap / read / copy             建立权重数据
  -> llama_model                   返回
```

关键点：

- GGUF 里不只有权重，还有 tokenizer、chat template、超参、量化类型等元数据；
- 默认 `load_mode=auto` 会尽量 mmap，减少加载时间和常驻内存；
- `n_gpu_layers` 决定哪些层放到 GPU；
- `split_mode` 决定多 GPU 怎么切；
- `tensor_buft_overrides` 可以把特定张量放到指定 buffer（高级用法）；
- `use_extra_bufts` 允许后端使用 repack 等额外 buffer 类型；
- `-lzm`（lazy mode）可以对超大 embedding 之类张量按需读盘。

**两台机器的加载差异，是本章最值得记住的实践知识：**

| | Metal（M5 Pro） | Vulkan（Intel） |
|---|---|---|
| 是否支持 `buffer_from_host_ptr` | 是 | 否 |
| 权重内存 | mmap 页直接给 GPU 用，一份 | 拷贝进 Vulkan buffer，加载峰值可能短暂翻倍 |
| `-lm mmap` | 默认，推荐 | 默认，加载后 mmap 页可被回收 |
| `-lm mlock` | 钉住文件页，少 swap 但吃余量 | 同上，通常不推荐 |
| 显存不足时 | 没有“显存”这回事，只有总内存 | 独显会真正 OOM；核显回到总内存 |

这就解释了一个常见困惑：为什么 Mac 上任务管理器显示的内存和 GGUF 文件大小差不多，而 Windows + Vulkan 独显上会同时看到系统内存和 VRAM 两份占用（一份是 mmap 缓存，一份是 VRAM）。

对应代码：

- `src/llama-model-loader.cpp`：GGUF 读取、split、mmap；
- `src/llama-model.cpp`：`llama_model::load_*`、`build_graph()`，以及 `buffer_from_host_ptr` 的选择逻辑；
- `src/models/<arch>.cpp`：具体架构的 hparams/tensors/graph。

### 3.4 decode 路径：一次调用发生了什么

以 `llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)` 为例（旧 API 是 `llama_decode(ctx, batch)`，内部同一条路）：

```text
llama_process
  -> llama_context::decode(const llama_batch_ext &)
     1. 校验 batch（n_tokens、embd、output 标志）
     2. balloc->init()                  规范化 batch（llama_batch_allocr）
     3. memory->init_batch()            找 KV slot，切成 ubatch
     4. 对每个 ubatch:
          a. mctx->apply()              把 token 写入 KV 位置
          b. model.build_graph(params)  构建该 ubatch 的计算图
          c. sched_alloc_graph()        分配/复用图内存
          d. set_inputs()               填 token/pos/seq_id 等输入
          e. graph_compute()            后端调度执行
     5. 收集 logits/embeddings
     6. 返回 0/1/2/-1/-2/-3
```

几个容易混淆的字段：

| 字段 | 含义 |
|---|---|
| `batch.token` | token id 数组 |
| `batch.embd` | 直接给 embedding（与 token 二选一） |
| `batch.pos` | 每个 token 的位置；NULL 则自动跟踪 |
| `batch.n_seq_id` / `seq_id` | 每个 token 属于哪些序列 |
| `batch.logits` | **是否为该 token 输出 logits**，不是 logits 本身 |
| `n_tokens` | 本 batch 的 token 数 |

`batch.logits[i] = 1` 表示要取第 i 个 token 的输出。生成时通常只关心最后一个 token；prefill 时如果只采样最后一个，也只需要最后一个。注意头文件里的注释：`logits` 为 NULL 时，非 embedding 模式只输出最后一个 token 的 logits。

**逻辑 batch 与物理 ubatch**：

- `n_batch`：一次调用最多接受多少 token；
- `n_ubatch`：一次图计算最多处理多少 token；
- 一个大 batch 会被切成多个 ubatch 依次计算，KV 写入和 logits 收集跨 ubatch 进行。

这解释了：

- 为什么 `n_batch` 很大不一定更快：还要看 `n_ubatch` 和内存；
- 为什么 prefill 的内存峰值与 `n_ubatch` 强相关（M5 Pro 上尤其明显，因为峰值直接吃统一内存）；
- 为什么 server 的 batching 受 `n_batch`/`n_ubatch` 约束。

返回码（0.9.0）：

| 返回值 | 含义 | 常见处理 |
|---|---|---|
| 0 | 成功 | 继续 |
| 1 | 找不到 KV slot | 减小 batch、增大 context、清理 KV |
| 2 | 被 abort | 检查 abort callback |
| -1 | 输入 batch 非法 | 检查 n_tokens、seq_id、token 范围 |
| -2 | 分配/准备失败 | 内存不足、上下文太大 |
| -3 | 计算失败 | 后端错误，看日志 |

### 3.5 最小 C API 循环

0.9.0 的 `examples/simple` 用的是新 API。下面这段保留同样的顺序，可以直接改造成自己的测试程序：

```cpp
llama_backend_init();

llama_model_params mp = llama_model_default_params();
mp.n_gpu_layers = 99;
llama_model * model = llama_model_load_from_file(path, mp);

const llama_vocab * vocab = llama_model_get_vocab(model);

llama_context_params cp = llama_context_default_params();
cp.n_ctx     = 4096;
cp.n_batch   = 2048;
cp.n_ubatch  = 512;
cp.no_perf   = false;              // 打开内部计时
llama_context * ctx = llama_init_from_model(model, cp);

llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

// 1. tokenize
int n = -llama_tokenize(vocab, prompt, strlen(prompt), nullptr, 0, true, true);
std::vector<llama_token> toks(n);
llama_tokenize(vocab, prompt, strlen(prompt), toks.data(), n, true, true);

// 2. 准备 batch
llama_batch_ext * batch = llama_batch_ext_init(ctx);

for (int i = 0; i < (int) toks.size(); ++i) {
    const int32_t idx = llama_batch_ext_add_token(batch, 0, toks[i]);
    const llama_pos pos = i;
    llama_batch_ext_set_pos(batch, idx, &pos);
    llama_batch_ext_set_output_logits(batch, idx, i == (int) toks.size() - 1);
}

// 3. prefill
if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) { /* error */ }

// 4. decode loop
int n_pos = toks.size();
for (int i = 0; i < n_predict; ++i) {
    llama_token id = llama_sampler_sample(smpl, ctx, -1);
    if (llama_vocab_is_eog(vocab, id)) break;

    llama_batch_ext_clear(batch);
    const int32_t idx = llama_batch_ext_add_token(batch, 0, id);
    llama_batch_ext_set_pos(batch, idx, &n_pos);
    llama_batch_ext_set_output_logits(batch, idx, true);

    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) { /* error */ }
    n_pos++;
}
```

说明：

- `llama_batch_ext_set_pos()` 接受指针，是因为多模态模型（M-RoPE）一个 token 可能需要多个位置；纯文本传一个即可；
- `llama_sampler_sample(smpl, ctx, -1)` 的 `-1` 表示取最后一个输出；
- 多序列时，每个序列的采样器和输出索引要分别管理；
- 有 encoder 的模型（如 T5 类）要先 `LLAMA_PROCESS_TYPE_ENCODE`，再用 decoder start token 进入 decode。

旧 API 的等价写法只有三行不同，读旧代码时能对上即可：

```cpp
llama_batch batch = llama_batch_get_one(toks.data(), toks.size());
llama_decode(ctx, batch);          // 等价于 llama_process(..., DECODE, ...)
```

`llama_batch_get_one()` 自动管理位置和 seq_id，适合学习和单序列；生产代码需要显式控制 `pos`、`seq_id`、`logits` 时，用 `llama_batch_init()` 或 `llama_batch_ext`。

### 3.6 改哪里：问题到文件

| 你想做的事 | 先看 |
|---|---|
| 调 CLI/server 参数 | `common/arg.cpp`、`tools/cli`、`tools/server` |
| 改 KV cache 行为 | `src/llama-kv-cache.cpp`、`src/llama-memory.*` |
| 改 batch 切分 | `src/llama-batch.cpp` |
| 改图构建 | `src/llama-graph.cpp`、`src/models/<arch>.cpp` |
| 加新模型 | `conversion/`、`gguf-py/`、`src/llama-arch.*`、`src/models/` |
| 加新算子 | `ggml/include/ggml.h`、`ggml/src/ggml.c`、各后端 |
| 加新后端 | `ggml/src/ggml-<backend>/`、`ggml-backend` 注册 |
| Metal kernel 调优 | `ggml/src/ggml-metal/kernels/`、`ggml-metal-ops.cpp` |
| Vulkan kernel 调优 | `ggml/src/ggml-vulkan/vulkan-shaders/` |
| 改量化 | `src/llama-quant.cpp`、`ggml/src/ggml-quants.c`、`ggml-common.h` |
| 改采样 | `src/llama-sampler.cpp` |
| 改 tokenizer/模板 | `src/llama-vocab.cpp`、`common/jinja` |

### 3.7 源码阅读路线（6 站）

不要从 `src/models/` 开始。按下面顺序，每站只回答一个问题：

1. **`examples/simple/simple.cpp`**：一个最小调用长什么样？
2. **`include/llama.h`**：`llama_model`、`llama_context`、`llama_batch`、`llama_sampler` 的字段是什么？
3. **`src/llama-context.cpp` 的 `decode()`**：batch 如何变成 ubatch，如何触发构图和计算？
4. **`src/llama-batch.cpp`**：`split_simple`/`split_equal`/`split_seq` 如何切？
5. **`src/llama-kv-cache.cpp`**：slot 如何分配、写入、读取、回收？
6. **`src/llama-model.cpp` + `src/models/llama.cpp`**：权重如何加载，图如何一层层搭出来？

读到第 6 站就停。其他模型架构、后端 kernel、server 细节，等有具体任务再读。

### 3.8 实验 3：让调用链可见

1. 在 `llama_context::decode` 入口打印 `n_tokens`、`n_ubatch`、`batch.n_tokens`；
2. 在 `process_ubatch` 打印 `ubatch.n_tokens`、`ubatch.n_seqs`；
3. 打开 `GGML_SCHED_DEBUG=2` 跑一次，观察 graph splits 和 backend；
4. 取消 `src/llama-context.cpp` 里 `ggml_graph_dump_dot` 的注释，生成 `llama.dot`，用 Graphviz 看一张图；
5. 用调试器在 `llama_model::build_graph` 断点，观察一次 prefill 和一次 decode 的图节点数差异。

调试器的用法：

```bash
# macOS
lldb -- ./build/bin/llama-cli -m model.gguf -p "hi" -n 4 -ngl 99
(lldb) breakpoint set -n 'llama_context::decode(llama_batch_ext const&)'
(lldb) run
(lldb) bt

# Linux
gdb --args ./build/bin/llama-cli -m model.gguf -p "hi" -n 4 -ngl 99
(gdb) break llama_context::decode
(gdb) run
(gdb) bt
```

macOS 上如果想用图形界面，在 Xcode 里打开仓库根目录，选 `llama-cli` scheme，下断点即可；Metal 的 GPU 侧要另用 Instruments（见第 5 章）。

### 3.9 本章自检

1. 为什么 `llama_model` 和 `llama_context` 要分开？
2. `batch.logits` 字段到底是什么意思？
3. `n_batch` 和 `n_ubatch` 的区别？各自影响什么？
4. `llama_process` 返回 1 时应该怎么处理？
5. 为什么图会复用？什么情况下会重建？
6. 为什么 Metal 上 mmap 的权重不需要第二份内存，而 Vulkan 上通常需要？

## 第 4 章 ggml：计算图、内存与后端

**本章目标**：理解 llama.cpp 如何把一层 Transformer 变成可在 Metal / Vulkan / CPU 上执行的图，以及“为什么某个算子会回退到 CPU”。

**先记结论**：

- ggml 是“先建图，后执行”；图里只有张量和算子，没有 Python 式的动态控制流。
- 权重、KV cache、激活值走不同的内存路径。
- 后端调度器按“哪个后端支持这个算子”切图，并在切分处插入拷贝。
- 一次图计算可能跨多个后端；切分越多，同步和拷贝越贵。
- M5 Pro 上“拷贝”往往是共享内存内的 memcpy，甚至没有拷贝，但仍然有同步成本。
- Intel 核显上 CPU 和 GPU 共享内存，独显上才是真正的 PCIe 拷贝。

### 4.1 ggml 的图模型

一个 ggml 图由 `ggml_tensor` 节点和 `src[]` 边组成。每个算子创建一个新张量，并记录输入张量：

```cpp
ggml_tensor * a = ggml_mul_mat(ctx, w, x);   // a = w @ x
ggml_tensor * b = ggml_add(ctx, a, bias);
ggml_tensor * c = ggml_silu(ctx, b);
```

建图时**不计算**，只记录依赖。最后：

```cpp
ggml_cgraph * gf = ggml_new_graph(ctx);
ggml_build_forward_expand(gf, c);
ggml_backend_sched_graph_compute_async(sched, gf);
```

推理引擎通常不会每 token 重新建整张图。llama.cpp 会缓存上一次的图，如果新 ubatch 的拓扑和输入形状不变，就复用图，只更新输入张量的数据。这就是 `n_reused` 指标（`llama_perf_context_print` 会打印 `graphs reused`）。

图不能有动态控制流。所有分支（比如不同 batch、不同序列长度）都要在构图时确定，或者拆成不同的图。这也是为什么 batch 形状变化会导致图重建/重新分配。

### 4.2 张量布局：读源码必须过的一关

`ggml_tensor` 的核心字段：

```cpp
enum ggml_type type;
int64_t ne[4];   // 每个维度的元素数
size_t  nb[4];   // 每个维度的字节步长
void *  data;
ggml_tensor * src[GGML_MAX_SRC];
```

对二维矩阵，ggml 的约定是：

- `ne[0]` = 内层维度（K），连续存放；
- `ne[1]` = 外层维度（N）；
- `nb[0]` = 元素/块字节数；
- `nb[1]` = 一行字节数。

所以 `ggml_mul_mat(w, x)` 里，`w` 的形状通常是 `{K, N}`，`x` 的形状是 `{K, M}`，输出 `{N, M}`。这与 PyTorch 的 `x @ w.T` 对应。

量化张量的 `ne[0]` 必须是 block size 的整数倍。比如 Q4_0 的 block 是 32，Q4_K 的 super-block 是 256。读取权重时不能假设“每个元素都是 float”，必须按 `ggml_type` 的 block 结构解释。

view 和 contiguous：

- `ggml_view_*` 可以改变形状/步长，不复制数据；
- 但很多算子要求特定维度连续，或者要求 `nb` 符合预期；
- 不满足时，可能需要 `ggml_cont` 复制，或者换用支持 strided 的算子；
- 转置在 ggml 里通常不是物理转置，而是改 `ne/nb`，让 kernel 按转置布局读。

这就是为什么“同一个数学公式，在不同框架里性能差很多”：布局决定 kernel 能不能高效向量化。Metal 的 GEMM kernel 和 Vulkan 的 shader 都针对 `ne[0]` 连续做了特化。

### 4.3 推理图里有哪些算子

以 Llama 的一层为例，图大致是：

```text
inpL
  -> RMSNorm
  -> QKV 投影（mul_mat）
  -> RoPE（对 Q/K）
  -> FlashAttention（或 QK^T -> softmax -> PV）
  -> Wo 投影
  -> 残差相加
  -> RMSNorm
  -> FFN gate/up（mul_mat）
  -> SiLU * up
  -> FFN down（mul_mat）
  -> 残差相加
  -> 下一层
```

常见算子：

| 算子 | 作用 |
|---|---|
| `GGML_OP_MUL_MAT` | 矩阵乘，最重要 |
| `GGML_OP_MUL_MAT_ID` | MoE 的按 expert 矩阵乘 |
| `GGML_OP_ROPE` | 旋转位置编码 |
| `GGML_OP_FLASH_ATTN_EXT` | 融合 attention |
| `GGML_OP_SOFT_MAX` | softmax |
| `GGML_OP_NORM` / `RMS_NORM` | 归一化 |
| `GGML_OP_SILU` / `GELU` | 激活 |
| `GGML_OP_GET_ROWS` | embedding 查表 |
| `GGML_OP_SET_ROWS` | 写 KV cache |
| `GGML_OP_CPY` / `CONT` | 拷贝/连续化 |
| `GGML_OP_ADD` / `MUL` | 残差、逐元素 |

llama.cpp 在 `src/llama-graph.cpp` 里封装了 `build_norm`、`build_attn`、`build_ffn` 等辅助函数，具体模型在 `src/models/<arch>.cpp` 里组合它们。

### 4.4 后端抽象：device、buffer、backend、scheduler

ggml 的后端层次：

```text
ggml_backend_reg        后端注册（CPU/CUDA/Metal/Vulkan/...）
  └─ ggml_backend_device    设备（一张 GPU 或 CPU）
       ├─ buffer_type       分配哪种内存（host/device/repack）
       └─ ggml_backend      执行流（提交图、拷贝、同步）
```

调度器 `ggml_backend_sched` 的职责：

1. 遍历图，为每个节点选择后端；
2. 如果某个后端不支持该算子，就切到支持的后端；
3. 在切分边界插入张量拷贝；
4. 分配图内存（激活值）；
5. 异步提交各段，处理事件/同步。

关键 API：

```cpp
ggml_backend_sched_new(...);
ggml_backend_sched_alloc_graph(sched, gf);
ggml_backend_sched_graph_compute_async(sched, gf);
ggml_backend_sched_synchronize(sched);
```

后端能力查询：

```cpp
ggml_backend_supports_op(backend, tensor);
ggml_backend_supports_buft(backend, buft);
ggml_backend_offload_op(backend, tensor);
```

“支持”不等于“快”。一个算子可能被支持，但走的是低效路径；也可能不支持，回退到 CPU，导致往返。`GGML_SCHED_DEBUG=2` 可以看到切分和后端分配。

### 4.5 CPU 后端：带宽、SIMD、线程

CPU 后端在 `ggml/src/ggml-cpu/`，核心工作：

- 为 `mul_mat` 选择 GEMM/GEMV 实现；
- 对量化权重做 dot product（反量化 + 乘加，或查表）；
- 用 SIMD（AVX2/AVX-512/NEON/SVE/RVV）向量化；
- 用线程池把行/列切分到多核；
- 对特定量化格式做 repack，把权重重排成更适合 SIMD 的布局；
- 可选 BLAS（Accelerate/OpenBLAS/MKL/BLIS）加速大 batch 的 GEMM。

几个实践点：

1. **线程数**：decode 用 `-t`，prefill 用 `-tb`。macOS 上默认值取自 `hw.perflevel0.physicalcpu`（最高性能簇的核数），不一定是总核数；Intel 混合核机器上默认值取物理核数，P 核和 E 核混在一起，未必最优。
2. **绑核/NUMA**：**Apple 平台上 `--cpu-mask` 是空操作**，ggml 的 `ggml_thread_apply_affinity` 在 `__APPLE__` 下直接返回成功不做事。Linux/Windows 上可用 `--cpu-mask`、`--cpu-range`、`--numa`。
3. **repack**：默认开启（`--repack`），加载时把 Q4_0/Q4_K 等重排成 SIMD 友好的块。调试时可 `--no-repack` 对比。注意 repack 只对 CPU 后端有意义，GPU 不读这个布局。
4. **BLAS**：主要提升 prefill 的大 GEMM，对 batch=1 的 decode 帮助很小。macOS 默认用 Accelerate；Linux 上 `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS`。
5. **mmap/mlock**：macOS 上权重是 mmap + host ptr 直接给 GPU，`-lm mlock` 只在你想避免页被换出时使用，代价是压缩可用余量。
6. **量化格式**：Q4_0/Q4_K 通常有优化 kernel；IQ 系列查表多，某些 CPU 上反而慢。不要只看 bpw。

Apple 平台的线程优先级是支持的（`--prio` 通过 `SCHED_FIFO`），但绑核不支持。Intel 平台上想避开 E 核，只能靠 `--cpu-mask` 手工圈 P 核。

### 4.6 Metal 后端（主力机）

#### 4.6.1 结构

```text
ggml/src/ggml-metal/
  ├─ ggml-metal.cpp          后端注册、buffer 类型、设备枚举
  ├─ ggml-metal-device.m     设备能力、pipeline、库加载、内存查询
  ├─ ggml-metal-context.m    命令缓冲、图提交、capture
  ├─ ggml-metal-ops.cpp      每个 ggml op 的 Metal 实现
  ├─ ggml-metal-fusion.cpp   算子融合
  ├─ ggml-metal-common.cpp   通用工具
  ├─ ggml-metal-tuning.cpp   离线调优表（FA vec 配置）
  └─ kernels/                .metal 源码
```

#### 4.6.2 一次 graph_compute 发生了什么

`ggml_metal_graph_compute` 大致做这些事：

1. 遍历图，统计节点，做融合判断；
2. 把节点分配到若干 command buffer（最多 8 个）；
3. 用 `dispatch_apply` 并发编码多个 command buffer；
4. 提交到 Metal queue；
5. decode 路径通常不等待完成（异步）；capture 或调试时才等待。

M5 Pro 上这条路径的重点是：GPU 和 CPU 共享内存，所以“上传权重”不是拷贝，而是把 mmap 的页映射给 GPU；“下载 logits”也只是从共享 buffer 读。真正贵的是 command buffer 数量、同步等待和 kernel 启动开销。

#### 4.6.3 三个可以关掉的优化

启动日志会打印：

```text
ggml_metal_init: use fusion         = true
ggml_metal_init: use concurrency    = true
ggml_metal_init: use graph optimize = true
```

对应的环境变量：

| 变量 | 作用 | 什么时候用 |
|---|---|---|
| `GGML_METAL_FUSION_DISABLE=1` | 关闭算子融合 | 怀疑融合 kernel 有精度/正确性问题 |
| `GGML_METAL_CONCURRENCY_DISABLE=1` | 单 command buffer 串行提交 | 怀疑并发导致竞争或调试图 |
| `GGML_METAL_GRAPH_OPTIMIZE_DISABLE=1` | 关闭图优化 | 对比“优化前”的图，或排查回退 |
| `GGML_METAL_GRAPH_DEBUG=N` | 打印图信息 | 看节点如何分组、如何融合 |
| `GGML_METAL_FUSION_DEBUG=N` | 打印融合决策 | 研究哪些算子被融合 |

另一个对 M5 很关键的开关是 Tensor API：

```text
ggml_metal_device_init: has tensor = true
```

M5/M6/A19/A20 默认启用；更老的芯片默认关闭（代码里注释说明 M2 Ultra 上反而慢约 5%，M4 上无显著差异）。想对比传统 simdgroup 路径，用 `GGML_METAL_TENSOR_DISABLE=1`；想在旧芯片上强制打开，用 `GGML_METAL_TENSOR_ENABLE=1`。

#### 4.6.4 内存：shared buffer、residency set、工作集

Metal 后端有三套内存相关机制：

- **shared buffers**：CPU 和 GPU 都能直接访问的 buffer，避免拷贝。`GGML_METAL_SHARED_BUFFERS_DISABLE=1` 可以关掉。
- **residency sets**：把常用 buffer 标记为常驻，减少驱动换入换出。`GGML_METAL_NO_RESIDENCY=1` 可以关掉。
- **working set**：`recommendedMaxWorkingSetSize`，即 0.7.2 里的预算上限。

这三者在 24GB 机器上的实际含义：**不要试图用环境变量“扩容”**。能做的只有减少权重、减少 KV、减少激活（`n_ubatch`）。

#### 4.6.5 什么时候会回退到 CPU

即使 `-ngl 99`，也不是所有算子都在 GPU 上：

- Metal 不支持的量化类型或形状（例如 FA 的 head_dim 不在列表里）；
- K/V 类型不在 Metal FA 支持列表里；
- 某些算子或形状后端不支持（例如特殊 view、低比特路径），会被留在 CPU；
- 图切分后，边界需要同步。

用 `GGML_SCHED_DEBUG=2` 看 `graph splits`。在 M5 Pro 上，理想情况是 1 个 split；出现 2 个以上就要问“是哪个算子不支持”。

### 4.7 Vulkan 后端（备用机）

#### 4.7.1 结构

```text
ggml/src/ggml-vulkan/
  ├─ ggml-vulkan.cpp          后端主体：设备、队列、pipeline、算子
  ├─ ggml-vulkan-buffers.cpp  内存分配与子分配
  ├─ ggml-vulkan-comm.cpp     多设备通信
  ├─ ggml-vulkan-debug.cpp    调试辅助
  ├─ ggml-vulkan-types.h      类型定义
  └─ vulkan-shaders/          GLSL 源码，构建期编译成 SPIR-V
```

#### 4.7.2 shader 与 matrix cores

Vulkan 后端是“构建期生成 shader”的：`vulkan-shaders-gen` 把 GLSL 编译成 SPIR-V，再编进二进制。所以：

- 构建机必须有 `glslc`；
- 换驱动通常不需要重编；
- 想改 kernel，改 `vulkan-shaders/`，重编。

设备日志里的 `matrix cores` 字段有三种值：

| 值 | 含义 | 典型设备 |
|---|---|---|
| `NV_coopmat2` / `NV_coopmat2v` | 第二代 cooperative matrix | 支持该扩展的 NVIDIA/AMD/Intel |
| `KHR_coopmat` | 第一代 cooperative matrix | 部分 Intel Arc |
| `none` | 无矩阵核心路径，走普通 shader | 较老的核显 |

Intel 上这一点很实用：`matrix cores: KHR_coopmat` 时，量化权重的 GEMM 会走 coopmat 路径，代码里还有“Intel coopmat1 偏好 f16 激活”的特化；如果是 `none`，prefill 会明显慢。

#### 4.7.3 UMA 与独显的区别

`uma: 1`（核显）：

- buffer 分配在系统内存；
- CPU 和 GPU 访问同一块内存；
- `-ngl` 不增加容量，和 Mac 一样；
- 但 Vulkan 后端没有 `buffer_from_host_ptr`，加载时仍然会有一次拷贝。

`uma: 0`（独显）：

- buffer 分配在 VRAM；
- `-ngl` 真正增加容量；
- 超出 VRAM 会 OOM 或回退到系统内存（看 `GGML_VK_ALLOW_SYSMEM_FALLBACK`）；
- 跨 PCIe 的拷贝有真实成本。

#### 4.7.4 常见回退与开关

Vulkan 的 supports_op 比 Metal 更“宽”，但不是万能。常见回退原因：

- FA 的 K/V 类型不在支持列表（见 2.4.1）；
- head_dim 不是 8 的倍数；
- 张量布局不连续，且没有对应 shader；
- 设备缺少 fp16/int dot 等特性。

调试用的环境变量（都在 `ggml-vulkan.cpp` 里）：

| 变量 | 作用 |
|---|---|
| `GGML_VK_VISIBLE_DEVICES=0,1` | 只暴露指定设备 |
| `GGML_VK_PREFER_HOST_MEMORY=1` | 优先 host-visible 内存（核显调试用） |
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | 禁用 host-visible VRAM |
| `GGML_VK_ALLOW_SYSMEM_FALLBACK=1` | VRAM 不足时允许回退系统内存 |
| `GGML_VK_DISABLE_COOPMAT=1` / `..._COOPMAT2=1` | 关闭矩阵核心路径，对比性能 |
| `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` | 关闭整数点积 |
| `GGML_VK_DISABLE_BFLOAT16=1` | 关闭 bf16 |
| `GGML_VK_DISABLE_MMVQ=1` / `GGML_VK_FORCE_MMVQ=1` | 控制量化 GEMV 路径 |
| `GGML_VK_DISABLE_FUSION=1` | 关闭算子融合 |
| `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` | 关闭图优化 |
| `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` | 允许用图形队列提交（默认只用 compute 队列） |

还有两个只在内存出问题时才用的：`GGML_VK_FORCE_MAX_ALLOCATION_SIZE`、`GGML_VK_SUBALLOCATION_BLOCK_SIZE`。

### 4.8 内存规划：权重、KV、激活

三类内存要分开算：

| 类型 | 大小 | 生命周期 | M5 Pro 上的位置 |
|---|---|---|---|
| 权重 | 模型文件大小（量化后） | 整个 context 生命周期 | 统一内存，mmap 映射 |
| KV cache | `n_ctx_seq * n_seq_max * bytes/token` | 整个 context，可复用/回收 | 统一内存 |
| 激活/临时 | 与 `n_ubatch`、图节点数相关 | 每次图计算，图分配器复用 | 统一内存 |

KV cache 的内部结构（标准 `llama_kv_cache`）：

- 每层有 K/V 两个张量，按“cell”组织；每个 cell 对应一个位置；
- 每个序列（`seq_id`）维护自己占用的一组 cell；
- `memory->init_batch()` 为新 token 找空 cell/slot，找不到就返回 1；
- 图里用 `set_rows` 把新 K/V 写进对应 cell，attention 读这些 cell；
- `llama_memory_seq_rm/cp/add/div` 支持删除、复制、平移、缩放序列位置，用于上下文复用/截断/移位；
- 多个序列可以通过 `llama_memory_seq_cp` 复制共享前缀的 cell，之后各自独立。

图分配器（graph allocator）会分析张量生命周期，复用临时 buffer。`ggml_backend_sched_alloc_graph` 在形状变化时可能重新分配；日志里的 `n_reallocs_same`/`n_reallocs_diff` 可以帮助判断是否有频繁重分配。

实践含义：

- `n_ubatch` 越大，激活峰值越高，越容易触发内存压力；
- `n_ctx` 越大，KV 越大；
- `n_parallel` 越大，KV 按序列数增加；
- 图复用可以减少分配和构图开销，但形状变化会打破复用。

想在 M5 Pro 上做容量规划，直接用工具：

```bash
# 打印“能放下的参数组合”，不真正加载
./build/bin/llama-fit-params -m model.gguf -c 32768

# 不想手动算时，交给 --fit 决定未指定的参数
./build/bin/llama-server -m model.gguf --fit on
```

`llama-server` 退出时会打印一张 memory breakdown 表（日志前缀 `llama_memory_breakdown_print`），把 model / context / compute / unaccounted 分开。这张表比任务管理器准。

### 4.9 把一个算子读到底

以 `MUL_MAT` 为例：

1. `src/models/llama.cpp` 调用 `build_ffn` / `ggml_mul_mat`；
2. `ggml.c` 创建 `GGML_OP_MUL_MAT` 节点；
3. 调度器选择后端；
4. CPU 后端根据 `src0->type` 选 vec_dot / GEMM；
5. Metal 后端按量化类型选 `kernel_mul_mm_*` / `kernel_mul_mv_*`，M5 上还可能走 tensor API；
6. Vulkan 后端按设备能力选 coopmat / 普通 shader / MMVQ；
7. kernel 读取量化块，反量化或直接整数点积，写出 FP32 输出。

建议练习：

```bash
./build/bin/test-backend-ops -o MUL_MAT
./build/bin/test-backend-ops perf -o 'MUL_MAT.*'
./build/bin/test-backend-ops -b Metal0 -o MUL_MAT      # macOS
./build/bin/test-backend-ops -b Vulkan0 -o MUL_MAT     # Intel
```

`test-backend-ops` 会对比后端结果与 CPU 参考，并测性能。它是加新算子/新后端时的第一道测试。

### 4.10 实验 4：看图和切分

1. `GGML_SCHED_DEBUG=2 llama-cli ...`，记录 splits 数量；
2. 对比 `-ngl 0`、`-ngl 10`、`-ngl 99` 的 splits（Mac 上会发现 `-ngl` 只影响切分，不影响内存）；
3. 对比 `-fa on/off` 的图节点数；
4. 在 Mac 上对比 `GGML_METAL_TENSOR_DISABLE=1` 前后的 prefill；
5. 在 Intel 上对比 `GGML_VK_DISABLE_COOPMAT=1` 前后的 prefill；
6. 跑 `test-backend-ops`，找一个“支持但慢”的量化类型。

### 4.11 本章自检

1. 为什么 ggml 的 `ne[0]` 是连续维度？这对 kernel 有什么影响？
2. 后端调度器在什么情况下插入拷贝？
3. 为什么 BLAS 主要提升 prefill，而不是 decode？
4. repack 是什么？它改变文件格式吗？
5. 部分 offload 为什么在 CUDA 上可能比全 CPU 还慢，而在 Mac 上通常不会更慢？
6. Metal 的 `has tensor` 是什么？为什么它只在 M5 及更新芯片上默认开启？
7. Vulkan 的 `uma: 1` 和 `uma: 0` 对 `-ngl` 的语义有什么影响？

## 第 5 章 性能评估与优化

**本章目标**：建立“测量 -> 假设 -> 实验 -> 结论”的闭环，能判断瓶颈在带宽、算力、KV、调度还是量化 kernel。

**先记结论**：

- 先分清 prefill 慢还是 decode 慢，再看是 CPU/GPU/带宽/内存。
- 单序列 decode 的第一性能杠杆是“减少每 token 读取的字节数”：量化、KV，然后才是 kernel。
- prefill 的第一性能杠杆是“把 GEMM 喂饱”：GPU、batch、FA、tensor/coopmat 路径。
- 多序列吞吐的第一性能杠杆是“continuous batching 和内存容量”，不是单序列 tok/s。
- M5 Pro 上，性能问题十有八九先是内存压力问题，然后才是 kernel 问题。
- 任何优化都要有 before/after 和误差范围；不要凭感觉。

### 5.1 指标体系

| 指标 | 定义 | 用途 |
|---|---|---|
| TTFT | 请求到第一个 token 的时间 | 交互体验、prefill 性能 |
| TPOT / ITL | 相邻输出 token 的平均间隔 | decode 速度、流式体验 |
| pp t/s | prompt token / prefill 时间 | 预填充吞吐 |
| tg t/s | 生成 token / decode 时间 | 解码吞吐 |
| 端到端 tok/s | 总输出 token / 总时间 | 用户感知 |
| 请求吞吐 | requests/s | 服务容量 |
| 输出吞吐 | output tokens/s | 服务容量 |
| p50/p95/p99 | 延迟分位数 | SLO、尾延迟 |
| 内存峰值 | 运行期最大占用 | 容量规划 |
| PPL/KLD/任务分 | 质量 | 量化/改造验证 |

关系：

```text
TTFT = 排队 + tokenize + prefill + 首次采样
TPOT = (总 decode 时间) / (输出 token 数 - 1)
端到端时间 = TTFT + (输出 token 数 - 1) * TPOT
```

服务端还要看 goodput：在 SLO 内完成的请求/token 数。单纯追求吞吐可能让 p99 爆炸。

### 5.2 测量协议与观测工具

一个可复现的 benchmark 至少要固定：

- 代码版本/commit、构建类型、编译选项；
- CPU/GPU 型号、系统版本、驱动版本；
- 模型、量化类型、文件大小；
- `n_ctx`、`n_batch`、`n_ubatch`、`n_parallel`；
- `-ngl`、`-fa`、KV 类型、`-t`/`-tb`；
- 输入长度、输出长度、并发数、重复次数；
- 预热次数、是否清页缓存、是否固定时钟。

两台的“环境卡”要分别记录：

| 项目 | M5 Pro | Intel + Vulkan |
|---|---|---|
| 芯片/CPU | `sysctl -n machdep.cpu.brand_string` | `lscpu` |
| 系统 | `sw_vers` | `uname -a` |
| 驱动 | 随 macOS，记录系统版本即可 | `vulkaninfo --summary` |
| 内存 | `sysctl -n hw.memsize` | `free -h` |
| 电源/散热 | 接电源、关闭低电量模式 | 关闭省电模式，注意降频 |
| 编译器 | `clang --version` | `gcc --version` 或 `clang --version` |

工具组合：

```bash
# 单序列 pp/tg
llama-bench -m model.gguf -p 512 -n 128 -r 5 -o md

# 长上下文
llama-bench -m model.gguf -p 512 -n 128 -d 8192

# 让 bench 自动找放得下的组合（Mac 上很实用）
llama-bench -m model.gguf -p 512 -n 128 -d 8192 -fitt 1024 -fitc 4096

# 多 batch 吞吐
llama-batched-bench -m model.gguf -c 16384 -b 2048 -ub 512 -npp 512 -ntg 128 -npl 1,2,4,8,16 -ngl 99

# 质量
llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw -c 512 -ngl 99
```

程序内计时：

```cpp
llama_perf_context_print(ctx);
```

它会打印 load、prompt eval、eval、reused 等数据，适合定位“时间花在加载、prefill 还是 decode”。想拿到内存分解，用 `common/fit.h` 里的 `common_memory_breakdown_print(ctx)`。

#### 5.2.1 macOS / Metal：怎么观测

macOS 没有 `nvidia-smi`，也没有 `intel_gpu_top`。主力机上的观测分四层，从便宜到贵。

**第一层：内存压力（最常用）**

```bash
vm_stat | head -10
memory_pressure
sysctl vm.swapusage
```

- `vm_stat` 里的 `Pageins`/`Pageouts` 持续增长，说明在换页；
- `memory_pressure` 的 “System-wide memory free percentage” 低于 30% 就要警惕；
- `vm.swapusage` 的 `used` 大于 0 且持续增长，tok/s 一定会掉。

这一层比任何 GPU 工具都重要。M5 Pro 上的“性能突然减半”几乎都能在这里找到证据。

**第二层：GPU 与功耗**

```bash
sudo powermetrics --samplers gpu_power,cpu_power,thermal -i 1000 -n 30
sudo powermetrics --samplers gpu_power --show-process-gpu -i 1000 -n 30
pmset -g therm
```

看三件事：GPU 频率是否掉、GPU 功耗是否接近封装上限、有没有 thermal pressure。如果频率在跑分过程中下降，先解决散热和电源，再谈 kernel。

**第三层：CPU 栈**

```bash
sample $(pgrep -n llama-cli) 10 -file /tmp/llama.sample.txt
xctrace record --template 'Time Profiler' --launch -- ./build/bin/llama-cli -m model.gguf -p hi -n 64
```

GPU 满载时 CPU 采样通常很“空”，这本身是正常信号。采样里如果大量时间在 `ggml_backend_sched_synchronize` 或 Metal 等待上，说明 GPU 是瓶颈；如果在 tokenizer、采样或拷贝上，说明瓶颈在 CPU 侧。

**第四层：GPU trace（最贵，最有用）**

两种方式：

1. **程序内 capture**（推荐，简单）：

```bash
METAL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=1 \
  ./build/bin/llama-cli -m model.gguf -p "hello" -n 8 -ngl 99
```

会在 `/tmp/perf-metal-<pid>.gputrace` 生成一个 trace，用 Xcode 打开（Open Developer Tool 里的 Metal 调试器）。`GGML_METAL_CAPTURE_COMPUTE=N` 表示第 N 次图计算时开始抓；默认值 0 表示立即抓。注意这会强制同步，测出来的时间不能当性能数字。

2. **Instruments**：

```bash
xctrace list templates
xctrace record --template 'Metal System Trace' --launch -- ./build/bin/llama-cli -m model.gguf -p hi -n 64
```

Metal System Trace 能看 command buffer 的时间线、kernel 占用、GPU 空闲间隙。看到大量空隙说明提交/同步有问题；看到某个 kernel 占满时间，就去看它的实现。

**图层面的开关**（不需要 GPU trace 也能定位）：

```bash
GGML_SCHED_DEBUG=2 GGML_METAL_GRAPH_DEBUG=1 ./build/bin/llama-cli -m model.gguf -p hi -n 8
GGML_METAL_FUSION_DEBUG=1 ./build/bin/llama-cli -m model.gguf -p hi -n 8
```

#### 5.2.2 Intel / Vulkan：怎么观测

**第一层：设备与能力**

```bash
vulkaninfo --summary
./build/bin/llama-cli --list-devices
```

确认三件事：用的是不是你想用的设备、`uma` 是 0 还是 1、`matrix cores` 是什么。很多“Vulkan 很慢”的问题，其实是选错了设备或驱动没装好。

**第二层：GPU 占用（Linux）**

```bash
sudo intel_gpu_top
cat /sys/class/drm/card*/gt_cur_freq_mhz   # i915 驱动；xe 驱动路径不同
turbostat --quiet --interval 1
```

`intel_gpu_top` 看 GPU busy、频率和功耗；`turbostat` 看 CPU 频率和温度。核显机器上两者共享功耗预算，GPU 跑满时 CPU 会降频，这是正常的。

Windows 上用任务管理器的 GPU 页、GPUView（Windows Performance Toolkit）或 Intel Arc Control 看同样的指标。

**第三层：API 层**

```bash
# 带 validation 的构建，排查描述符/同步错误
cmake -B build-vk -DGGML_VULKAN=ON -DGGML_VULKAN_VALIDATE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
./build-vk/bin/test-backend-ops -b Vulkan0 -o MUL_MAT

# 只看 loader 的问题
VK_LOADER_DEBUG=error,warn ./build/bin/llama-cli -m model.gguf -p hi -n 8
```

RenderDoc 或 gpuvis 可以抓一帧看 pipeline 和 barrier；只在怀疑同步/布局问题时才需要。

**第四层：A/B 路径对比**

Vulkan 后端有很多可选路径，最有效的诊断方法是逐个关掉：

```bash
GGML_VK_DISABLE_COOPMAT2=1 ./build/bin/llama-bench -m model.gguf -p 512 -n 128
GGML_VK_DISABLE_MMVQ=1    ./build/bin/llama-bench -m model.gguf -p 512 -n 128
GGML_VK_DISABLE_FUSION=1  ./build/bin/llama-bench -m model.gguf -p 512 -n 128
```

如果关掉某个路径后变快，说明那条路径在你的设备上不划算；如果关掉后变慢，说明它在起作用。这比读源码猜要快得多。

### 5.3 瓶颈决策树（统一内存版）

```text
性能不达预期
├─ 先看 TG（decode）
│  ├─ 模型是否完全在 GPU？（--list-devices + 日志确认）
│  │  ├─ 否 -> 是否 -ngl 太小 / 后端不支持某些层
│  │  └─ 是 -> 是否 memory-bound？
│  │     ├─ 是 -> 算“文件大小 * t/s = 有效带宽”，对比峰值
│  │     │        ├─ 有效率 < 40% -> 检查内存压力/swap、KV 太大、kernel 回退
│  │     │        └─ 有效率 40-70% -> 正常范围，换更低比特或加 batch
│  │     └─ 否 -> 检查 kernel 利用率、同步、command buffer 数量
│  └─ 是否开了不必要的 KV offload / 跨后端切分？
├─ 再看 PP（prefill）
│  ├─ 是否 compute-bound？
│  │  ├─ 是 -> 更大 ubatch / FA / tensor、coopmat 路径
│  │  └─ 否 -> 检查数据拷贝、tokenize、图切分
│  └─ 长 prompt 是否 O(T^2) 明显？
└─ 最后看服务
   ├─ 并发是否打满？
   ├─ 队列是否堆积？
   └─ p99 是否被长请求拖垮？
```

“有效带宽”是全书最实用的一个诊断量：

```text
有效带宽 GB/s = 模型文件大小 GB * tg t/s
```

8B Q4（4.9GB）跑 35 tok/s，就是 171 GB/s。M5 Pro 峰值 307 GB/s，有效率 56%，正常。跑 15 tok/s 就是 74 GB/s，有效率 24%，不正常，先查内存压力和回退。

### 5.4 macOS / Metal 优化清单

按收益排序：

1. **确认不是内存压力**：`vm_stat` 看 `Pages swapped out` 和 `Pages compressed`，`memory_pressure` 看百分比。绿色之外就开始掉性能。
2. **权重和 KV 都放得下**：24GB 的边界很硬。宁可用 Q4 而不是 Q6，先把上下文保住。
3. **FA 打开**：`-fa on`。长上下文和 KV 量化都依赖它。
4. **KV 量化**：`-ctk q8_0 -ctv q8_0`，Qwen/Llama 这类 head_dim=128 的模型都在支持列表里。省一半 KV，通常 tg 不变或略好。
5. **`n_ubatch`**：prefill 调大（512 到 1024），decode 不需要。注意内存峰值随它线性涨。
6. **确认 tensor API 生效**：日志里 `has tensor = true`。对比 `GGML_METAL_TENSOR_DISABLE=1` 的 prefill，差 10% 以上说明它在起作用。
7. **`-ngl 99` 明确写出来**：默认是 `auto`，但基准测试要可复现。
8. **线程数**：GPU 干活时 `-t` 只影响 CPU 侧算子，通常 4 到 8 足够；CPU-only 回退时才用 P 核数。
9. **电源与散热**：接电源、关低电量模式、`caffeinate -d` 防止睡眠。14 寸机身长时间满载会降频。
10. **别用 `-lm mlock` 硬扛**：它减少换出，但也压缩余量，容易从“慢”变成“OOM”。

### 5.5 Intel / Vulkan 优化清单

1. **先确认设备**：`--list-devices`、`vulkaninfo --summary`。只看到 CPU 时先修驱动。
2. **核显用 `uma: 1` 的心态调**：`-ngl` 不扩容，优化目标是 prefill 和 CPU 占用。
3. **独显用 `uma: 0` 的心态调**：`-ngl` 真正扩容，先全放，放不下再谈量化。
4. **FA**：`-fa on`，确认 K/V 类型在 Vulkan 支持列表里。
5. **KV**：q8_0 通常稳；`iq4_nl` 只有 Vulkan FA 支持，Metal 不支持，跨机器对比时注意。
6. **n_ubatch**：独显上调大（512/1024）提升 prefill；核显上注意和系统内存的竞争。
7. **matrix cores**：日志里 `KHR_coopmat` 比 `none` 快很多。没有就是没有，别指望改参数变出来。
8. **驱动**：Intel 核显在 Linux 上依赖 Mesa ANV，在 Windows 上依赖 Intel 驱动。同一张卡换驱动，性能可能差 20%。
9. **别开图形队列**：默认只用 compute 队列，`GGML_VK_ALLOW_GRAPHICS_QUEUE=1` 只在调试时用。
10. **热设计**：核显和 CPU 共享功耗预算，长时间满载会一起降频；笔记本上尤其明显。

### 5.6 KV cache 与长上下文

长上下文的成本：

```text
KV bytes = 2 * n_layer * n_head_kv * head_dim * bytes_per_elem * n_ctx_seq * n_seq_max
```

优化顺序：

1. 降 `n_ctx` 到实际需要；
2. 降 `n_parallel`（server 的每个 slot 都占 KV）；
3. KV 量化：先 `q8_0`，再考虑 `q4_0`；
4. 开 FA（量化 V 必需，0.9.0 会在 `-fa auto` 时自动打开并打印 `enabling flash_attn since it is required for quantized V cache`）；
5. 使用支持 SWA 的模型，或保持 `--swa-full` 关闭（默认）让窗口 cache 更小；
6. 前缀复用/上下文 checkpoint，避免重复 prefill；
7. 真放不下时，降 `-ngl` 在 Mac 上没用；要么降 KV，要么换更小的模型。

注意：KV 量化不是免费的。q4_0 KV 在长上下文/工具调用上可能明显掉点；q8_0 通常更稳。

在 24GB 机器上做长上下文实验时，建议直接测三组：

```bash
# 1) 32k，f16 KV
llama-bench -m model.gguf -p 512 -n 128 -d 32768 -c 32768 -fa on -ctk f16 -ctv f16

# 2) 32k，q8_0 KV
llama-bench -m model.gguf -p 512 -n 128 -d 32768 -c 32768 -fa on -ctk q8_0 -ctv q8_0

# 3) 64k，q8_0 KV
llama-bench -m model.gguf -p 512 -n 128 -d 65536 -c 65536 -fa on -ctk q8_0 -ctv q8_0
```

如果第 1 组开始 swap，第 2 组通常就正常了。这就是“在 Mac 上 KV 量化是容量手段”的实际含义。

### 5.7 Batching 与服务吞吐

server 的核心：一个 `llama_context`，多个 slot，每个 slot 是一个序列；`update_slots()` 把所有活跃 slot 的 token 拼成一个 batch，调用一次 decode。

- `-np N`：并行序列数（默认 1）；
- `-c`：总上下文预算，按 `-np` 分摊（每个 slot 约 `n_ctx / n_parallel`，按 256 对齐）；
- `--kv-unified-per-slot N`：显式给每个 slot 一个上下文上限；
- `-b`/`-ub`：逻辑/物理 batch；
- 长 prompt 会占住 slot，导致其他请求排队；
- 吞吐提升来自“权重读取被多个序列摊薄”，但 KV 和 attention 成本随序列增长。

调优：

- 先测单序列延迟，再测 1/2/4/8/16 并发的吞吐和 p99；
- 如果 p99 飙升，限制并发或做请求分片；
- 如果 GPU 利用率低，增大 `-np` 或 `-ub`；
- 如果内存吃紧，降 `-c`、`-np`、KV 量化；
- 对长 prompt 做 prefix cache/checkpoint，避免重复计算。

24GB 上的容量直觉：8B Q4 + 4 路并发 + 每路 8k 上下文，KV 约 4GB（q8_0 减半），总量仍在 12GB 以内，可以跑；如果每路 32k，KV 就是 16GB，必须减并发或减上下文。

### 5.8 投机解码：只在 decode 上赚

原理：

1. 用一个小 draft 模型（或 n-gram）快速猜 k 个 token；
2. 目标模型一次 forward 验证这 k 个 token；
3. 接受匹配的前缀，拒绝的位置用目标分布修正；
4. 接受的 token 越多，摊薄的验证成本越划算。

收益取决于：

- 接受率；
- draft 成本 / target 成本；
- k 的大小；
- 任务的可预测性（代码补全、重复文本通常更高）。

实践：

```bash
llama-server -m target.gguf -md draft.gguf --spec-type draft-simple
```

0.9.0 还支持 EAGLE-3、DFlash、DSpark、n-gram 等变体；先理解“draft-verify”，其余按需查 `docs/speculative.md`。

注意：

- 只加速 decode，不加速 prefill；
- 接受率低时会变慢；
- draft 模型也占内存。**在 24GB 的 M5 Pro 上，draft 模型往往挤掉 KV 的空间，收益可能为负**。先在 `-c 8192` 的小上下文上验证接受率，再决定要不要上。
- 采样正确性由接受/拒绝算法保证，不要自己改坏。

### 5.9 量化与速度：不是单调关系

通常：

- decode 受带宽限制，量化越小越快；
- prefill 受算力限制，量化可能让整数 kernel 更快，也可能因反量化变慢；
- Metal 上 Q4_0 通常比 K-quant 快，因为 MUL_MV 有专门的 Q4_0 路径；Q4_K_M 的质量更好，速度略慢；
- Vulkan 上 Q4_0/Q4_K 都有 shader，但 coopmat 路径只对部分类型特化；
- CPU 上 Q4_0/Q4_K 有 repack；IQ 查表多，可能更慢；
- KV 量化省带宽，但 attention kernel 不支持时会回退。

所以“Q4 一定比 Q8 快”是错的。正确做法：在你的硬件、你的上下文、你的 batch 上测。

### 5.10 三个案例

**案例 A：M5 Pro 24GB，8B Q4_K_M，tg 只有 22 t/s**

- 理论：4.9GB / 307GB/s = 16ms/token，约 63 tok/s；
- 实测 22，有效带宽约 108 GB/s，只有峰值的 35%，偏低；
- 检查顺序：`vm_stat` 是否 swap -> 上下文是否 32k 以上且 KV 为 f16 -> 是否 `-fa on` -> 是否误设了 `GGML_METAL_TENSOR_DISABLE` -> 是否在跑别的 GPU 任务；
- 常见结论：KV 从 f16 降到 q8_0，或 `-c` 从 32768 降到 8192，tg 回到 35 到 40。

**案例 B：Intel 核显，8B Q4，tg 9 t/s，pp 只有 80 t/s**

- 理论：90GB/s / 4.9GB = 18 tok/s；实测 9，有效率 50%，decode 正常；
- pp 只有 80，说明 Vulkan 没吃到 GPU：检查 `--list-devices` 是否真的选了 GPU、日志里 `uma: 1` 后面的 `matrix cores`、是否 `-ngl` 没生效、是否驱动回退到 CPU；
- 如果 pp 仍然低，试 `-ub 1024` 和 `GGML_VK_DISABLE_COOPMAT=1` 对比，确认是不是走了慢路径。

**案例 C：14B Q4，32k 上下文，Mac 上开始 swap**

- 权重约 9GB；若每 token KV 约 192 KiB，32k 就是 6GB；加激活和系统，超过 Metal 工作集；
- 处理：`-ctk q8_0 -ctv q8_0`（KV 减半到 3GB）、`-c 16384`、减少后台应用；
- 仍不够就换 8B 或 Q3_K_M。不要在 Mac 上用“降 `-ngl`”解决容量问题。

### 5.11 报告模板

```text
日期/commit:
硬件: 芯片/CPU/GPU/内存/系统/驱动
构建: Release, 后端, 编译选项
模型/量化:
运行参数: n_ctx, n_batch, n_ubatch, ngl, fa, kv type, threads, np
测试: pp512, tg128, depth, concurrency, repetitions
结果: 均值/标准差/p50/p99, 内存峰值
有效带宽: 文件大小 * tg t/s, 占峰值百分比
质量: PPL/KLD/任务抽检
观察:
假设:
改动:
改动后结果:
结论/风险:
```

### 5.12 常见测量陷阱

- 用 Debug 构建；
- 第一次运行就当基线（下载、缺页、初始化、shader 加载）；
- 不固定上下文深度，长上下文和短上下文混比；
- 只报平均，不报方差和 p99；
- 同时跑其他 GPU 任务；
- 把 tokenization/sampling 算进 `llama-bench`；
- 比较不同量化、不同模型、不同 tokenizer 的 PPL；
- 跨模型直接比 tok/s，忽略 tokenizer 差异（不同 tokenizer 的 token 不等价）；
- 忽略温度墙/降频（Mac 上尤其常见）；
- 用合成 prompt 代表真实业务；
- 在 Mac 上开着 Chrome 跑 benchmark，然后怪 llama.cpp。

### 5.13 实验 5：优化挑战

选一个模型，完成：

1. 基线：M5 Pro 和 Intel 各一组 pp/tg；
2. 找理论 decode 上限，算有效带宽占比；
3. 做 5 组单变量实验（线程、ngl、fa、kv、ubatch）；
4. 画一张表，标出每组的变化和峰值内存；
5. 写一页结论：瓶颈是什么，下一步做什么。

### 5.14 本章自检

1. TTFT 和 TPOT 分别由什么决定？
2. 为什么“全量 offload”在 Mac 上仍然是正确的默认，但理由和 CUDA 不同？
3. 为什么增大 `-np` 能提高吞吐但可能恶化 p99？
4. 什么情况下投机解码会变慢？
5. 为什么 KV 量化在长上下文上收益更大？
6. 你如何证明瓶颈是带宽而不是算力？
7. Mac 上性能突然掉一半，先看哪三个命令？

## 第 6 章 量化：把权重压小，而不把模型压坏

**本章目标**：理解 block 量化的基本数学，能选型、能跑量化流程、能用 PPL/KLD 和任务抽检验证，不把“小”当成“好”。

**先记结论**：

- 量化主要节省**权重字节数**，因此最直接收益在 decode（memory-bound）。
- 同一 bpw 下，block 越大、scale 越精细、校准越好，质量通常越好。
- Q4_K_M 是通用默认；Q5_K_M/Q6_K 用于质量敏感；Q8_0 接近无损；IQ 系列用于极限压缩。
- imatrix 用激活重要性指导量化，低比特时收益更明显。
- KV 量化是另一条线：省的是上下文内存，不是权重。
- 量化是否更快取决于 kernel，不只看 bpw。Metal 和 Vulkan 的支持矩阵不一样。

### 6.1 量化的基本数学

最朴素的对称块量化：

```text
对于一个 block 的权重 w[0..B-1]：
  scale = max(abs(w)) / qmax
  q[i] = round(w[i] / scale)      // 整数
  反量化：w'[i] = q[i] * scale
```

非对称量化多一个 zero-point/min：

```text
w'[i] = q[i] * scale + min
```

block 越小，scale 越贴近局部数据，精度越高，但 scale 的存储开销也越大。block 越大，压缩率越高，但异常值会抬高整个 block 的 scale，伤害精度。

这就是量化格式设计的核心权衡：**bit 数、block 大小、scale 精度、是否用 codebook、是否用校准数据**。

### 6.2 ggml 的主要格式

0.9.0 的 `ggml_type` 里，推理常用的有：

| 格式 | block | 典型 bpw | 特点 | 适用 |
|---|---|---:|---|---|
| F32 | - | 32 | 基准 | 调试/参考 |
| F16/BF16 | - | 16 | 半精度 | 转换中间格式、小模型 |
| Q4_0 | 32 | 4.5 | fp16 scale + 4-bit | CPU/Metal 快、通用 |
| Q4_1 | 32 | 5.0 | scale + min | 比 Q4_0 稍好 |
| Q5_0/Q5_1 | 32 | 5.5/6.0 | 5-bit | 质量优先 |
| Q8_0 | 32 | 8.5 | 8-bit | 接近无损、GPU 快 |
| Q2_K..Q6_K | 256 | 2.5-6.6 | 超块 + 子块 scale | K-quants 主力 |
| IQ1/IQ2/IQ3/IQ4_XS | 256 | 1.5-4.5 | codebook/查表 | 极限压缩 |
| TQ1_0/TQ2_0 | - | 约 1.7/2.1 | 三值 | 特殊场景 |
| MXFP4 | 32 | 约 4.25 | 新硬件浮点块格式 | Metal、支持 FP4 的 Vulkan |
| NVFP4 | 64（4 个 16 元素子块） | 约 4.5 | E4M3 scale 的 FP4 | Metal、支持 FP4 的 Vulkan |
| Q1_0/Q2_0 | 128/64 | 约 1.1/2.25 | 新极低比特 | 实验 |
| F8_E4M3 | - | 8 | FP8 | 新硬件 |

几个常见名字的含义：

- **Q4_K_M**：K-quant 的 4-bit 混合版，部分敏感张量用更高比特；“M”是 medium。
- **Q4_K_S**：更小、更激进；“S”是 small。
- **Q5_K_M**：5-bit 混合，质量更好，体积更大。
- **Q6_K**：6-bit，通常接近无损。
- **Q8_0**：8-bit，常作为“量化基线”。
- **IQ**：importance-aware quantization，使用 codebook/网格，低比特下比朴素 round 更好，但 kernel 可能更慢。
- **MXFP4/NVFP4**：硬件浮点块格式，只有在后端和硬件都支持时才快，否则会回退甚至无法加载。

注意：bpw 是平均值。embedding/output 等张量可能保持更高精度，所以文件实际大小和“理论 bpw x 参数量”会有偏差。IQ4_NL 是例外，它的 block 是 32（不是 256），在 Metal/Vulkan 的 KV 和部分权重路径上更常见。

### 6.3 为什么 K-quants 在相同 bpw 下更好

Q4_0 是“每 32 个权重一个 fp16 scale”。Q4_K 是“每 256 个权重一个 super-block，内部再分 8 个子块，每个子块有自己的 6-bit scale/min”。

好处：

- super-block 共享更粗的 scale，减少开销；
- 子块 scale 适应局部变化；
- 对异常值更鲁棒；
- 在 4-bit 附近通常明显优于 Q4_0。

代价：

- 解包/反量化更复杂；
- kernel 实现更难；
- 某些硬件上不如 Q4_0 快。Metal 就是典型：Q4_0 的 GEMV 路径更成熟。

I-quants 更进一步：不只存 scale + 整数，还用 codebook/网格逼近，把多个权重联合编码。它在 2-3 bit 附近优势明显，但查表和 kernel 开销更大。Intel 核显上 IQ 系列经常比 Q4_0 慢，因为查表在 shader 里代价高。

### 6.4 完整量化流程

以 Hugging Face 模型为例：

```bash
# 0. 依赖（Python 侧）
pip install -r requirements.txt

# 1. 转 GGUF（bf16/f16 作为量化输入）
python convert_hf_to_gguf.py /path/to/hf-model --outtype bf16 --outfile model-bf16.gguf

# 2. 可选：采集重要性矩阵
llama-imatrix -m model-bf16.gguf -f calibration.txt -o imatrix.gguf -ngl 99

# 3. 量化
llama-quantize --imatrix imatrix.gguf model-bf16.gguf model-Q4_K_M.gguf Q4_K_M

# 4. 跑起来
llama-cli -m model-Q4_K_M.gguf -p "hello" -n 64 -ngl 99

# 5. 评估
llama-perplexity -m model-Q4_K_M.gguf -f wikitext-2-raw/wiki.test.raw -c 512 -ngl 99
```

在 M5 Pro 上，第 2 步（imatrix）会用到 Metal，明显比纯 CPU 快；第 3 步（quantize）主要跑 CPU。24GB 内存适合处理 8B 级的 bf16 输入（约 16GB）；14B 以上要用分片输入（`--keep-split`）或换更大内存的机器。

常用选项：

| 选项 | 作用 |
|---|---|
| `--imatrix FILE` | 使用重要性矩阵 |
| `--include-weights` / `--exclude-weights` | 只对某些张量使用 imatrix（不能混用） |
| `--output-tensor-type` | 指定 output 张量类型 |
| `--token-embedding-type` | 指定 token embedding 类型 |
| `--tensor-type regex=type` | 按张量名精细覆盖，可多次指定 |
| `--pure` | 所有张量同一类型，不做混合 |
| `--allow-requantize` | 允许对已量化张量再量化（一般不推荐） |
| `--leave-output-tensor` | 不量化 output，提升质量但变大 |
| `--prune-layers` | 删层（实验性，慎用） |
| `--dry-run` | 只估算大小，不写文件 |
| `--keep-split` | 保持分片输出 |

### 6.5 怎么选

| 场景 | 建议 |
|---|---|
| 通用聊天/写作，内存有限 | Q4_K_M |
| 代码、数学、长上下文，质量敏感 | Q5_K_M 或 Q6_K |
| 需要接近无损的基线 | Q8_0 |
| 内存极度受限（24GB 跑 14B 以上） | Q3_K_M、IQ3、IQ2（先评估质量） |
| Metal，追求 decode 速度 | Q4_0 往往最快，质量比 Q4_K_M 略差 |
| Intel Vulkan 核显 | Q4_0/Q4_K_M，避免 IQ 查表 |
| 支持 FP4 的硬件 | MXFP4/NVFP4（先确认日志里 fp4: 1） |
| 只做 embedding/分类 | 按任务选，Q8_0 通常够 |
| 小模型（小于 3B） | 不要压到 Q2/Q3，伤害更明显 |

经验：

- 7B/8B 模型，Q4_K_M 通常是质量/体积/速度的甜点；
- 14B 在 24GB 上，Q4_K_M + q8_0 KV 是可持续的组合；
- 30B 级稠密模型放不下，别硬试；
- 小模型对量化更敏感；
- 多模态 projector、embedding、output 往往保持较高精度；
- 量化后一定要做任务抽检，不能只看 PPL。

### 6.6 imatrix：用激活重要性指导量化

普通量化只看权重本身。imatrix 额外统计校准数据上每个张量/通道的激活重要性，让量化器在重要位置保留更多精度。

流程：

```bash
llama-imatrix -m model-bf16.gguf -f calibration.txt -o imatrix.gguf -ngl 99
llama-quantize --imatrix imatrix.gguf model-bf16.gguf model-Q4_K_M.gguf Q4_K_M
```

要点：

- 校准数据应接近目标业务（通用文本、代码、对话混合）；
- 不是越多越好，几百万 token 通常足够；
- 低比特（Q2/Q3/IQ）收益更明显；
- Q4_K_M 上也可能有小幅提升；
- output/embedding 默认不参与 imatrix 统计，避免副作用；
- imatrix 是“辅助”，不能把 Q2 变成 Q6。

### 6.7 KV cache 量化

KV 量化的目标是长上下文内存：

```bash
llama-cli -m model.gguf -fa on -ctk q8_0 -ctv q8_0 -c 32768
```

- `-ctk`：K cache 类型；
- `-ctv`：V cache 类型；
- 量化 V 需要 FlashAttention 开启（0.9.0 在 `-fa auto` 下会自动开启并打印提示）；
- q8_0 通常接近无损，q4_0 更激进；
- K 和 V 可以不同；一般先都 q8_0，再尝试 q4_0；
- 长上下文、工具调用、代码任务对 KV 量化更敏感，务必抽检。

**支持矩阵（0.9.0 源码）**：

| KV 类型 | Metal | Vulkan |
|---|---|---|
| f16/f32 | 支持 | 支持 |
| bf16 | 需要 bfloat | 支持 |
| q8_0 | 支持 | 支持 |
| q4_0/q4_1/q5_0/q5_1 | 支持 | 支持 |
| iq4_nl | 不支持 | 支持 |

收益估算：f16 到 q8_0 约减半，f16 到 q4_0 约四分之一，但还有对齐和 kernel 开销。

### 6.8 质量评估

| 指标 | 看什么 | 局限 |
|---|---|---|
| PPL | 语言建模损失 | 对聊天质量不敏感，不能跨 tokenizer 比较 |
| KLD | 与 FP16 的 logits 分布差异 | 需要保存大 logits 文件，计算成本高 |
| 任务评测 | 代码/数学/问答/工具调用 | 覆盖有限，需固定 prompt |
| 人工抽检 | 真实业务 | 主观、样本小 |
| 长上下文测试 | 32k/128k 检索、摘要 | 最容易暴露 KV 量化问题 |

建议组合：PPL + KLD + 一组固定任务 prompt + 长上下文抽检。

在两台机器上跑同一份量化模型，PPL 应该非常接近（通常差异在 0.1% 以内）；只有同一后端、同一实现才会逐位一致。如果差异明显，先怀疑后端 kernel 有 bug，而不是“量化的随机性”。

### 6.9 量化与硬件

同一个量化文件，不同后端速度不同：

- CPU：Q4_0/Q4_K 有 repack + SIMD dot；IQ 查表多，可能慢；
- Metal：Q4_0 的 GEMV 路径成熟；K-quants 支持但未必最快；MXFP4/NVFP4 在 0.9.0 已支持；
- Vulkan：支持矩阵较宽，但 coopmat 只对部分类型特化；IQ4_NL 只在 Vulkan 的 FA 里受支持；
- 新格式（MXFP4/NVFP4/FP8）：需要硬件和 kernel 支持，否则回退。

查支持矩阵的两个办法：

```bash
# 1. 仓库里生成的算子支持表
less docs/ops/Metal.csv
less docs/ops/Vulkan.csv

# 2. 直接测
./build/bin/test-backend-ops -b Metal0 -o MUL_MAT
./build/bin/test-backend-ops -b Vulkan0 -o MUL_MAT
./build/bin/test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT
```

所以选型顺序是：

1. 先确定质量底线；
2. 再在目标硬件上测 pp/tg；
3. 最后看内存/带宽收益。

### 6.10 实验 6：量化矩阵

选一个 1B-8B 模型，做：

```bash
for q in Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M; do
  llama-quantize model-bf16.gguf model-$q.gguf $q
  llama-bench -m model-$q.gguf -p 512 -n 128 -ngl 99 -o md
  llama-perplexity -m model-$q.gguf -f wikitext-2-raw/wiki.test.raw -c 512 -ngl 99
done
```

记录：

| 量化 | 文件大小 | pp512 | tg128 | PPL | 备注 |
|---|---:|---:|---:|---:|---|
| Q8_0 |  |  |  |  |  |
| Q6_K |  |  |  |  |  |
| Q5_K_M |  |  |  |  |  |
| Q4_K_M |  |  |  |  |  |
| Q3_K_M |  |  |  |  |  |

再加三组对比：

1. Q4_0 与 Q4_K_M（Metal 上通常 Q4_0 的 tg 更快）；
2. 加 imatrix 的 Q4_K_M；
3. 32k 上下文下 KV f16 与 q8_0。

### 6.11 常见误区

- “bpw 越低越快”：不一定，kernel 支持决定实际速度。
- “PPL 没掉就是没坏”：聊天/工具调用可能已经退化。
- “Q4_K_M 可以无脑用”：小模型、长上下文、代码任务要单独验证。
- “imatrix 能救 Q2”：只能缓解，不能逆转信息损失。
- “KV 量化免费”：q4_0 KV 可能明显掉点。
- “量化后模型大小 = 参数量 x bpw / 8”：还有 embedding/output、对齐、元数据。
- “拿不同 tokenizer 的 PPL 比较”：没有意义。
- “Mac 上 Q4_K_M 一定比 Q4_0 慢所以别用”：质量优先时仍然值得，先测再决定。

### 6.12 本章自检

1. Q4_0 和 Q4_K 的 block 结构差异？为什么同 bpw 下质量不同？
2. imatrix 统计的是什么？为什么能提升低比特质量？
3. 为什么量化 V cache 需要 FA？
4. 量化为什么通常加速 decode，却不一定加速 prefill？
5. 你如何判断一个量化模型“可以上线”？
6. Metal 和 Vulkan 的 KV 量化支持差异，哪些组合会静默回退？

## 第 7 章 适配：模型、后端、服务

**本章目标**：知道“接一个新模型/新硬件/新服务需求”分别要动哪些层，能按流程推进并验证。

**先记结论**：

- 模型适配 = 转换器 + GGUF 元数据/张量命名 + C++ 架构 + 图 + tokenizer/模板 + 测试。
- 后端适配 = 设备/缓冲区/执行流 + 算子支持 + kernel + 正确性/性能测试。
- 大多数“模型效果不对”不是引擎 bug，而是 tokenizer、chat template、BOS/EOS、量化或 prompt 格式。
- 先让模型“跑对”，再让它“跑快”。
- 你手上正好有两种性质相反的后端：Metal（统一内存、命令缓冲、融合）和 Vulkan（显式资源、shader、跨厂商）。两个都碰过，后端适配的通用模型就建立起来了。

### 7.1 三种适配

| 类型 | 目标 | 主要工作 |
|---|---|---|
| 模型适配 | 让新架构能加载、推理 | 转换脚本、GGUF、C++ arch、graph、tokenizer |
| 硬件适配 | 让新后端能执行 | backend 注册、buffer、算子、kernel、测试 |
| 服务适配 | 让推理接入业务 | server、batching、API、缓存、监控 |

### 7.2 新模型接入全链路

以 Hugging Face 上的一个新 causal LM 为例：

```text
HF config/tokenizer/safetensors
  -> conversion/<model>.py        读配置、映射张量名、写 GGUF
  -> gguf-py/gguf/constants.py    MODEL_ARCH / MODEL_TENSORS
  -> gguf-py/gguf/tensor_mapping.py
  -> model.gguf
  -> src/llama-arch.h/.cpp        LLM_ARCH_*、名称映射
  -> src/models/<arch>.cpp        hparams、tensors、graph
  -> src/llama-model.cpp          注册架构、RoPE 类型等
  -> tokenizer/chat template
  -> 测试：加载、logits、PPL、量化、各后端
```

#### 7.2.1 转换器

在 `conversion/` 下新增模型类，注册 HF 的 `architectures`：

```python
@ModelBase.register("MyModelForCausalLM")
class MyModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MYMODEL
```

需要处理：

- `set_gguf_parameters`：n_layer/n_head/n_embd/rope/attention 等；
- `set_vocab`：tokenizer 类型、special tokens；
- `modify_tensors`：张量名/形状/转置；
- 必要时处理 tied embeddings、QKV 融合、MoE 重排。

张量名要遵循 GGUF 约定，通常以 `.weight`/`.bias` 结尾；量化工具依赖这个约定。

#### 7.2.2 GGUF 常量

在 `gguf-py/gguf/constants.py`：

- `MODEL_ARCH` 加枚举；
- `MODEL_ARCH_NAMES` 加字符串；
- `MODEL_TENSORS` 列出该架构的张量；
- 在 `tensor_mapping.py` 增加 HF 名到 GGUF 名的映射。

架构字符串一旦发布就不要改，否则社区已有的 GGUF 会失效。

#### 7.2.3 C++ 侧

- `src/llama-arch.h`：加 `LLM_ARCH_*`；
- `src/llama-arch.cpp`：名称映射、KV/tensor 名称；
- `src/llama-model-loader.cpp`：非标准元数据读取；
- `src/models/<arch>.cpp`：
  - `load_arch_hparams()`：读超参；
  - `load_arch_tensors()`：创建张量、绑定权重；
  - `build_arch_graph()`：搭计算图；
- `src/llama-model.cpp`：架构到模型类的映射、RoPE 类型等；
- 检查 `src/llama-model-saver.cpp` 等遍历 `LLM_ARCH_*` 的地方。

图实现的写法：参考 `src/models/llama.cpp`、`qwen3.cpp`、`gemma3.cpp`。不要从最复杂的 MoE/多模态模型开始。

#### 7.2.4 Tokenizer 与 chat template

- tokenizer 类型：SPM/BPE/WPM/UGM；
- BOS/EOS/EOT/PAD 等 special tokens；
- chat template：Jinja 字符串，放在 GGUF 元数据；
- 工具调用/推理模型可能有额外模板和 parser。

0.9.0 的 server 默认使用 Jinja 模板引擎（`--jinja` 默认开）。旧教程里“必须加 `--jinja`”的说法已经过时，但模板本身仍然可能有问题。

大多数“模型答非所问/不停止/输出特殊 token”的问题，先查这里。

#### 7.2.5 正确性验证

1. 加载：`llama-cli -m model.gguf -p "hello" -n 1`；
2. logits 对比：用相同 prompt，与 HF/PyTorch 的 logits 比对；
3. `llama-eval-callback`：逐算子打印/对比中间结果；
4. PPL：与参考实现对比；
5. 量化：Q8_0/Q4_K_M 是否能跑；
6. 后端：Metal 和 Vulkan 至少各跑一次；
7. 长上下文/多轮/工具调用按模型能力抽检。

第 6 步的输出要能解释：同一 GGUF 在 Metal 和 Vulkan 上的 logits 在前几个 token 应该几乎一致；如果差很多，通常是某个算子在某个后端上回退了，用 `test-backend-ops` 缩小范围。

### 7.3 新后端/新算子适配

后端接口的核心：

```text
ggml_backend_reg        注册后端
ggml_backend_device     设备与能力
ggml_backend_buffer_type 分配内存
ggml_backend            提交图、拷贝、同步
```

一个后端至少要：

- 枚举设备、报告内存；
- 分配/释放 buffer；
- 实现 `graph_compute`；
- 对每个算子实现 `supports_op`；
- 不支持时允许回退；
- 处理 host <-> device 拷贝；
- 通过 `test-backend-ops`。

加一个新算子：

1. `ggml.h` 加 op 枚举/参数；
2. `ggml.c` 加形状推断/前向；
3. CPU 参考实现；
4. 各后端 kernel；
5. `test-backend-ops` 覆盖类型/形状；
6. 性能测试（`test-backend-ops perf`）。

#### 7.3.1 在 Metal 上加 kernel

步骤：

1. 在 `ggml/src/ggml-metal/kernels/` 加 `.metal` 实现；
2. 在 `ggml-metal-ops.cpp` 里接上 op 分发；
3. 在 `ggml-metal-device.m` 的 `supports_op` 里声明支持条件（类型、dims、连续性）；
4. 在 `ggml-metal-device.m` 里注册 pipeline（如果新 kernel 需要参数）；
5. 重新构建（shader 会重新编进二进制）；
6. `test-backend-ops -b Metal0 -o <OP>` 验证。

注意事项：

- `supports_op` 写得太宽会算错，写得太窄会回退到 CPU，两者都要避免；
- M5 上可以走 tensor API（`mpp::tensor_ops`），但需要 Metal 4 和 Apple GPU family；
- 改完先关掉 fusion 测（`GGML_METAL_FUSION_DISABLE=1`），排除融合干扰。

进阶工具是 `ggml-metal-tuning`：它是离线 kernel 配置扫描器，目前只调 FA vec 的 `(Q, NE)` 配置，按 Apple GPU family 生成可粘贴的表格行。用法见 `tools/tuning/README.md`。这是一个很好的“性能工程”练习，但耗时以小时计。

#### 7.3.2 在 Vulkan 上加 kernel

步骤：

1. 在 `ggml/src/ggml-vulkan/vulkan-shaders/` 写 GLSL（可以用 `.comp` 模板 + 生成器）；
2. 在 `vulkan-shaders/CMakeLists.txt` 里注册 shader；
3. 在 `ggml-vulkan.cpp` 里加 pipeline 创建和 op 分发；
4. 在 `ggml_backend_vk_device_supports_op` 里声明支持条件；
5. 重新构建（`glslc` 会编译新 shader）；
6. `test-backend-ops -b Vulkan0 -o <OP>` 验证。

调试建议：

```bash
cmake -B build-vk-debug -DGGML_VULKAN=ON -DGGML_VULKAN_VALIDATE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
./build-vk-debug/bin/test-backend-ops -b Vulkan0 -o MUL_MAT
```

validation layer 会抓住描述符/内存屏障错误，这是 Vulkan 开发最容易出错的地方。性能问题另用 RenderDoc 或 gpuvis 看。

注意事项：

- 先保证正确性，再优化；
- 量化类型 x 形状组合很多，测试要覆盖；
- 不支持的组合要明确回退，不要静默算错；
- 大改要先和社区讨论（llama.cpp 对复杂度很敏感）。

### 7.4 多模态、LoRA、Control Vector（支线）

- **多模态**：`mtmd` 把图像/音频编码器和 projector 作为单独 GGUF（mmproj），LLM 部分不变；主线只需知道“编码器输出 embedding，再喂给 LLM”。
- **LoRA**：在权重旁加低秩增量；llama.cpp 支持 `llama_adapter_lora`，适合多任务切换；量化模型上使用 LoRA 需要注意精度。
- **Control Vector**：在激活层加方向向量，改变风格/行为；属于实验性功能。

这些都不影响主线：权重、KV、图、后端。

### 7.5 服务适配

- `llama-server` 提供 OpenAI 兼容 API；
- 一个 context + 多个 slot，continuous batching；
- `-np` 控制并发，`-c` 控制总上下文；
- 默认监听 `127.0.0.1:9931`；
- 支持 prompt cache/checkpoint、流式输出、多模态；
- 常用观测端点：`/health`、`/props`、`/slots`、`/metrics`（需要 `--metrics`）；
- router 模式可管理多个模型实例；
- 生产环境还要考虑鉴权、限流、超时、监控、优雅退出。

服务层不是“推理核心”，但会直接影响 p99 和容量规划。在 24GB 机器上，服务端的第一个约束不是 QPS，而是“并发数 x 每路上下文”的 KV 预算。

### 7.6 测试与回归

| 测试 | 用途 |
|---|---|
| `test-backend-ops` | 算子正确性/性能 |
| `test-llama-archs` | 架构枚举/元数据一致性 |
| `test-tokenizer-0` | tokenizer |
| `test-chat-template` | chat template |
| `test-quantize-fns` | 量化函数 |
| `test-llama-context` | 上下文/KV 行为（`tests/llama-context/`） |
| `llama-perplexity` | 质量回归 |
| `llama-bench` | 性能回归 |
| `test-model-load-cancel` | 加载取消/健壮性 |

改完代码至少跑：

```bash
cmake --build build -j
./build/bin/test-backend-ops -o MUL_MAT
./build/bin/test-tokenizer-0 models/ggml-vocab-llama-spm.gguf
./build/bin/llama-bench -m model.gguf -p 512 -n 128
```

在两台机器上跑同一套。Metal 上再补一次 `-b Metal0`，Vulkan 上补 `-b Vulkan0`。同一份代码在两个后端上都过，才叫“后端无关的正确性”。

### 7.7 职业路线映射

| 层级 | 能力 | 证据 |
|---|---|---|
| L1 使用/部署 | 构建、跑模型、server、bench、排障 | 两张机器的基线表、部署脚本 |
| L2 性能 | roofline、profiling、CPU/GPU 调优、量化选型 | 优化报告（before/after） |
| L3 原理/源码 | 图、KV、batch、scheduler、sampler | 调用链笔记、图 dump |
| L4 量化/适配 | 量化矩阵、imatrix、新模型、tokenizer | 量化报告、模型适配记录 |
| L5 后端/内核/架构 | 新算子/后端、kernel 优化、分布式/服务 | kernel 改动、吞吐/延迟提升 |

不要跳过 L1/L2 直接做 L5。推理引擎的“高手”不是会背源码，而是能用数据定位瓶颈、用最小改动解决问题。

### 7.8 12 周计划（每周 10 到 15 小时）

| 周 | 内容 | 产出 |
|---|---|---|
| 1-2 | 第 0-1 章：两台机器构建、跑通、bench | 两张基线表 |
| 3-4 | 第 2 章：Transformer 数学、KV/roofline | 手算报告 + 最小 C 程序 |
| 5-6 | 第 3 章：对象模型、decode 调用链 | 调用链笔记 + 图 dump |
| 7-8 | 第 4 章：ggml 图、Metal 与 Vulkan 后端 | 切分实验 + test-backend-ops |
| 9-10 | 第 5 章：性能工程 | 一页优化报告 |
| 11 | 第 6 章：量化、imatrix、KV 量化 | 量化矩阵 + 质量抽检 |
| 12 | 第 7 章：选一个小模型/算子/服务项目 | 可复现的项目报告 |

如果只有 4 周：只做第 0-3 章 + 第 5 章基础 + 第 6 章选型。

### 7.9 毕业项目（选 2 个）

1. **性能基线**：同一模型在 Metal 和 Vulkan、3 种量化、2 种上下文下的 pp/tg/内存/质量矩阵。
2. **最小推理程序**：C/C++ 实现 tokenize -> prefill -> decode -> stream，支持 KV 复用和多轮。
3. **量化研究**：Q8/Q6/Q5/Q4/Q3 + imatrix + KLD + 长上下文抽检，给出上线建议。
4. **模型适配**：接一个 tiny HF 模型，完成转换、C++ 图、tokenizer、测试。
5. **算子/后端**：给 CPU 后端加一个缺失算子，或在 Vulkan 上实现一个缺失的量化类型 kernel，跑正确性/性能测试。
6. **服务容量**：server 的并发/上下文/KV 量化组合，测吞吐、p99、内存边界，写容量规划。
7. **后端对比报告**：同一模型在 Metal 与 Vulkan 上的切分、内存、kernel 路径差异，解释性能差距的来源。

### 7.10 反模式

- 一上来读 173 个模型实现；
- 不看日志，直接改代码；
- 没有 baseline 就“优化”；
- 比较不同上下文/量化/模型的数字；
- 把 PPL 当聊天质量；
- 认为量化越狠越快；
- 忽略 tokenizer/chat template；
- 追最新模型/特性，却不理解数据流；
- 改超过自己能 review 的代码量；
- 在没跑测试的情况下提交 PR；
- 把 CUDA 机器的调参经验直接搬到 Mac 或 Intel 核显上。

### 7.11 本章自检

1. 新模型接入最少要改哪几个文件？
2. 为什么 tokenizer/chat template 错误会让模型“看起来变笨”？
3. 新后端如何处理不支持的算子？
4. 量化后如何验证“能上线”？
5. 你现在的岗位更接近 L1-L5 的哪一层？下一步证据是什么？
6. 同一个模型在 Metal 和 Vulkan 上 logits 不一致，你会怎么定位？

---

## 附录 A：术语表

| 术语 | 含义 |
|---|---|
| Prefill / pp | 处理 prompt，一次算多个 token，通常 compute-bound |
| Decode / tg | 逐 token 生成，通常 memory-bound |
| TTFT | Time To First Token，请求到首 token |
| TPOT / ITL | Time Per Output Token / Inter-Token Latency |
| KV cache | 缓存历史 K/V，避免 decode 重算 |
| GQA / MQA | Grouped/Multi Query Attention，多个 query head 共享 K/V |
| RoPE | 旋转位置编码 |
| MoE | Mixture of Experts，稀疏激活 FFN |
| SWA | Sliding Window Attention，只看最近窗口 |
| MLA / DSA | 压缩/稀疏 KV 的 attention 变体 |
| GGUF | llama.cpp 的模型文件格式（元数据 + 张量） |
| ggml | 张量/计算图/后端库 |
| Graph split | 调度器按后端能力切图，切分处插入拷贝 |
| Offload | 把权重/KV/算子放到 GPU 或其他设备 |
| Repack | 加载时把量化权重重排成 SIMD 友好的布局（仅 CPU 后端） |
| bpw | bits per weight，平均每权重比特数 |
| imatrix | 重要性矩阵，用激活统计指导量化 |
| PPL | Perplexity，困惑度 |
| KLD | KL Divergence，与参考模型 logits 分布的差异 |
| Q4_K_M | 4-bit K-quant medium 混合量化 |
| Q8_0 | 8-bit 量化，接近无损 |
| FA | FlashAttention |
| Slot | server 中一个并发序列的上下文 |
| Continuous batching | 动态把多个请求的 token 拼进同一 batch |
| Speculative decoding | draft 猜多个 token，target 一次验证 |
| MFU | Model FLOPs Utilization，算力利用率 |
| UMA | Unified Memory Architecture，CPU/GPU 共享物理内存 |
| 统一内存 | Apple silicon 上 CPU 与 GPU 共用同一块内存，无独立显存 |
| 有效带宽 | 模型文件大小 x tg t/s，用来判断 decode 的带宽利用率 |
| Working set | Metal 的 `recommendedMaxWorkingSetSize`，系统建议的 GPU 工作集上限 |
| MTLGPUFamilyAppleN | Metal 的 GPU 家族编号，kernel 调优表按它分桶 |
| Tensor API | Metal 4 的 `mpp::tensor_ops` matmul 路径，M5 起默认启用 |
| Fusion | Metal 后端把多个小算子合成一个 kernel |
| Command buffer | Metal 的提交单元，ggml 最多并发 8 个 |
| Coopmat | Vulkan cooperative matrix，即“矩阵核心” |
| ICD | Vulkan Installable Client Driver，显卡驱动入口 |

---

## 附录 B：命令速查

### 核心公式

```text
权重字节数        = P * b / 8
每 token KV 字节  = 2 * n_layer * n_head_kv * head_dim * bytes_per_elem
decode 时间下界   = 权重字节数 / 内存带宽
prefill FLOPs     = 2 * P * T
decode 算术强度   = 16 / b
prefill 算术强度  = 16 * T / b
有效带宽          = 模型文件大小 * tg t/s
```

### 构建

macOS（Metal 默认开）：

```bash
xcode-select --install
xcodebuild -downloadComponent MetalToolchain   # 较新的 Xcode
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j "$(sysctl -n hw.ncpu)"
```

Linux + Vulkan：

```bash
sudo apt-get install libvulkan-dev glslc spirv-headers
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build --config Release -j "$(nproc)"
```

纯 CPU 兜底 / BLAS：

```bash
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake -B build-blas -DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
```

### 运行

```bash
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF
llama-cli -m model.gguf -p "hello" -n 128 -c 4096 -ngl 99 -fa on
llama serve -m model.gguf -c 8192 -np 4
llama-cli --list-devices
```

服务（默认 127.0.0.1:9931）：

```bash
curl http://127.0.0.1:9931/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"local","messages":[{"role":"user","content":"hi"}]}'
curl http://127.0.0.1:9931/health
curl http://127.0.0.1:9931/props
```

### 基准

```bash
llama-bench -m model.gguf -p 512 -n 128 -r 5 -o md
llama-bench -m model.gguf -p 512 -n 128 -d 8192 -ngl 99 -fa on
llama-bench -m model.gguf -p 512 -n 128 -d 8192 -fitt 1024 -fitc 4096
llama-batched-bench -m model.gguf -c 16384 -b 2048 -ub 512 -npp 512 -ntg 128 -npl 1,2,4,8 -ngl 99
```

### 质量

```bash
./scripts/get-wikitext-2.sh
llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw -c 512 -ngl 99
llama-perplexity -m quant.gguf -f wikitext-2-raw/wiki.test.raw --kl-divergence-base fp16.kld --kl-divergence
```

### 量化

```bash
python convert_hf_to_gguf.py /path/to/hf --outtype bf16 --outfile model-bf16.gguf
llama-imatrix -m model-bf16.gguf -f calibration.txt -o imatrix.gguf -ngl 99
llama-quantize --imatrix imatrix.gguf model-bf16.gguf model-Q4_K_M.gguf Q4_K_M
llama-quantize --dry-run model-bf16.gguf Q4_K_M
```

### 查看 GGUF

```bash
python gguf-py/gguf/scripts/gguf_dump.py model.gguf
llama-cli -m model.gguf --verbose -p "hi" -n 1   # 启动日志里也有完整元数据
```

### 测试

```bash
test-backend-ops -o MUL_MAT
test-backend-ops perf -o 'MUL_MAT.*'
test-backend-ops -b Metal0 -o MUL_MAT
test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT
test-tokenizer-0 models/ggml-vocab-llama-spm.gguf
test-chat-template
test-llama-archs
```

### 调试

```bash
GGML_SCHED_DEBUG=2 llama-cli -m model.gguf -p "hi" -n 1
LLAMA_TRACE=1 llama-cli -m model.gguf -p "hi" -n 1
GGML_METAL_GRAPH_DEBUG=1 llama-cli -m model.gguf -p "hi" -n 1
```

在 `src/llama-context.cpp` 取消 `ggml_graph_dump_dot` 注释，可生成 `llama.dot`：

```bash
dot -Tsvg llama.dot -o llama.svg
```

### 性能观测

macOS：

```bash
vm_stat
memory_pressure
sysctl vm.swapusage
sudo powermetrics --samplers gpu_power,cpu_power,thermal -i 1000 -n 30
xctrace list templates
xctrace record --template 'Time Profiler' --launch -- ./build/bin/llama-cli -m model.gguf -p hi -n 64
```

Linux / Vulkan：

```bash
vulkaninfo --summary
sudo intel_gpu_top
perf stat -e task-clock,cycles,instructions ./build/bin/llama-cli -m model.gguf -p hi -n 64
```

### 环境变量速查

| 变量 | 平台 | 作用 |
|---|---|---|
| `GGML_METAL_TENSOR_DISABLE=1` | Metal | 关闭 Metal Tensor API |
| `GGML_METAL_TENSOR_ENABLE=1` | Metal | 在旧芯片上强制打开 |
| `GGML_METAL_FUSION_DISABLE=1` | Metal | 关闭算子融合 |
| `GGML_METAL_CONCURRENCY_DISABLE=1` | Metal | 串行提交 |
| `GGML_METAL_GRAPH_OPTIMIZE_DISABLE=1` | Metal | 关闭图优化 |
| `GGML_METAL_GRAPH_DEBUG=N` | Metal | 打印图 |
| `GGML_METAL_NO_RESIDENCY=1` | Metal | 关闭 residency set |
| `GGML_METAL_CAPTURE_COMPUTE=N` | Metal | 第 N 次图计算时抓 gputrace |
| `METAL_CAPTURE_ENABLED=1` | Metal | 允许程序化 capture |
| `GGML_VK_VISIBLE_DEVICES=0` | Vulkan | 选择设备 |
| `GGML_VK_DISABLE_COOPMAT2=1` | Vulkan | 关闭矩阵核心，对比用 |
| `GGML_VK_DISABLE_MMVQ=1` | Vulkan | 关闭 MMVQ |
| `GGML_VK_DISABLE_FUSION=1` | Vulkan | 关闭融合 |
| `GGML_VK_ALLOW_SYSMEM_FALLBACK=1` | Vulkan | VRAM 不足时回退 |
| `VK_ICD_FILENAMES=...` | Vulkan | 指定 ICD |
| `GGML_SCHED_DEBUG=2` | 通用 | 打印切分 |
| `LLAMA_TRACE=1` | 通用 | 模型加载追踪 |

---

## 附录 C：源码地图

| 文件/目录 | 先问什么 |
|---|---|
| `include/llama.h` | 公开对象和 API 是什么？ |
| `src/llama.cpp` | API 如何转发？ |
| `src/llama-context.cpp` | decode/encode 的主流程？ |
| `src/llama-batch.cpp` | batch 如何切 ubatch？ |
| `src/llama-memory.*` | 记忆抽象如何工作？ |
| `src/llama-kv-cache.cpp` | KV slot 如何分配/读写？ |
| `src/llama-model.cpp` | 模型如何加载、构图？ |
| `src/models/llama.cpp` | 标准 Transformer 图怎么搭？ |
| `src/llama-graph.cpp` | 通用 attention/ffn/norm 辅助？ |
| `src/llama-sampler.cpp` | 采样链如何实现？ |
| `src/llama-vocab.cpp` | tokenizer 如何实现？ |
| `src/llama-quant.cpp` | 量化流程与类型选择？ |
| `ggml/include/ggml.h` | 张量/图/算子 API？ |
| `ggml/include/ggml-backend.h` | 后端/scheduler API？ |
| `ggml/src/ggml.c` | 算子形状/调度？ |
| `ggml/src/ggml-cpu/` | CPU kernel/repack/线程？ |
| `ggml/src/ggml-metal/` | Metal 设备/命令缓冲/融合？ |
| `ggml/src/ggml-metal/kernels/` | Metal kernel 怎么写？ |
| `ggml/src/ggml-vulkan/` | Vulkan 设备/队列/算子？ |
| `ggml/src/ggml-vulkan/vulkan-shaders/` | Vulkan shader 怎么写？ |
| `common/` | CLI/server 共用逻辑？ |
| `common/fit.cpp` | 内存估算与 breakdown？ |
| `app/` | 统一入口怎么分发命令？ |
| `conversion/`、`gguf-py/` | HF -> GGUF？ |
| `tools/llama-bench/` | 基准如何测？ |
| `tools/tuning/` | Metal kernel 离线调优？ |
| `tools/server/` | 服务如何调度？ |
| `docs/ops/*.csv` | 每个后端支持哪些算子/类型？ |

---

## 附录 D：性能报告模板

```markdown
# 性能报告：<模型> on <硬件>

## 1. 环境
- 机器：Apple M5 Pro / Intel + Vulkan（具体型号）
- commit: cd959bae (0.9.0)
- 构建: Release / 后端 / 编译选项
- 系统: macOS 版本 或 Linux 内核 / Windows 版本
- CPU/GPU/内存:
- 驱动/Metal family/recommendedMaxWorkingSetSize:
- 编译器:

## 2. 模型与参数
- 模型/量化:
- 文件大小/参数量:
- n_ctx / n_batch / n_ubatch / n_parallel:
- ngl / fa / ctk / ctv / threads / load-mode:

## 3. 结果
| test | 配置 | t/s | 标准差 | 内存峰值 | 备注 |
|---|---|---:|---:|---:|---|

## 4. 理论对照
- decode 理论上限（文件大小 / 带宽）:
- 有效带宽 = 文件大小 * tg t/s:
- 占峰值百分比:
- prefill FLOPs 估算:

## 5. 观察与假设
- 观察到:
- 假设:
- 验证实验:

## 6. 改动与结果
| 改动 | before | after | 变化 | 风险 |
|---|---:|---:|---:|---|

## 7. 结论
- 瓶颈:
- 推荐配置:
- 后续工作:
```

---

## 附录 E：自检题参考答案（简版）

### 第 1 章

1. **bench tg 为什么不是用户看到的 tok/s？** bench 只测模型计算，不含 tokenization、sampling、网络和排队；用户看到的是端到端。
2. **n_ctx/n_batch/n_ubatch？** n_ctx 是 KV 总容量；n_batch 是一次调用的逻辑 token 上限；n_ubatch 是每次图计算的物理 token 上限。
3. **8B Q4 的理论上限？** 约 4.9GB / 307GB/s = 16ms/token，即约 63 tok/s；M5 Pro 实测通常 30 到 45，差在有效带宽和内存压力。
4. **为什么 Mac 上“降 ngl 腾内存”是错的？** Metal buffer 和 CPU 用同一块物理内存，降 ngl 只换计算单元，不释放内存。
5. **为什么必须先 Release？** Debug 关掉优化、插入检查，可能慢数倍，性能数字没有意义。

### 第 2 章

1. **为什么 decode 的算术强度与 P 无关？** 计算量 2P、读取量 P*b/8，P 约掉，只剩 16/b。
2. **GQA 如何影响 KV？** 减少 n_head_kv，KV 字节数按比例下降，K/V 投影计算也下降。
3. **attention 复杂度？** prefill 每层 O(T^2*d)，decode 每 token 每层 O(T*d)。
4. **量化为什么加速 decode 不一定加速 prefill？** decode 被带宽限制，字节少就快；prefill 被算力/GEMM 效率限制，反量化可能拖慢。
5. **MoE 为什么计算少内存大？** 每 token 只算 k 个 expert，但全部 expert 权重都要常驻。
6. **FA 支持哪些 KV？** Metal：f32/f16/q8_0/q4_0/q4_1/q5_0/q5_1/bf16（head_dim 限列表）；Vulkan：再加 iq4_nl，head_dim 需为 8 的倍数。

### 第 3 章

1. **为什么 model 和 context 分开？** model 是只读权重，可被多个 context 共享；context 持有 KV、batch、输出等可变状态。
2. **batch.logits 是什么？** 标记该 token 是否输出 logits，不是 logits 数据。
3. **n_batch 与 n_ubatch？** 前者是提交上限，后者是单次图计算上限；大 batch 会被切成多个 ubatch。
4. **返回 1 怎么处理？** 找不到 KV slot：减小 batch、增大 context、清理或复用 KV。
5. **图为什么复用/重建？** 形状和拓扑不变就复用；batch 形状或算子组合变化会重建并重新分配。
6. **Metal 与 Vulkan 的权重内存差异？** Metal 声明 buffer_from_host_ptr，mmap 页直接映射给 GPU；Vulkan 不支持，权重拷贝进 Vulkan buffer。

### 第 4 章

1. **为什么 ne[0] 连续？** 它是最内层 K 维，连续才能向量化读取、对齐量化 block。
2. **何时插入拷贝？** 算子不被当前后端支持、buffer 类型不匹配、或图跨设备切分时。
3. **BLAS 为什么主要帮 prefill？** 它优化大 GEMM；decode 是 batch=1 的 GEMV，受带宽限制。
4. **repack 是什么？** 加载时把量化权重按 SIMD 友好的块重排，只影响运行期内存布局，不改 GGUF。
5. **部分 offload 为什么可能更慢？** 切图导致跨设备拷贝和同步；CUDA 上是 PCIe，Mac 上是命令缓冲同步，代价不同但都存在。
6. **has tensor 是什么？** Metal 4 的 mpp::tensor_ops matmul 路径，M5/M6/A19/A20 默认启用，旧芯片上默认关闭。
7. **uma 1 与 uma 0？** uma=1 时 ngl 不扩容，uma=0 时 ngl 真正把权重放进 VRAM。

### 第 5 章

1. **TTFT/TPOT 由什么决定？** TTFT = 排队 + tokenize + prefill + 首次采样；TPOT = decode 时间 / (输出 token 数 - 1)。
2. **为什么 Mac 上仍建议全量 offload？** 避免切图和 CPU/GPU 同步，GPU 算得更快；内存总量不变，但这是性能默认，不是容量手段。
3. **为什么 -np 提高吞吐但恶化 p99？** 多序列摊薄权重读取，但 KV/attention 竞争和排队增加。
4. **投机解码何时变慢？** 接受率低、draft 太贵、k 太大、draft 挤占内存。
5. **为什么 KV 量化长上下文收益大？** KV 随上下文线性增长，长上下文时占比高，减半直接省出大块容量。
6. **如何证明瓶颈是带宽？** 有效带宽接近峰值，同时改 ubatch 对 tg 影响小；反之改 ubatch 明显则更像算力/调度问题。
7. **Mac 上性能掉一半先看什么？** `vm_stat`、`memory_pressure`、`sysctl vm.swapusage`。

### 第 6 章

1. **Q4_0 与 Q4_K 的结构差异？** Q4_0 每 32 权重一个 fp16 scale；Q4_K 每 256 权重一个 super-block，内含 8 个子块及 6-bit scale/min。
2. **imatrix 统计什么？** 校准数据上的激活重要性，让重要通道保留更多精度。
3. **为什么量化 V 需要 FA？** 量化后的 V 需要融合 attention kernel 直接读量化 cache；非 FA 路径通常要求 f16。
4. **为什么量化不一定加速 prefill？** prefill 是 compute-bound，反量化/查表增加额外计算。
5. **如何判断能上线？** PPL + KLD + 固定任务 + 长上下文抽检 + 目标硬件性能回归。
6. **哪些 KV 组合会回退？** iq4_nl 在 Metal 上不支持 FA；head_dim 不在 Metal 列表、或不是 8 的倍数（Vulkan）时回退。

### 第 7 章

1. **新模型最少改什么？** 转换器、GGUF 常量/张量映射、C++ arch、hparams/tensors/graph、tokenizer/模板、测试。
2. **模板错为什么“变笨”？** 输入格式偏离训练分布，模型看到的是错误的角色/分隔符。
3. **新后端不支持算子怎么办？** supports_op 返回 false，调度器回退到支持的后端并插入拷贝，同时在日志/测试中可见。
4. **量化后如何验证能上线？** 质量、长上下文、工具调用、多后端一致性、目标硬件性能全部通过。
5. **L1-L5 自评？** 看你能拿出什么证据：基线表、优化报告、调用链笔记、量化报告、kernel 改动。
6. **Metal 与 Vulkan logits 不一致怎么查？** 先用 test-backend-ops 对比单算子，再用 GGML_SCHED_DEBUG=2 看是否回退，最后查 KV/FA 的类型支持。

---

## 附录 F：推荐阅读

按顺序：

1. **Transformer 基础**
   - *Attention Is All You Need*: https://arxiv.org/abs/1706.03762
   - The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
2. **位置编码与注意力变体**
   - RoPE: https://arxiv.org/abs/2104.09864
   - GQA: https://arxiv.org/abs/2305.13245
   - FlashAttention: https://arxiv.org/abs/2205.14135
3. **量化**
   - GPTQ: https://arxiv.org/abs/2210.17323
   - AWQ: https://arxiv.org/abs/2306.00978
   - llama.cpp quantize README: `tools/quantize/README.md`
   - imatrix README: `tools/imatrix/README.md`
4. **工程**
   - ggml: https://github.com/ggml-org/ggml
   - GGUF spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
   - llama.cpp docs: `docs/`
   - Roofline: https://dl.acm.org/doi/10.1145/1498765.1498785
5. **Apple / Metal**
   - Metal Shading Language Specification: https://developer.apple.com/metal/
   - Metal Best Practices Guide（Apple Developer Documentation）
   - `docs/build.md` 的 Metal 一节
   - `tools/tuning/README.md`（FA vec 调优表怎么生成）
6. **Vulkan / Intel**
   - Vulkan Specification: https://registry.khronos.org/vulkan/
   - Vulkan Tutorial: https://vulkan-tutorial.com/
   - Intel 显卡驱动与 `intel_gpu_top` 文档
   - `docs/build.md` 的 Vulkan 一节
7. **性能工具**
   - macOS：`powermetrics`、Instruments、`xctrace`、`vm_stat`
   - Linux：`perf`、`intel_gpu_top`、`vulkaninfo`
   - `test-backend-ops`、`llama-bench`、`llama-batched-bench`

---

## 附录 G：版本提示

本快照是 **0.9.0**（commit `cd959bae`，2026-11-04）。`CMakeLists.txt` 里的版本号是 `0.9.0`，非 release tag 构建会显示 `0.9.0-dev`。

这个版本已经包含：

- 新 batch API（`llama_batch_ext` / `llama_process`），`examples/simple` 已切换；
- `llama_memory` 抽象（KV/iSWA/recurrent/hybrid/DSA 等）；
- 新的量化格式（MXFP4/NVFP4/Q1_0/Q2_0/F8_E4M3/TQ1_0/TQ2_0/IQ 系列）；
- 新的投机解码类型（EAGLE-3/DFlash/DSpark/ngram）；
- Metal Tensor API 在 M5/M6/A19/A20 上默认启用；
- Metal 支持 MXFP4/NVFP4；
- `ggml-metal-tuning` 离线 kernel 调优工具；
- 后端采样、融合算子、MTP；
- `-fit`（默认 on）、`-ngl auto`、`--jinja` 默认开、server 默认端口 9931。

这些变化不影响本书的主线，但意味着：

1. 不要背 API 签名，理解对象和数据流；
2. 以 `include/llama.h` 和工具 README 为准；
3. 升级版本后重跑基线，不要假设性能/质量不变；
4. 读旧博客/旧教程时，先确认版本，尤其是默认值和旧的 `llama_kv_cache_*` 接口；
5. 两台机器用的必须是同一个 commit 和同一份 GGUF，否则对比没有意义；
6. 本书的命令、默认值、日志字段和源码路径都在 0.9.0 的树上核对过，升级后请重新核对一遍。

---

## 最后一页：三句话

1. **推理引擎的核心，是把“读多少字节、做多少 FLOPs、放在哪块内存”安排明白。**
2. **先测量，再优化；先跑对，再跑快；先理解，再改代码。**
3. **llama.cpp 的主线只有一条：token -> batch -> KV -> graph -> backend -> logits -> token。** 把这条线走通，其他都是支线。

---

*完。*
