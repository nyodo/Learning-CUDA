# RTX 5090 Flash Attention 优化与 Profiling 记录

本文记录 `src/kernels.cu` 中 Flash Attention 前向实现的环境、版本演进、正确性、性能、
NCU/NSYS/SASS 证据和最终取舍。数据用于解释当前实现，不代表官方排行榜环境的最终名次。

## 1. 约束与交付范围

- 实现 `flashAttention<float>` 与 `flashAttention<half>` 前向。
- 支持 causal、GQA、`Tq != Tk`，`D <= 256`。
- 不实现 backward、dropout、任意 attention mask。
- 不修改公开函数、显式模板实例、Makefile 和 tester。
- 正式代码不依赖全局缓存、持久 workspace 或调用者生命周期假设。
- CUDA 11 API 范围内保留通用 SIMT/WMMA 代码；RTX 5090 实验构建使用 CUDA 13
  `-arch=sm_120`。
- 没有查看或复制公开 FlashAttention/CUTLASS/Triton kernel 源码。

## 2. 平台

| 项目 | 值 |
|---|---|
| Git 基线 | `4c7bf8fe4f39a984c2aa75edbcd0c2226f50c5f9` |
| GPU0 | RTX 5090, UUID `GPU-ca94ba5a-1a2f-8928-b793-076fe9f6e0a4` |
| GPU1 | RTX 5090, UUID `GPU-e1dd35dc-1ace-5e86-9fff-78f47749381c` |
| Compute Capability | 12.0 |
| 显存 | 每卡 32607 MiB |
| 驱动 | 580.126.09 |
| nvcc | CUDA 13.0, V13.0.88 |
| Nsight Compute | 2025.3.1.0 |
| Nsight Systems | 2025.3.2 |
| Compute Sanitizer | 2025.3.1.0 |

拓扑中 GPU0 对应 NUMA node 0，CPU affinity 为 `0-11,24-35`；GPU1 对应 node 1。
所有主性能数据使用：

```bash
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 ...
```

GPU1 只做正确性复验，不混入性能表。预编译 tester 依赖 CUDA 13 的
`cudaGetDeviceProperties_v2`，构建时链接 `/tmp/cuda13_abi_shim.o`；shim 不进入仓库。

完整环境原始记录：`~/flash_attention_profiles_5090/environment.log`。

## 3. 版本演进

| 版本 | 核心变化 | 结果与决定 |
|---|---|---|
| V0 | 原仓库 TODO | 建立 tester 维度、容差和 host 生命周期基线 |
| V1 | 一 warp 一 query，逐 key online softmax | 通用正确；half 最大用例 kernel/函数仍有较大优化空间 |
| V2 | 同 KV head 的 8 个 query row/CTA，shared K/V tile32 | GQA/短序列复用改善；float 需保持严格累加顺序 |
| V3 | float 三遍稳定 softmax | 与参考顺序更接近，覆盖任意 `D<=256` |
| V4 | float D32 编译期固定数组并展开 | 官方最大 float 保持零 diff；94 regs，零 spill |
| V5 | half D32/D64 以 WMMA 做 QK，FP32 softmax + SIMT PV | 官方 half 通过；最大 kernel 约 0.410 ms |
| V6 | 尝试 float warp 并行 QK | case 6/14 超过严格容差，回退，不进入正式代码 |
| V7 | 评估 `cp.async`、host 多流、Graph、UM | NCU/NSYS 不支持合入正式同步接口，保留研究结论 |

关键教训是：float 的数学等价并不足以通过紧容差。warp tree reduction 与串行 FMA 的
舍入顺序不同；即便最大绝对差只有约 `1.66e-5`，仍可能超过某个输出的动态 tolerance。

## 4. 最终 dispatch

### Float

```text
D == 32
  -> float D32 固定数组三遍路径，128 threads/CTA
否则 D <= 64、Tk >= 16、rows_per_kv >= 8
  -> 8 query rows/CTA 的 shared-KV 三遍路径
否则
  -> 一线程一 query row 的通用三遍路径，D <= 256
```

float 不使用 TF32。通用三遍路径为 `D=256` 预留本地数组，ptxas 报 2048 B stack frame；
这是罕见尺寸的正确性 fallback，不用于官方最大 D32 热路径。

### Half

```text
D in {32,64}、Tk >= 64、rows_per_kv >= 16
  -> half WMMA online 路径，128 threads/CTA，64 query rows/CTA
否则 Tk <= 256、D <= 64、rows_per_kv >= 8
  -> shared-KV 三遍路径
否则
  -> 一 warp 一 query 的 online fallback，D <= 256
```

WMMA 只用于 QK。softmax 状态与 PV 均为 FP32，最后一次性转 half。

## 5. 正确性

### 官方 tester

- 默认 `-O0`：RMSNorm float/half 26/26，Attention float/half 28/28。
- `-O3 -lineinfo -arch=sm_120`：Attention float/half 28/28。
- 最大官方 shape：`(B,Tq,Tk,Hq,Hkv,D)=(4,512,2048,64,64,32)`。
- 官方覆盖 `D=1/2/4/8/16/32/64`、GQA 比例 2/3/4、两种 causal、`16x32`
  与 `32x16` 非方形序列。

O3 官方日志中两个较大用例的完整函数均通过：

| case | 类型 | Avg Time | Max Diff | Max Tolerance |
|---|---:|---:|---:|---:|
| 13 | float | 10.208 ms | 0 | 0 |
| 13 | half | 1.595 ms | 0.0078125 | 0.0469141 |
| 14 最大 | float | 54.644 ms | 0 | 0 |
| 14 最大 | half | 20.408 ms | 0.0078125 | 0.0502734 |

完整函数时间包含 pageable H2D、分配、kernel、D2H 和释放，不能当作 kernel-only 时间；
它也会随主机内存状态明显波动，因此 dispatch 判断以独立 kernel profile 为主。

### 扩展矩阵

独立 CPU double reference 覆盖：

- float 36/36，half 36/36。
- `D=7/31/32/33/64/128/256`。
- GQA、causal、非方形、序列尾 tile。
- 全零、常量、随机正负、大动态范围、极端 softmax logits。
- float 最坏绝对误差 `3.1584118e-06`。
- half 最坏绝对误差 `0.00097627926`。
- 零尺寸路径通过。

阈值为 float `abs<=2e-4` 或 `rel<=2e-3`；half `abs<=5e-3` 或
`rel<=1e-2`。

GPU0 与 GPU1 均复验。GPU1 在设置 `CUDA_VISIBLE_DEVICES=1` 后，程序内部看到的 ordinal
仍是 0，这是 CUDA 可见设备重映射的正常行为。

### Compute Sanitizer

最终扩展驱动分别执行：

| 工具 | 结果 |
|---|---|
| memcheck | 0 errors，0 leaks |
| initcheck | 0 errors |
| racecheck | 0 hazards / errors / warnings |
| synccheck | 0 errors |

日志位于 `~/flash_attention_profiles_5090/sanitizer_*.log`。

## 6. ptxas 与 SASS

最终命令：

```bash
make build \
  CFLAGS='-std=c++17 -O3 -lineinfo -arch=sm_120 -Xptxas=-v' \
  EXTRA_LIBS=/tmp/cuda13_abi_shim.o
cuobjdump --dump-sass src/kernels.o > \
  ~/flash_attention_profiles_5090/kernels_sm120.sass
```

资源摘要：

| kernel | regs/thread | dynamic shared | stack | spill |
|---|---:|---:|---:|---:|
| half WMMA D32 | 56 | 约 10 KiB | 0 | 0 |
| half WMMA D64 | 72 | 约 16 KiB | 0 | 0 |
| shared-KV | 40 | 随 D 变化 | 0 | 0 |
| float D32 | 94 | 0 | 0 | 0 |
| float 通用 D<=256 | 40 | 0 | 2048 B | 0 |

SASS 发现 12 条静态 `HMMA.16816.F32` 指令，证明 WMMA 路径生成 Tensor Core 指令。
`LDGSTS` 数量为 0，证明正式路径没有暗中生成 `cp.async`。整个目标中存在 local 指令，
但 ptxas 明确显示所有热专用 kernel 零 spill；local 指令主要来自大 D 通用 fallback 等代码。

原始文件：

- `~/flash_attention_profiles_5090/ptxas_sm120.log`
- `~/flash_attention_profiles_5090/kernels_sm120.sass`

## 7. Nsight Compute

NCU 使用 `--clock-control base`，每个报告只采一个固定 kernel。stall 百分比按
`pcsamp ... not_issued` 各原因样本归一化，适合看构成，不应理解为时间百分比。

| 场景 | kernel | 时间 us | SOL % | DRAM % | L1 % | L2 % | regs | shared KiB | theoretical/achieved occ % | waves/SM | Long SB % | Short SB % | Barrier % | LG throttle % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny float | 通用三遍 | 17.024 | 1.95 | 0.09 | 4.82 | 0.30 | 40 | 1.02 | 100/2.23 | 0.00 | 57.14 | 0.00 | 0.00 | 0.00 |
| 非方形 half `32x16,D32` | shared-KV | 32.800 | 14.32 | 0.11 | 81.07 | 0.24 | 40 | 10.29 | 100/16.62 | 0.03 | 6.52 | 22.10 | 4.53 | 0.00 |
| GQA half `128x256,D32,Hq16/Hkv4` | WMMA | 43.520 | 2.29 | 0.28 | 19.03 | 1.07 | 56 | 11.26 | 75/8.24 | 0.02 | 57.63 | 17.89 | 1.84 | 0.00 |
| square half `256x256,D64` | WMMA | 139.136 | 5.12 | 0.33 | 27.48 | 0.72 | 72 | 17.41 | 41.67/8.30 | 0.04 | 51.31 | 16.21 | 1.36 | 1.03 |
| 最大 half | WMMA | 409.856 | 62.44 | 3.48 | 74.76 | 9.56 | 56 | 11.26 | 75/53.60 | 1.34 | 16.41 | 15.70 | 19.29 | 0.92 |
| 最大 float | D32 固定 | 17569.280 | 71.63 | 0.16 | 99.43 | 20.17 | 94 | 1.02 | 41.67/33.33 | 1.20 | 62.26 | 0.05 | 0.00 | 36.51 |

所有表中 kernel 的 `derived__local_spilling_requests` 都是 0。

Pipe 指标也单独核对：最大 half 的 Tensor pipe active 为 1.76%，NCU 的 XU（用于观察
MUFU/特殊函数类指令压力）为 7.47%；最大 float 分别为 0% 与 0.07%。SASS 全目标可见
`MUFU.RCP/MUFU.RSQ` 等指令。Tensor pipe 的低占比说明 half kernel 的时间还包含大量
softmax、SIMT PV、同步与数据布局工作，不能只用“已经生成 HMMA”推断 Tensor Core 已饱和。

### 7.1 最大 half 的结论

- 0.410 ms，已经接近 PyTorch cuDNN 黑盒的 0.405 ms。
- DRAM 仅 3.48%，L1 74.76%，L2 hit 约 82%。
- achieved occupancy 53.60%，1.34 waves/SM，存在尾波影响。
- stall 更分散，Barrier 19.29%，Long/Short Scoreboard 各约 16%。

这不是 DRAM 带宽打满或纯 HBM latency 场景。下一步更可能来自 query/key tile、shared
布局、PV Tensor Core 化或减少 softmax/同步，而不是简单加 global-to-shared 双缓冲。

### 7.2 最大 float 的结论

- L1 throughput 99.43%，DRAM 仅 0.16%，L2 hit 约 99%。
- Long Scoreboard 62.26%，LG Throttle 36.51%。
- 94 regs 把 theoretical occupancy 限制到 41.67%，但零 spill。
- global request 存在大量 excessive sectors。

因此它的“Long Scoreboard”主要对应 L1/请求依赖和访问形状，不能仅凭这个名字断言需要
`cp.async`。若要大幅加速 float，必须改变并行 QK/PV；但 TF32 被精度要求排除，warp
reduction 又未通过紧容差，第一版保留严格路径。

原始报告和汇总：

- `tiny_float.ncu-rep`
- `rect_half.ncu-rep`
- `gqa_half.ncu-rep`
- `square_half_d64.ncu-rep`
- `wmma_half_max.ncu-rep`
- `float_d32_max.ncu-rep`
- `ncu_summary.tsv`

均位于 `~/flash_attention_profiles_5090/`。

## 8. 为什么不合入 `cp.async`

预设门槛是：Long Scoreboard 明显、DRAM 未接近上限、寄存器/occupancy 有余量，并且
原型在代表尺寸提升至少 5%。实际数据只满足“有部分 scoreboard 等待”，不满足完整因果链：

1. 最大 half DRAM 3.48%，但数据主要命中 L2/L1，kernel 已与 cuDNN 接近。
2. 最大 float 的 L1 已达 99.43%，请求合并和串行依赖才是问题。
3. 双缓冲会额外占 shared memory，并增加 barrier/pipeline bookkeeping。
4. 最大 half kernel 仅占完整函数的很小部分，kernel 再快 10% 对同步接口端到端也不足 5%。

所以正式 SASS 保持 `LDGSTS=0`。这不是说 `cp.async` 永远无用；若未来接口变成 device
tensor 输入、kernel 成为端到端主项，再重新以较大 tile 做 A/B。

## 9. Nsight Systems 官方流程

`official_end_to_end.nsys-rep` 对整个 tester 的汇总：

### CUDA API

| API | 总时间 | 占比 | 调用数 |
|---|---:|---:|---:|
| `cudaMemcpy` | 2183.5 ms | 76.0% | 2202 |
| `cudaMalloc` | 311.4 ms | 10.8% | 706 |
| `cudaFree` | 269.8 ms | 9.4% | 706 |
| kernel launch | 约 51.7 ms | 1.8% | 多次 |

### GPU memops

| 方向 | 总时间 | 占比 | 总大小 |
|---|---:|---:|---:|
| H2D | 1481.9 ms | 88.6% | 3045.652 MB |
| D2H | 190.4 ms | 11.4% | 437.623 MB |

结论：`CUDA API 共多少毫秒` 是 tester 中许多次同步调用的累计值，不是一发 kernel 的
launch overhead。它主要是同步 pageable memcpy，其次是每次 fresh allocation/free。

## 10. Host 系统方案 A/B

场景为最大 half，Event 驱动在无 profiler 下重复 5 组，表中取中位数：

| 方案 | 中位数 ms | 相对 pageable fresh | 是否可直接合入 |
|---|---:|---:|---|
| pageable + fresh contiguous malloc | 47.105 | 基线 | 当前正式语义 |
| persistent device workspace + pageable sync | 45.226 | 快 3.99% | 未达到 5%，且需要持久状态 |
| 预填充 pinned + 两条 H2D stream | 3.912 | 快 91.69% | 否，假设调用者已持有 pinned 输入 |
| pageable vector -> pinned staging + 两条 stream | 43.172 | 快 8.35% | 否，CPU staging 仍很重且要管理缓存 |
| 固定 pinned CUDA Graph | 3.982 | 快 91.55% | 否，固定地址/shape/生命周期；本轮还略慢于直接 pinned stream |
| Managed + advise + prefetch | 217.065 | 慢 360.8% | 否 |

这里最容易误读的是“pinned + 两流很快，为何不用”：`3.912 ms` 的输入已经提前放在
pinned buffer，计时没有包含 `std::vector -> pinned`。把接口真实成本加回后变成
`43.172 ms`。它仍比基线快约 8%，但需要全局/持久 pinned cache、并发规则、尺寸复用、
异常清理和内存上限策略；这些都超出当前无状态公开接口，且官方每个调用的 vector 地址
和尺寸可变化。

Graph 同理：理想固定地址版本很快，但它优化的是重复固定序列，不会自动解决 pageable
输入。当前测试中直接 pinned 双流甚至比 Graph 稍快，说明 launch 不是剩余主瓶颈。

### 10.1 A/B 的 NSYS 证据

`system_ab.nsys-rep` 包含全部六种路径。因 profiler 会显著扰动 API 时间，绝对性能使用
上面的 Event 表；NSYS 只解释组成：

| 项目 | NSYS 聚合结果 |
|---|---:|
| `cudaMemcpy_ptds` | 1454.5 ms，CUDA API 时间的 62.8% |
| `cudaStreamSynchronize` | 331.8 ms，14.3% |
| `cudaHostAlloc` | 224.6 ms，9.7% |
| `cudaMemPrefetchAsync` | 152.4 ms，6.6% |
| GPU H2D | 1329.8 ms，memop 时间的 86.3% |
| GPU D2H | 157.2 ms，10.2% |
| UM H2D | 28.6 ms，1336 个迁移操作 |
| UM D2H | 25.2 ms，3056 个迁移操作 |

UM 的 GPU copy engine 时间看似不大，但 CPU 每轮写 managed pages、prefetch、同步和大量
小块迁移共同构成 217 ms。`advise` 是提示，`prefetch` 是提前发生迁移，不是消除迁移。

原始文件：`system_ab.nsys-rep`、`system_ab.stats.txt`、
`system_ab_event_runs.log`。

## 11. PyTorch/cuDNN 黑盒对比

一次性环境为 PyTorch `2.11.0+cu128`。只调用公开
`torch.nn.functional.scaled_dot_product_attention` 和 `sdpa_kernel` 选择 backend，
未读取实现源码。每个 backend 预热 20 次、200 次计时、5 组取中位数。

### 数值

方形 causal half 小尺寸相对 PyTorch MATH：

| backend | max abs | mean abs |
|---|---:|---:|
| FLASH_ATTENTION | 0.0009766 | 0.0000375 |
| CUDNN_ATTENTION | 0.0009766 | 0.0000374 |

### 性能

| 场景 | 本实现 kernel-only | PyTorch/cuDNN | PyTorch Flash |
|---|---:|---:|---:|
| GQA half `1,128,256,Hq16,Hkv4,D32,causal` | 0.04352 ms | 0.40367 ms | 该非方形 causal 形状无可用 kernel |
| 最大 half `4,512,2048,H64,D32,causal` | 0.40986 ms | 0.40526 ms | 该非方形 causal 形状无可用 kernel |

最大场景中本实现约 83.84 TFLOP/s，cuDNN 约 84.79 TFLOP/s，差约 1.1%。GQA 小尺寸
中本实现更快，主要因为 cuDNN 固定开销与该形状的 work partition 不理想；不要把一个
小尺寸结果泛化成所有模型场景。

为确认 Flash backend 不是在 5090 上整体不可用，额外测试方形
`1,512,512,H8,D32,causal`：Flash 为 0.44640 ms，cuDNN 为 0.40471 ms。它拒绝的是
当前 PyTorch 版本中的非方形 causal 组合，不是 GPU。

日志：`~/flash_attention_profiles_5090/pytorch_sdpa_blackbox.log`。

## 12. 未采用方案

| 方案 | 未采用原因 |
|---|---|
| TF32 float | 会改变 float 语义和误差，不符合严格 tester |
| half 概率 + WMMA PV | half 概率量化风险；当前 FP32 PV 已接近 cuDNN 最大场景 |
| `cp.async` 双缓冲 | DRAM 非瓶颈，增加 shared/barrier，端到端收益不足 |
| TMA/WGMMA | Hopper/数据中心 Blackwell 路线，不是第一版 5090 通用依赖 |
| `tcgen05`/Tensor Memory/2-CTA MMA | 面向 `sm_100/101` 能力，RTX 5090 是 `sm_120` |
| split-K | 需要第二次归并/workspace，当前 query 方向并行已足够覆盖最大用例 |
| persistent `cudaMallocAsync` workspace | 需要状态与并发策略，单独 device workspace 提升不足 5% |
| pinned 双流 | 理想收益依赖调用者持有 pinned 输入；当前 `std::vector` staging 成本高 |
| CUDA Graph | 固定地址/shape/生命周期不符合无状态公开函数 |
| Managed + advise + prefetch | CPU/GPU 往返迁移，实测慢 4.61 倍 |

## 13. 当前结论与下一步

当前 half D32 最大 kernel 已与 PyTorch cuDNN 黑盒相差约 1%，可以认为第一版在这个主要
目标上接近实用上限；正式接口的主要剩余时间在 pageable host 传输和同步生命周期。

float D32 仍明显慢，但 profiler 已把问题定位为 L1 请求压力、访问合并与严格累加依赖，
不是 HBM 或缺少 prefetch。继续优化需要在“改变并行归约顺序”和“官方零 diff”之间做
新的数值设计，而不是继续堆异步 API。

若未来允许改变接口，最值得做的是提供 device-tensor/stream/workspace API，让调用者持久
管理 pinned 输入、device buffer 和 Graph。若仍保持当前 `std::vector` 同步接口，正式版
已经选择了较好的复杂度、正确性与端到端收益平衡。

## 14. 原始产物索引

所有报告都在仓库外：`~/flash_attention_profiles_5090/`。

- `environment.log`：GPU、驱动、CUDA、工具和 NUMA。
- `final_o3_gpu0.log`：O3 官方测试。
- `extended_gpu0.log`、`extended_gpu1.log`：扩展正确性。
- `sanitizer_*.log`：四种 Sanitizer。
- `ptxas_sm120.log`、`kernels_sm120.sass`：资源与指令。
- `*.ncu-rep`、`ncu_summary.tsv`：六个 NCU 场景。
- `official_end_to_end.nsys-rep`：官方端到端流程。
- `system_ab.nsys-rep`、`system_ab.stats.txt`：系统 A/B 时间线。
- `system_ab_event_runs.log`：无 profiler 的五组 A/B。
- `pytorch_sdpa_blackbox.log`：PyTorch/cuDNN/Flash 黑盒。

## 15. 复现命令模板

官方默认：

```bash
cd ~/Learning-CUDA
make clean
make build EXTRA_LIBS=/tmp/cuda13_abi_shim.o
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
  ./test_kernels --verbose
```

O3 sm120：

```bash
make clean
make build \
  CFLAGS='-std=c++17 -O3 -lineinfo -arch=sm_120 -Xptxas=-v' \
  EXTRA_LIBS=/tmp/cuda13_abi_shim.o
```

NCU：

```bash
sudo ncu --clock-control base \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section LaunchStats \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section SourceCounters \
  --launch-count 1 -o REPORT ./ONE_CASE_DRIVER CASE
```

NSYS：

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-memory-usage=true -o REPORT ./PROGRAM
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,\
cuda_gpu_mem_size_sum REPORT.nsys-rep
```

报告时间可能受基础时钟、CPU page cache、pageable pinning、NUMA 和 profiler replay 影响。
因此所有优化结论同时要求正确性、资源/SASS、NCU 瓶颈和无 profiler Event 中位数相互印证。
