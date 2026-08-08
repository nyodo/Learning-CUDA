# RTX 5090 RMSNorm CUDA 优化与 Profiling 记录

## 1. 范围与结论

本轮只实现 NVIDIA 平台的 `rmsNorm<float>` 和 `rmsNorm<half>`。Flash Attention 保持
TODO，公开函数签名、显式模板实例、Makefile 和 tester 均未修改。

最终实现包含：

- 任意合法维度可用的标量路径；
- 一行一个 block 的两级规约；
- float4 和 half8 的 128-bit 寄存器缓存路径；
- FP32 平方和、归一化与权重乘法；
- 一次连续、256 字节分段对齐的设备分配；
- 所有 CUDA Runtime 调用和 kernel launch 的错误检查；
- 实验用编译期开关 `RMSNORM_FORCE_SCALAR`。

在两张 RTX 5090 上，默认官方测试和扩展测试均全部通过。GPU0 的四类 Compute
Sanitizer 检查全部为零错误。NCU、NSYS、ptxas、cuobjdump 和 nvdisasm 都取得了有效
数据。最终没有把 `cp.async`、多流、Managed Memory、CUDA Graph 或全局 workspace
加入正式代码，因为它们不能同时满足当前接口、兼容性、无隐式状态和稳定收益约束。

这里最重要的结论是：

> 当前 kernel 已经完成一轮可验证的规约、向量化和访存优化；当前公开函数的端到端
> 主瓶颈却是每次调用的分配、释放和 Host/Device 传输，而不是这个约 4 us 的 kernel。

## 2. 新平台环境

仓库从 `https://github.com/nyodo/Learning-CUDA.git` 克隆到
`/home/zhaoyk/Learning-CUDA`，基线 HEAD 为：

```text
fbfbd638b75403efa899684d06297a57ad417996
```

实验环境：

| 项目 | 值 |
|---|---|
| 操作系统 | Ubuntu 24.04.4 LTS |
| GPU | 2 x NVIDIA GeForce RTX 5090, CC 12.0 |
| 主测试卡 | 物理 GPU0, NUMA node 0 |
| 复验卡 | 物理 GPU1, NUMA node 1 |
| Driver | 580.126.09 |
| CUDA Toolkit | 13.0, nvcc 13.0.88 |
| Nsight Compute | 2025.3.1 |
| Nsight Systems | 2025.3.2 |
| Compute Sanitizer | 2025.3.1 |
| CPU | 2 x Intel Xeon Silver 4410Y, 24 cores / 48 threads |
| GPU0 NUMA CPUs | 0-11,24-35 |
| GPU1 NUMA CPUs | 12-23,36-47 |

GPU0 性能命令均使用：

```bash
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 ...
```

GPU1 只跑正确性，未将其数据混入性能表。NCU 硬件计数器受
`RmProfilingAdminOnly: 1` 限制，因此只对 NCU 使用 `sudo`；NCU 显式使用
`--clock-control base`，结束后没有留下锁频设置。

完整环境快照和原始报告位于仓库外：

```text
/home/zhaoyk/rmsnorm_profiles_5090/
```

## 3. CUDA 13 tester ABI 兼容

预编译的 `tester/tester_nv.o` 引用了旧符号 `cudaGetDeviceProperties_v2`，而 CUDA 13 的
Runtime 只导出当前 `cudaGetDeviceProperties`。原始 TODO 代码第一次链接因此失败。

测试时在 `/tmp` 编译了一个临时转发对象：

```cpp
#include <cuda_runtime.h>

extern "C" cudaError_t cudaGetDeviceProperties_v2(
    cudaDeviceProp* properties, int device) {
  return cudaGetDeviceProperties(properties, device);
}
```

构建时通过 `EXTRA_LIBS=/tmp/codex-cuda-compat.o` 链接。该 shim 不进入仓库，也不是
正式实现的依赖。原始 TODO 在 shim 下可以运行 tester，26 个 RMSNorm 测例全部失败，
证明 tester 流程本身有效，也建立了真实的零实现基线。

## 4. 算子语义与性能模型

对第 `i` 行：

```text
sum_square[i] = sum_j input[i, j]^2
inv_rms[i] = rsqrt(sum_square[i] / hidden_dim + eps)
output[i, j] = input[i, j] * inv_rms[i] * weight[j]
```

float 和 half 都在 FP32 中累加。half 只在读取时转成 FP32，在最后写回时舍入成 FP16。

标量路径不缓存整行 input，因此每个元素的逻辑请求流量是：

```text
input 第一次读取 + input 第二次读取 + weight 读取 + output 写入
```

| 路径 | float | half |
|---|---:|---:|
| 标量 | 16 bytes/element | 8 bytes/element |
| 寄存器缓存 | 12 bytes/element | 6 bytes/element |

每个元素主要是平方累加和两次乘法，算术强度很低。向量路径的价值不只是把四条或八条
标量指令写成一个宽指令，更重要的是第一次读 input 时把它保留到输出阶段，消除第二次
input 全局读取。表中的逻辑流量不等于最终 DRAM 流量，因为重复使用的 weight 可以命中
L2，NCU 必须用于观察真实层级行为。

## 5. 通用标量路径

### 5.1 拓扑

```text
grid.x = rows
一个 block 负责一行
block size 从 32、64、128、256 中选择
```

线程以 `BlockSize` 为步长遍历列。小维度选择更小的完整 warp block，大维度最多使用
256 个线程，避免为单行引入跨 block 规约、原子操作或第二个 kernel。

### 5.2 两级规约

1. 每线程在寄存器中用 `fmaf(x, x, sum)` 累加局部平方和。
2. 每个 warp 用 `__shfl_down_sync(0xffffffffu, ...)` 规约。
3. 每个 warp 的 lane 0 把部分和写入少量 shared memory。
4. `__syncthreads()` 后，首 warp 读取这些部分和并再次 shuffle。
5. 线程 0 计算 `rsqrtf(sum / hidden_dim + eps)`，写入 shared 标量。
6. 再次同步，所有线程进行输出遍历。

所有 block 大小都是 32 的整数倍。越界线程贡献零，但仍参加 shuffle 和同步，因此
`0xffffffffu` mask 对每个参与 warp 都是合法的。shared memory 只保存每 warp 一个
FP32 部分和以及一个 `inv_rms`，不存在矩阵式 shared-memory tile，也没有值得担心的
bank conflict。

## 6. float4 与 half8 路径

### 6.1 float4

float 路径把 input、weight 和 output 解释为 `float4`。每个 pack 正好 16 字节，包含
4 个 float。每线程缓存 1、2、4 或 8 个 pack。

### 6.2 half8

half 路径用一个 `uint4` 携带 16 字节，也就是 8 个 half。每个 32-bit 字段通过 half2
转换成两个 FP32 值。缓存保存原始 half bits，而不是同时保存 8 个 FP32 临时值；输出
阶段按 half2 转换、计算并舍入。

### 6.3 对齐证明

`cudaMalloc` 返回满足设备要求的对齐地址，代码又把 input、weight、output 在同一连续
allocation 内按 256 字节分段。向量路径还要求：

```text
float: hidden_dim % 4 == 0
half:  hidden_dim % 8 == 0
```

因此每行 stride 也是 16 字节的整数倍，不只是第 0 行对齐。任一条件不满足时回退标量
路径，尾部不需要危险的未对齐向量访问。

### 6.4 pack 上限与 dispatch

向量路径固定 256 threads，pack 数向上选择为 1、2、4、8，超过 8 就回退。最终规则：

```text
hidden_dim < 4096                -> scalar
不能被向量宽度整除              -> scalar
每线程需要超过 8 个 16B pack    -> scalar
其他情况                         -> float4 / half8
```

示例：

| 类型与维度 | vectors/row | packs/thread | 结果 |
|---|---:|---:|---|
| float 4096 | 1024 | 4 | float4 |
| float 8192 | 2048 | 8 | float4 |
| float 12288 | 3072 | 12 | scalar fallback |
| half 4096 | 512 | 2 | half8 |
| half 8192 | 1024 | 4 | half8 |
| half 12288 | 1536 | 6, 实例取 8 | half8 |

新平台 Event 显示 4096 的差异通常在约 1% 噪声内，所有已测回退小于 5%；NCU base
clock 则显示 half 4096 有 6.7% 至 9.7% 收益。保留 4096 阈值既没有触发回退条件，也
保留了 half 的可测收益。没有按某一次最快结果把 dispatch 细分成 rows 特判。

## 7. 正确性与安全性

### 7.1 官方测试

- 默认 `-O0`：13 float + 13 half，`26/26` 通过。
- `-O3 -lineinfo -arch=sm_120`：`26/26` 通过。
- 物理 GPU1：`26/26` 复验通过。

官方尺寸从 `1x1` 到 `8x4096`，还包含 `3x769` 和 `5x1536` 非向量尾部。

### 7.2 扩展矩阵

rows 为 `1、7、64、512`，hidden dimension 为：

```text
1, 31, 32, 33, 127, 128, 255, 256, 257,
768, 1024, 4096, 8192, 12288
```

再加入 `3x769` 与 `5x1536`，每种类型 58 组，共 116 组。数据覆盖零值、常量、随机
正负值和大动态范围。误差判据：

- float：`abs <= 2e-5` 或 `rel <= 2e-4`；
- half：`abs <= 2e-3` 或 `rel <= 2e-3`。

GPU0 和 GPU1 都是 `116/116` 通过。

### 7.3 Compute Sanitizer

GPU0 对全部 116 组分别运行：

| 工具 | 结果 |
|---|---|
| memcheck | 0 errors，0 越界/泄漏报告 |
| initcheck | 0 errors |
| racecheck | 0 hazards，0 errors，0 warnings |
| synccheck | 0 errors |

## 8. CUDA Event kernel-only A/B

编译参数为 `-O3 -lineinfo -arch=sm_120`。每个尺寸预热 20 次、计时 200 次、重复
5 组取中位数，并交替标量/向量顺序。以下是持久化日志中的一轮代表结果：

| Type | Rows | Hidden | Scalar (ms) | Vector (ms) | Speedup |
|---|---:|---:|---:|---:|---:|
| float | 32 | 2048 | 0.003761 | N/A | scalar dispatch |
| half | 32 | 2048 | 0.003751 | N/A | scalar dispatch |
| float | 8 | 4096 | 0.003805 | 0.003827 | 0.994x |
| half | 8 | 4096 | 0.003745 | 0.003793 | 0.987x |
| float | 64 | 4096 | 0.003820 | 0.003859 | 0.990x |
| half | 64 | 4096 | 0.003755 | 0.003759 | 0.999x |
| float | 64 | 8192 | 0.004106 | 0.003858 | 1.064x |
| half | 64 | 8192 | 0.003736 | 0.003793 | 0.985x |
| half | 64 | 12288 | 0.004106 | 0.003765 | 1.091x |
| float | 512 | 4096 | 0.006151 | 0.006150 | 1.000x |
| half | 512 | 4096 | 0.004104 | 0.003905 | 1.051x |

重复执行时，float `64x8192` 为 `1.064x` 至 `1.067x`；half `64x8192` 为
`0.985x` 至 `1.095x`。这说明 4 us 量级下 launch、boost 和系统状态会显著影响小差异。
因此 Event 用于真实热态时间，NCU base-clock 用于解释稳定的微架构差异，二者不能互相
替代。没有任何正式优化尺寸慢超过 5%。

## 9. NCU 硬件计数器

每份报告只采一个固定 kernel，预热后用 `--launch-skip 20 --launch-count 1`，并采集：

```text
SpeedOfLight
MemoryWorkloadAnalysis
LaunchStats
Occupancy
SchedulerStats
WarpStateStats
```

NCU 2025.3 把请求表拆成独立的 `MemoryWorkloadAnalysis_Tables`，因此另外对
`64x8192` 四条路径补采该 section。

### 9.1 float 标量与 float4

| Rows x Hidden | Path | Duration us | DRAM GB/s | DRAM % | Long Scoreboard % | Regs | Achieved Occ. | Waves/SM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8x4096 | scalar | 4.38 | 36.4 | 2.1 | 46.5 | 40 | 16.3% | 0.01 |
| 8x4096 | float4 | 4.48 | 34.6 | 2.0 | 61.3 | 40 | 15.5% | 0.01 |
| 64x4096 | scalar | 5.02 | 214.4 | 12.3 | 53.9 | 40 | 16.1% | 0.06 |
| 64x4096 | float4 | 4.83 | 222.0 | 12.8 | 62.7 | 40 | 16.1% | 0.06 |
| 64x8192 | scalar | 6.56 | 326.5 | 18.7 | 60.0 | 40 | 16.1% | 0.06 |
| 64x8192 | float4 | 6.24 | 342.8 | 19.6 | 72.8 | 56 | 15.8% | 0.09 |
| 512x4096 | scalar | 12.22 | 688.6 | 39.1 | 71.4 | 40 | 45.1% | 0.50 |
| 512x4096 | float4 | 11.81 | 712.5 | 40.5 | 77.6 | 40 | 47.6% | 0.50 |

### 9.2 half 标量与 half8

| Rows x Hidden | Path | Duration us | DRAM GB/s | DRAM % | Long Scoreboard % | Regs | Achieved Occ. | Waves/SM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8x4096 | scalar | 4.42 | 19.9 | 1.1 | 44.3 | 40 | 16.5% | 0.01 |
| 8x4096 | half8 | 4.03 | 20.3 | 1.2 | 56.5 | 36 | 16.3% | 0.01 |
| 64x4096 | scalar | 4.61 | 118.6 | 6.8 | 46.4 | 40 | 16.4% | 0.06 |
| 64x4096 | half8 | 4.32 | 125.1 | 7.2 | 58.1 | 36 | 16.1% | 0.06 |
| 64x8192 | scalar | 6.21 | 173.9 | 9.9 | 56.6 | 40 | 16.3% | 0.06 |
| 64x8192 | half8 | 5.57 | 193.0 | 11.0 | 69.5 | 48 | 15.7% | 0.08 |
| 512x4096 | scalar | 7.46 | 565.6 | 32.3 | 56.9 | 40 | 47.3% | 0.50 |
| 512x4096 | half8 | 7.26 | 579.6 | 33.1 | 67.3 | 36 | 48.0% | 0.50 |

NCU Duration 是 base clock、replay 环境下的解释性数据，不应拿来替代 Event 排名。它显示
向量路径提高有效 DRAM 速率，但绝大多数尺寸离峰值仍远，原因不是访问不合并，而是 grid
太小。RTX 5090 有 170 个 SM；8 行只有 8 blocks，64 行只有 64 blocks，NCU 明确报告
约 `0.01` 和 `0.06-0.09 waves/SM`。

### 9.3 请求合并与 spill

`64x8192` 的 L1TEX global 请求表：

| Path | Global load requests | Load sectors/request | Store requests | Store sectors/request | Local sectors |
|---|---:|---:|---:|---:|---:|
| float scalar | 49,152 | 4 | 16,384 | 4 | 0 |
| float4 | 8,192 | 16 | 4,096 | 16 | 0 |
| half scalar | 49,152 | 2 | 16,384 | 2 | 0 |
| half8 | 4,096 | 16 | 2,048 | 16 | 0 |

一个 sector 是 32 字节。float4/half8 的一个 warp 宽请求覆盖 16 个连续 sector，也就是
512 字节，符合 32 threads x 16 bytes。更高的 sectors/request 是宽请求的预期结果，不是
浪费。向量路径显著减少 load 指令请求数，并且 local load/store 都为零。

## 10. ptxas、cuobjdump 与 nvdisasm

`-Xptxas=-v` 的 sm_120 结果：

| Kernel instance | Registers/thread | Static shared | Spill loads/stores |
|---|---:|---:|---:|
| scalar, all block sizes | 40 | 8-36 B | 0 / 0 |
| float4 pack1/2/4/8 | 22 / 30 / 40 / 56 | 36 B | 0 / 0 |
| half8 pack1/2/4/8 | 25 / 36 / 48 / 80 | 36 B | 0 / 0 |

SASS 明确包含 `LDG.E.128` 和 `STG.E.128`。cuobjdump、nvdisasm 和 NCU 都没有发现
`LDL/STL` 或 local-memory traffic，证明源代码里的向量类型没有被编译器拆成伪向量路径，
寄存器数组也没有 spill。

## 11. Nsight Systems 官方端到端结果

O3 官方流程共 286 次 RMSNorm launch。NSYS 汇总：

| 项目 | 累计时间 |
|---|---:|
| CUDA API | 58.217 ms |
| GPU memcpy | 1.707 ms |
| RMSNorm kernels | 0.462 ms |

CUDA API 占比：

| API | API 时间占比 |
|---|---:|
| cudaFree | 38.1% |
| cudaMalloc | 30.0% |
| cudaMemcpy | 23.0% |
| cudaLaunchKernel | 4.9% |

这说明当前公开函数每次都分配、两次 H2D、一次 D2H、释放，kernel 优化只能影响端到端
时间中的很小一部分。一次连续 allocation 已经把三次 `cudaMalloc` 合成一次，但公开接口
仍不允许跨调用复用设备内存。

## 12. 系统级 A/B

固定 `64x8192`，正式 kernel 不变。下表是仓库外持久日志的一轮端到端中位数；括号为
第二轮观察到的范围，用于展示 pageable copy 和 UM 的运行间波动。

| 方案 | float us | half us | 正式合入 |
|---|---:|---:|---|
| 同步 allocation + pageable copy | 897.5 (848-897) | 561.9 (552-562) | 是，当前接口 |
| 持久 device workspace + pageable copy | 1268.8 (732-1269) | 399.1 (396-399) | 否，接口无 workspace |
| cudaMallocAsync / pool + pageable copy | 1029.2 (962-1029) | 575.6 (575-581) | 否，不稳定更快，且要求 CUDA 11.2+ |
| pinned host + 两条 H2D stream | 182.3 (182-198) | 98.9 (99-134) | 否，输入是 pageable std::vector |
| Managed + MemAdvise + 双向 prefetch | 1327.1 (1327-4021) | 2675.6 (2676-2989) | 否，明显回退 |
| pinned fixed-sequence CUDA Graph | 195.1 (194-195) | 100.4 (100-115) | 否，固定地址/尺寸与持久状态 |

### 12.1 为什么 pinned 有效但不能直接合入

NSYS 中 float pageable H2D/D2H 平均约 `110.0/80.7 us`；pinned 方案约
`43.7/79.5 us`，CUDA kernel 仍只有约 `2.8 us`。pinned buffer 避免 pageable staging，
也允许两次 H2D 进入不同 stream，但公开输入由调用者拥有的 `std::vector` 给出。函数内部
每次先申请 pinned buffer 再复制一次，只会把成本挪到另一处。真正的收益要求调用者长期
持有 pinned buffer 或采用异步 device API。

### 12.2 为什么 memory pool 未胜出

NSYS 的 float 中位 API 时间约为：

```text
cudaMallocAsync       114.5 us
cudaFreeAsync           7.5 us
cudaMemcpyAsync       329.2 us
cudaStreamSynchronize 169.6 us
```

memory pool 减少 free 成本，却没有解决 pageable copy 和每次返回 CPU output 前的同步。
它还要求 CUDA 11.2，而任务假设要求 CUDA 11.0 兼容，所以即使个别运行更快也不能合入。

### 12.3 为什么 Managed Memory 更慢

Managed float 报告累计迁移约 `65.044 MB H2D`、`62.915 MB D2H`，有 1088 次 GPU
page faults；half 为 `32.522/31.457 MB` 和 798 次 GPU faults。`cudaMemAdvise` 和
prefetch 没有消除数据必须在 CPU 与 GPU 间往返的事实，反而引入页迁移管理。当前模式是
一次性、方向明确、远小于显存容量的数据流，显式 copy 更适合。

### 12.4 为什么 Graph 没有进入正式代码

Graph profile 中 `cudaGraphLaunch` 中位 host API 时间约 18 us，而
`cudaStreamSynchronize` 约 158 us。Graph 适合固定地址、固定尺寸、重复执行相同操作图；
本题每次接收不同 `std::vector` 地址和尺寸，并必须同步返回 CPU output。内部维护 graph
cache 会引入设备、stream、尺寸、线程安全和析构问题，也违反无隐式全局状态的约束。

## 13. 为什么没有实现 cp.async 双缓冲

NCU 确实显示 Long Scoreboard 高，DRAM 吞吐未满，但完整判断还要看结构：

1. 8 或 64 行只产生 8 或 64 blocks，无法填满 170 个 SM。
2. `cp.async` 只能改变 block 内 global-to-shared 搬运，不能增加 grid 的 block 数。
3. 每个 input 元素在平方阶段只做一次 FMA，计算太薄，难以覆盖流水线首尾与同步成本。
4. 输出必须等待整行规约完成。若 shared tile 不能保留整行，输出阶段仍要再次读 input。
5. 若把整行留在 shared memory，float 8192 需要 32 KiB，12288 需要 48 KiB，每 block
   shared 占用会显著限制 residency；寄存器缓存路径已经无 spill 地保留线程负责的数据。
6. float4 8192 已用 56 regs，理论 occupancy 66.7%；half8 8192 用 48 regs，理论
   occupancy 83.3%，继续增加 pipeline 状态并非免费。

因此观测到的 Long Scoreboard 主要来自低并行 grid 下无法隐藏的依赖，而不是“不使用
异步加载”这一单一原因。当前证据不满足实现原型的完整前提，正式 kernel 不增加复杂度。

## 14. 最终取舍

| 方案 | 结论 |
|---|---|
| 两级 warp/block reduction | 合入，通用且 Sanitizer 干净 |
| float4/half8 + register cache | 合入，真实 128-bit 指令、零 spill |
| scalar fallback | 合入，覆盖任意维度与尾部 |
| 单次连续 cudaMalloc | 合入，不改接口即可减少分配次数 |
| persistent workspace | 研究有效，但需要公开 API |
| cudaMallocAsync/pool | 不合入，兼容性与稳定收益不满足 |
| pinned multistream | 不合入，需要调用者数据生命周期配合 |
| Managed + advise/prefetch | 不合入，实测明显更慢且有大量迁移/fault |
| CUDA Graph | 不合入，需要固定地址、尺寸和持久状态 |
| cp.async shared pipeline | 不实现，不能解决主要的 grid 并行度问题 |

成功标准已经达到：官方与扩展正确性通过、Sanitizer 全干净、正式向量尺寸未出现超过 5%
回退、NCU/NSYS 可以解释最终选择。排行榜仍取决于官方统一环境、尺寸、编译参数和计时
范围，本文不把本机结果外推成名次承诺。

## 15. 资料与复现入口

本地讲义的学习顺序：

1. `并行编程与CUDA编程入门.pdf`
2. `2_性能模型与逐元素优化.pdf`
3. `2026夏季_内存模型与规约优化.pdf`
4. `向量规约.pdf`
5. `4_分块与不规则访存.pdf`
6. `2026夏季_异步并行-底层控制与系统优化.pdf`

外部一手资料：

- [RMSNorm, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
- [Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)

报告目录中的 `environment.txt`、`event_kernel_ab.csv`、`system_ab.txt`、
`ncu_commands.sh`、`nsys_commands.sh`、`.ncu-rep`、`.nsys-rep`、CSV、ptxas 和 SASS
文件提供了本文数字的原始证据。
