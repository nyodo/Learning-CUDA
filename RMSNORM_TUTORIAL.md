# 从公式到 NCU/NSYS：RMSNorm CUDA 完整教程

这份教程解释 [`src/kernels.cu`](src/kernels.cu) 中 RMSNorm 的设计。重点不是背一段
kernel，而是建立一套可迁移的方法：先读算子依赖，再做流量模型，设计正确的规约，最后
用正确工具验证正确性、机器码、kernel 性能和系统性能。

建议第一次按顺序阅读；第二次对照代码，把每个设计决定写成自己的理由。

## 1. 先把数学问题说清楚

输入布局是 row-major：

```text
input  : [rows, hidden_dim]
weight : [hidden_dim]
output : [rows, hidden_dim]
```

每一行独立计算：

```text
sum_square = sum_j input[j]^2
mean_square = sum_square / hidden_dim
inv_rms = rsqrt(mean_square + eps)
output[j] = input[j] * inv_rms * weight[j]
```

RMSNorm 与 LayerNorm 的关键差别是它不减均值，也不需要方差的第二个统计量。但输出仍
依赖整行的 `sum_square`，所以它不是完全独立的逐元素算子，而是：

```text
一次行规约 + 一次逐元素变换
```

最朴素的 CPU 版本可以写成：

```cpp
for (size_t row = 0; row < rows; ++row) {
  float sum = 0.0f;
  for (size_t col = 0; col < hidden_dim; ++col) {
    const float x = static_cast<float>(input[row * hidden_dim + col]);
    sum += x * x;
  }

  const float inv_rms =
      1.0f / std::sqrt(sum / static_cast<float>(hidden_dim) + eps);

  for (size_t col = 0; col < hidden_dim; ++col) {
    const float x = static_cast<float>(input[row * hidden_dim + col]);
    const float w = static_cast<float>(weight[col]);
    output[row * hidden_dim + col] = x * inv_rms * w;
  }
}
```

两个循环之间存在整行依赖。CUDA 设计的核心就是并行完成第一个循环，把一个标量结果
广播给 block，再并行完成第二个循环。

## 2. 数值精度为什么统一用 FP32

输入允许 float 或 half，但平方和始终使用 FP32：

- float 直接参与 FP32 FMA；
- half 用 `__half2float` 或 `__half22float2` 转成 FP32；
- 除以 `hidden_dim`、加 `eps`、`rsqrtf` 都在 FP32；
- input、inv_rms、weight 的乘法在 FP32；
- half 只在最后写回时舍入。

若用 FP16 累加 4096 或 12288 个平方项，舍入误差会迅速积累，大值还可能溢出。输入
类型决定存储格式，不等于中间计算也必须保持该类型。

代码用 `toFloat<T>` 和 `fromFloat<T>` 把类型差异集中起来。这样标量规约主体不需要为
float/half 复制两套控制逻辑。

## 3. 写 kernel 前先做流量模型

### 3.1 标量路径

没有保存整行 input 时，输出阶段必须再次读取：

```text
read input for sum       sizeof(T)
read input for output    sizeof(T)
read weight              sizeof(T)
write output             sizeof(T)
total                    4 * sizeof(T)
```

float 为 16 B/element，half 为 8 B/element。

### 3.2 寄存器缓存路径

第一次读 input 时保留线程负责的值：

```text
read input               sizeof(T)
read weight              sizeof(T)
write output             sizeof(T)
total                    3 * sizeof(T)
```

float 为 12 B/element，half 为 6 B/element，逻辑请求流量减少 25%。

### 3.3 这为什么是 memory-bound 算子

一个元素主要做一次平方 FMA 和两次乘法，计算量只有几个 FLOP。即使按约 4 FLOP 估算，
float 标量的算术强度也只有约 `4/16 = 0.25 FLOP/B`，缓存路径约 `0.33 FLOP/B`。
这远低于现代 GPU 需要用满计算单元的强度，所以优化重点是：

- 合并访问；
- 降低请求和指令数量；
- 避免第二次 input 读取；
- 不因缓存导致 spill；
- 给 GPU 足够多的 blocks 隐藏延迟。

最后一项由 `rows` 决定。再漂亮的 block 内流水线，也不能把 8 行变成 170 个可并行 blocks。

## 4. 为什么是一行一个 block

最终映射：

```text
block 0 -> row 0 -> inv_rms[0]
block 1 -> row 1 -> inv_rms[1]
...
```

优点：

1. 行之间本来就独立，直接映射到 blocks。
2. 整行平方和只在一个 block 内规约。
3. 不需要原子加、跨 block 同步或中间 buffer。
4. 归一化标量可以用 shared memory 广播。
5. 相邻线程访问相邻列，天然适合合并访存。

为什么不是一行一个 warp？对 4096/8192 维，32 个线程每线程要串行处理太多元素，内存
级并行度不足。

为什么不是一行多个 blocks？多个 blocks 会各自得到部分和，必须用原子操作、第二个
kernel 或 cooperative grid 同步合并。对当前维度，一行一个 256-thread block 已能完成，
额外机制增加的 launch 和中间流量很难偿还。

这一设计的弱点也必须承认：当 rows 小于 SM 数时，grid 严重 underfill。NCU 在 RTX
5090 上测得 8 行约 `0.01 wave/SM`，64 行约 `0.06-0.09 wave/SM`。这是小 batch 的
结构性上限。

## 5. 线程内遍历

标量路径中，每个线程处理：

```cpp
for (size_t col = threadIdx.x; col < hidden_dim; col += BlockSize) {
  ...
}
```

例如 `hidden_dim=769`、`BlockSize=256`：

- thread 0 处理 0、256、512、768；
- thread 1 处理 1、257、513；
- thread 255 处理 255、511、767。

每一列被恰好覆盖一次。最后一次循环中部分线程已经越界，但它们只是不读取数据，仍会
继续进入 block reduction 和同步。

block size 选择：

```text
work items <= 32  -> 32
             <=64 -> 64
            <=128 -> 128
otherwise         -> 256
```

全部是完整 warp，布局不会因 partial warp 改变。

## 6. warp shuffle 规约

每线程先得到一个局部平方和。一个 warp 内执行：

```cpp
for (int offset = 16; offset > 0; offset >>= 1) {
  value += __shfl_down_sync(0xffffffffu, value, offset);
}
```

可以把它想成树形合并：

```text
offset 16: lane 0 加 lane 16
offset  8: lane 0 加 lane 8 的部分和
offset  4: 再合并
offset  2: 再合并
offset  1: lane 0 得到整个 warp 的和
```

shuffle 让 lane 直接交换寄存器值，避免每一步都写 shared memory 和同步。

### mask 为什么可以是 0xffffffff

mask 表示哪些 lane 必须共同执行该 shuffle。当前 block 大小总是 32 的倍数，而且没有
让越界线程提前 return，所以每个 warp 的 32 个 lane 都参加。越界线程的数值是 0，
但控制流仍合法。

危险写法是让部分 lane 提前退出，然后剩余 lane 仍用 full mask。那会把未参与 lane 列入
同步契约，行为未定义。

## 7. 从 warp reduction 到 block reduction

一个 256-thread block 有 8 个 warps：

1. 每个 warp 的 lane 0 写一个 `warp_sums[warp]`。
2. 全 block `__syncthreads()`。
3. 首 warp 的前 8 个 lane 读 8 个部分和，其余 lane 读 0。
4. 首 warp 再做一次 full-warp shuffle。

shared 数组只需 `BlockSize / 32` 个 float。256 threads 时是 8 个 float，也就是 32 B。
这比把每线程局部和都放 shared memory 更小，也更少同步。

`blockReduceSum` 返回后，只有线程 0 的 `total` 有意义。线程 0 计算 `inv_rms` 写到一个
shared float，再同步一次。第二次同步不能省略，否则其他线程可能在写入完成前读取旧值。

## 8. 标量路径为什么必须读 input 两次

平方和完成前不知道 `inv_rms`。若不保存 input，输出阶段只能重新从 global memory 读取。
保存整行有三种常见位置：

- 寄存器：最快，但每线程容量有限，会增加寄存器压力；
- shared memory：block 可共享，但整行占用大，会降低 residency；
- global memory：就是再次读取，没有节省。

标量路径选择第二次读取，换取任意维度、低寄存器风险和简单尾部处理。它是可靠基线，
不是应该删除的“慢代码”。优化路径必须随时能回退到它。

## 9. float4 路径逐步理解

### 9.1 16 字节 pack

`float4` 含 4 个 float，总共 16 B。线程每次加载一个 pack：

```cpp
const float4 value = input4[vector_row_offset + vec];
```

SASS 已确认它生成 `LDG.E.128`，写回生成 `STG.E.128`。

### 9.2 寄存器数组

```cpp
float4 cached[PacksPerThread];
```

第一次循环：加载、缓存、把四个分量加入平方和。规约后，第二个循环直接读取 `cached`，
只加载 weight，然后写 output。这样省掉第二次 input load。

编译器会为 `PacksPerThread=1/2/4/8` 分别实例化，循环完全展开。运行时只在四个已知实例
之间 dispatch，不使用运行时长度的局部数组。

### 9.3 代价

pack8 的 float4 实例使用 56 regs/thread，理论 occupancy 为 66.7%；标量使用 40 regs，
理论 occupancy 为 100%。但 occupancy 不是目标函数。如果减少一次 input 请求后 kernel
更快，66.7% 可以是正确交换；若发生 spill 或速度回退，就应降低 pack 上限。

## 10. half8 路径逐步理解

8 个 half 也是 16 B。代码用 `uint4` 完成宽加载，因为它恰好是四个 32-bit 字段：

```text
uint4.x -> 2 x half
uint4.y -> 2 x half
uint4.z -> 2 x half
uint4.w -> 2 x half
```

union `Half2Bits` 只做位级重解释，再使用 `__half22float2` 转换为 FP32。输出阶段使用
`__floats2half2_rn` 两个一组地舍入。

为什么不把平方和也用 half2 指令一直保留 FP16？因为规约精度比转换开销更重要，尤其在
大 hidden dimension 下。half8 解决的是内存请求宽度，不是降低中间精度。

pack8 的 half8 实例使用 80 regs/thread，但仍为零 spill。实际 8192 使用 pack4，48 regs；
4096 使用 pack2，36 regs；12288 选择 pack8 实例，部分 pack 由边界条件置零。

## 11. 对齐必须证明两次

向量类型不是“写出来就一定合法”。至少要证明：

1. allocation 和子缓冲起点 16 B 对齐；
2. 每一行起点也 16 B 对齐。

正式 host 代码执行一次 `cudaMalloc`，再用 256 B 对齐 offset 划分 input、weight、output。
第二条由 dispatch 条件保证：

```text
float hidden_dim % 4 == 0 -> row bytes % 16 == 0
half  hidden_dim % 8 == 0 -> row bytes % 16 == 0
```

如果 hidden=769，第一行即使对齐，下一行也不满足向量 stride；该尺寸自动走标量路径。

验证对齐不能只看 C++ 类型。最终还要看 SASS 是否真的有 128-bit load/store，并用
Compute Sanitizer 检查未对齐或越界访问。

## 12. 为什么 host 端只做一次 cudaMalloc

逻辑上需要 input、weight、output 三块设备内存。分别 `cudaMalloc` 会引入三次昂贵 API。
代码先计算：

```text
input_offset  = 0
weight_offset = align256(input_bytes)
output_offset = align256(weight_offset + weight_bytes)
total_bytes   = output_offset + output_bytes
```

然后一次分配、三个 typed pointer 绑定。这样：

- 分配调用从三次降为一次；
- 子缓冲满足向量对齐；
- 只需一次 `cudaFree`；
- 不引入跨调用全局状态。

每次调用仍然分配一次，是公开同步接口的限制。若要消除它，正确做法是公开 workspace 或
device-pointer API，而不是在函数内部藏一个无法管理并发和多设备的 static pointer。

## 13. dispatch 是风险控制，不只是性能开关

当前向量路径要求：

1. `hidden_dim >= 4096`；
2. hidden 能被 pack 元素数整除；
3. 256 threads 下每线程最多 8 个 pack。

否则走标量路径。

为什么不让所有维度都向量化？小维度时 launch 和规约固定成本占主导，寄存器缓存没有
足够流量可省；非整除维度需要复杂尾部；超大维度需要更多寄存器，可能 spill。dispatch
把优化限制在已验证区域，通用性由标量路径承担。

为什么 4096 没因部分 Event 数据持平就删除？因为两轮 Event 的差异都小于 1.3%，没有
达到 5% 回退阈值，而 base-clock NCU 的 half 4096 显示 6.7% 至 9.7% 改善。阈值选择应
结合重复测量和解释性指标，不能追逐一次几纳秒差异。

## 14. 验证金字塔

### 第一层：参考值

CPU 参考按题目公式计算。float 和 half 分别使用明确误差标准，而不是只检查有限值。

### 第二层：尺寸与数据分布

必须覆盖：

- 小于 warp、等于 warp、跨过 warp 的 31/32/33；
- 跨过 block 选择点的 127/128、255/256/257；
- 非向量尾部 769；
- 模型常见维度 768/1024/4096/8192/12288；
- rows 1/7/64/512；
- 零值、常量、随机正负、大动态范围。

### 第三层：Sanitizer

- memcheck：越界、未对齐、泄漏；
- initcheck：未初始化设备数据；
- racecheck：shared/global 数据竞争；
- synccheck：barrier 和 warp 同步错误。

本实现的 116 组矩阵在四种工具下都是零错误。

### 第四层：机器码

- `-Xptxas=-v` 看 registers、shared、spill；
- `cuobjdump --dump-resource-usage` 交叉核对资源；
- `cuobjdump --dump-sass` 和 `nvdisasm` 看 `LDG.E.128/STG.E.128`；
- 搜索 `LDL/STL`，并用 NCU local sectors 再确认无 spill。

### 第五层：Event、NCU、NSYS

它们回答不同问题，不能互相替代。

## 15. CUDA Event 应该怎样读

Event 放在同一 CUDA stream 上，测量 GPU 时间线中的 kernel，不包含 host 计时误差。流程：

```text
预热 20 次
记录 start event
launch 200 次
记录 stop event
同步 stop
总时间 / 200
重复 5 组取中位数
```

标量和向量的测试顺序交替，减少温度和 boost 对固定顺序的偏差。

Event 最适合回答“热态 kernel 哪个更快”。它不告诉你为什么快，也不包括 malloc 和传输。
对于约 4 us kernel，1% 只是约 40 ns，必须报告重复区间，不能只保留最快一次。

## 16. NCU 应该怎样读

NCU 会 replay kernel 多次以采集互斥硬件计数器。本实验每份报告 17 passes，memory table
为 31 passes，并使用 base clock。因此 NCU Duration 与 Event 不同是正常现象。

### 16.1 Duration 与 throughput

`Duration` 用于同一 NCU 配置下比较。`DRAM Throughput %` 表示相对峰值利用率；
`DRAM GB/s` 是实际速率。64x8192 float4 从 326.5 提升到 342.8 GB/s，Duration 从
6.56 降到 6.24 us。

不能看到 DRAM 只有 20% 就直接断言访存不合并。若 grid 只有 64 blocks，整个 GPU 本来
就没有足够并行工作把所有 memory partitions 长时间填满。

### 16.2 Long Scoreboard

Long Scoreboard 表示 warp 等待 L1TEX 相关 load dependency。向量路径数值可能更高，
因为每线程缓存后存在更长的数据依赖链，而活动 warps 又少。

高 Long Scoreboard 只说明“warp 经常等 load”，不自动推出“应该用 cp.async”。还要问：

- grid 是否足够大；
- 访问是否合并；
- DRAM/L2 是否饱和；
- registers 和 occupancy 是否有余量；
- shared tile 是否被复用；
- 异步流水能否跨过算法依赖。

### 16.3 waves 与 occupancy

64x8192 float4 的理论 occupancy 66.7%，achieved 约 15.8%。这不是简单的“寄存器太多
导致只跑 15.8%”，因为整个 grid 只有 64 blocks，而 GPU 有 170 SM。`waves/SM=0.09`
更直接揭示 underfill。

512x4096 时 waves/SM 到 0.5，achieved occupancy 约 45-48%，DRAM 也提高到 float
约 700 GB/s。rows 改变比调整一两条 block 内指令更能影响延迟隐藏。

### 16.4 sectors/request

NCU 的 sector 是 32 B。标量 float 每个 warp load 请求覆盖 4 sectors，half 覆盖
2 sectors；16B/thread 的向量请求覆盖 16 sectors。对连续地址来说，这正好是预期宽度。

判断浪费不能机械地认为 sectors/request 越低越好。应先算一次 warp 指令理论上请求多少
字节，再对比实际 sector 数，同时观察请求数和 local traffic。这里向量路径请求数显著
下降，local sectors 为 0。

## 17. NSYS 应该怎样读

NSYS 看完整 CPU/GPU 时间线，不需要像 NCU 那样 replay。它回答：

- 哪些 CUDA API 最耗 host 时间；
- H2D、kernel、D2H 的顺序和重叠；
- malloc/free 是否频繁同步；
- Graph launch 是否减少提交；
- Managed Memory 是否出现 page fault 和迁移。

官方流程中 286 个 kernels 累计只有 0.462 ms，而 CUDA API 累计 58.217 ms。仅
`cudaMalloc/cudaFree/cudaMemcpy` 就占 API 时间 91.1%。这解释了为什么 kernel 提升
10% 几乎不改变完整函数的数百微秒时间。

NSYS 会插桩，profile 下的绝对中位数可能比无 profile 慢。用它分解原因，用正常运行的
host timer/Event 报告性能，不要把插桩时间当排行榜时间。

## 18. 异步优化逐项判断

### 18.1 异步加载与 cp.async

典型适用场景是：global tile 搬到 shared，计算 tile A 时预取 tile B，并且 tile 会被
多个线程或多次计算复用。

RMSNorm 第一阶段每元素只做一次平方 FMA，输出又必须等待整行规约。若只做小 tile，
规约后仍需第二次读 input；若把整行放 shared，8192 float 要 32 KiB，12288 要 48 KiB，
residency 会下降。寄存器缓存已经更直接地保留线程自己的数据。

NCU 虽有高 Long Scoreboard，但主要短板是 8/64 blocks 无法填满 170 SM。`cp.async`
不能创造更多 rows，所以本轮没有实现只为展示 API 的复杂原型。

### 18.2 多流

单次调用依赖图：

```text
H2D input  ----\
                +--> kernel --> D2H output --> return
H2D weight ----/
```

两个 H2D 可以进入两条 stream，但只有 pinned host memory 才能可靠异步。公开输入是
pageable `std::vector`，且 D2H 必须在函数返回前完成。真正的流水化需要跨多个请求，
让请求 B 的传输与请求 A 的计算重叠，并由上层持有 pinned buffer 和 stream。

实验中 pinned 多流比同步 pageable 快很多，证明优化方向有效；它同时证明正式接口缺少
实现该方向所需的数据生命周期，而不是证明在函数内部多建两条 stream 就足够。

### 18.3 cudaMallocAsync 与 memory pool

它能降低重复 allocation/free 的部分成本，但：

- 不消除 pageable copy；
- 每次仍同步返回 CPU output；
- 实测没有稳定优于当前同步路径；
- API 从 CUDA 11.2 开始，超出 CUDA 11.0 兼容目标。

因此只保留研究数据。

### 18.4 persistent workspace

这是最合理的系统 API 改造方向之一：由调用者提供 device buffer 或显式 allocator，跨调用
复用。它需要新公开接口，不能在当前函数内安全隐藏：

- 多线程会竞争；
- 多 GPU/context 的归属不明确；
- 尺寸增长需要重分配；
- 退出和异常路径可能泄漏；
- 默认 stream 与其他 stream 的同步语义不清晰。

### 18.5 Managed Memory、MemAdvise 与 prefetch

这些机制非常适合访问模式复杂、CPU/GPU 多阶段共享、数据超显存或显式迁移难维护的
程序。本题的数据方向却很简单：CPU 输入到 GPU，一次计算，CPU 读取输出。

Managed A/B 即使用了 preferred location、read-mostly 和双向 prefetch，NSYS 仍观察到
数十 MB 迁移和数百至上千 faults，端到端明显变慢。prefetch 改变迁移发起方式，不会让
必须传输的字节消失。

### 18.6 CUDA Graph

Graph 适合固定地址、固定尺寸、重复执行相同操作序列。A/B 中 graph 使用 persistent
device/pinned buffers，因而能把重复 launch 稳定在约 100-195 us。但这同时改变了数据
生命周期，不能把全部收益归因于 Graph。

当前 API 每次 host vector 地址和尺寸可变，还要同步返回。要正确采用 Graph，应设计一个
长期对象，显式拥有 buffers、graph exec、stream、device 和尺寸 cache，而不是塞进一个
无状态模板函数。

## 19. 为什么“异步 API 已调用”不等于“延迟已隐藏”

判断异步优化是否有效，要在时间线上看到真正重叠：

```text
copy B 与 kernel A 同时发生
或
load tile B 与 compute tile A 同时发生
```

仅把 `cudaMemcpy` 改成 `cudaMemcpyAsync`，随后立刻 `cudaStreamSynchronize`，通常只是
改了调用形式。仅调用 prefetch，随后 kernel 因 page migration 等待，也没有隐藏延迟。

本次 A/B 的教学价值就在这里：

- pinned multistream 显著改善传输路径，但受接口限制；
- pool 没解决传输和同步；
- Managed prefetch 产生大量迁移/fault；
- Graph 减少重复提交，却要求固定生命周期；
- kernel 的首要结构瓶颈是小 grid，而不是缺少一个异步关键字。

## 20. 常见错误清单

### 20.1 partial warp 还使用 full mask

保证整个 warp 参与，或构造正确 active mask。不要让部分 lane 提前 return 后继续 full-mask
shuffle。

### 20.2 忘记 inv_rms 写入后的同步

线程 0 写 shared 标量后，其他线程读取前需要 block barrier。

### 20.3 half 输入就用 half 累加

存储精度和累计精度是两件事。大规约应优先 FP32 累加。

### 20.4 只证明 allocation 起点对齐

还必须证明 row stride 对齐，否则第二行就可能非法。

### 20.5 源代码用了 float4 就宣称向量化

必须看 SASS 的 128-bit 指令，并检查 local spill。

### 20.6 只跑一个尺寸

rows 决定 grid 并行度，hidden 决定每线程循环、流量和寄存器需求。一个赢家不代表全矩阵。

### 20.7 用 NCU Duration 做排行榜

NCU 会 replay 和控制时钟。排名使用无 profiler 的 Event/官方计时，NCU 用于解释。

### 20.8 把 occupancy 当唯一目标

更高 occupancy 只代表更多潜在驻留 warps。若工作量不够、访问不合并或指令更多，它不会
自动更快。最终目标是正确前提下的实际时间。

## 21. 推荐复现顺序

### 21.1 默认官方测试

CUDA 13 与当前预编译 tester 组合需要 `/tmp` ABI shim：

```bash
cd ~/Learning-CUDA
make clean
SKIP_ATTENTION=1 make VERBOSE=true \
  EXTRA_LIBS=/tmp/codex-cuda-compat.o
```

正常匹配 tester ABI 的 CUDA 环境不需要 shim。

### 21.2 目标架构构建

```bash
make clean
make build \
  CFLAGS='-std=c++17 -O3 -lineinfo -arch=sm_120' \
  EXTRA_LIBS=/tmp/codex-cuda-compat.o
```

`sm_120` 只用于 RTX 5090 实验；正式源代码没有使用 CUDA 13 专属能力。

### 21.3 Sanitizer

```bash
compute-sanitizer --tool memcheck ./your_extended_test
compute-sanitizer --tool initcheck ./your_extended_test
compute-sanitizer --tool racecheck ./your_extended_test
compute-sanitizer --tool synccheck ./your_extended_test
```

### 21.4 SASS

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  -DPLATFORM_NVIDIA -Xptxas=-v -c src/kernels.cu -o /tmp/kernels.o
cuobjdump --dump-resource-usage /tmp/kernels.o
cuobjdump --dump-sass /tmp/kernels.o > /tmp/kernels.sass
grep -E 'LDG.*128|STG.*128|LDL|STL' /tmp/kernels.sass
```

### 21.5 NCU

不要直接对包含数百 launch 的完整 tester 用 `--set full`。建立一个只预热并 launch 一个
固定 kernel 的 driver：

```bash
sudo env CUDA_VISIBLE_DEVICES=0 \
  numactl --cpunodebind=0 --membind=0 \
  ncu --clock-control base \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section LaunchStats \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --launch-skip 20 --launch-count 1 \
  -o /tmp/rmsnorm \
  ./profile_driver float-vector 64 8192
```

需要请求表时另加 `MemoryWorkloadAnalysis_Tables`。

### 21.6 NSYS

```bash
CUDA_VISIBLE_DEVICES=0 SKIP_ATTENTION=1 \
numactl --cpunodebind=0 --membind=0 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  --cuda-memory-usage=true \
  --cuda-um-cpu-page-faults=true \
  --cuda-um-gpu-page-faults=true \
  -o /tmp/rmsnorm-e2e \
  ./test_kernels

nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,um_total_sum \
  /tmp/rmsnorm-e2e.nsys-rep
```

Graph 节点级追踪需要 `--cuda-graph-trace=node`；默认 `graph` 粒度只记录整张 graph，
kernel/memcpy 节点汇总可能为空。

## 22. 建议亲手完成的练习

1. 画出 hidden=769、block=256 时 thread 0、1、255 的列索引。
2. 用 64 个线程手算两级规约，标出两次 `__syncthreads()` 的作用。
3. 暂时把 half 平方和改成 FP16，比较 4096/12288 的误差，再恢复。
4. 把向量阈值改为 2048、4096、8192，在固定条件下重复 Event，不看单次最快值。
5. 把最大 pack 从 8 改为 4，观察 float8192 回退、寄存器和 occupancy 的变化。
6. 在 SASS 中分别找到 input、weight 的 `LDG.E.128` 和 output 的 `STG.E.128`。
7. 用 NCU 对比 rows=8、64、512，解释 DRAM % 为什么随 rows 大幅变化。
8. 用 NSYS 打开 Managed 报告，沿时间线找到 prefetch、fault/migration、kernel 和同步。
9. 设计一个显式 `RmsNormWorkspace` API，列出它在多线程、多 GPU 和变尺寸下的所有权。
10. 回答：为什么 float4 理论 occupancy 下降后仍可能更快？答案必须同时提到请求流量、
    spill、grid 大小和实际 Duration。

## 23. 最终心智模型

这次实现可以浓缩成九步：

1. 从公式识别“规约后才能输出”的依赖。
2. 用一行一个 block 把依赖限制在 block 内。
3. 用线程内累加、warp shuffle、首 warp 汇总完成规约。
4. 用完整 warp 和零贡献线程保证 shuffle mask 合法。
5. 用 FP32 保证 float/half 累加精度。
6. 用标量路径覆盖所有合法尺寸。
7. 用 128-bit pack 和寄存器缓存减少 input 重读，同时用 pack 上限防 spill。
8. 用测试、Sanitizer、SASS、Event、NCU、NSYS 分别验证不同层面。
9. 只在数据生命周期允许时采用异步机制，不把 API 名称当作性能证据。

真正成熟的 CUDA 优化不是堆更多特性，而是知道瓶颈在哪一层、证据回答了什么、哪些复杂度
在当前约束下没有收益。这个 RMSNorm 已经把 kernel 层做到了一个扎实、可解释的版本；若
未来允许重设 API，优先研究持久 device workspace、调用者持有的 pinned buffer、stream
参数和跨请求流水，而不是继续在单个低并行 kernel 里增加异步搬运状态。
