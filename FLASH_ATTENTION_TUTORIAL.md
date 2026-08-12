# Flash Attention 前向实现教程：从公式到 RTX 5090 内核

本文解释 `src/kernels.cu` 中 `flashAttention<float>` 与
`flashAttention<half>` 的完整设计过程。目标不是背诵某一份实现，而是学会从算子语义、
数值稳定性、GPU 执行模型和 profiler 证据逐步推导实现。

本实现只包含前向计算，支持 causal attention、GQA 和非方形 Q/K 序列；不包含反向、
dropout 和任意 attention mask。论文和官方文档用于理解算法，代码是独立编写的，没有
查看或复制 FlashAttention、CUTLASS、Triton 等项目的 kernel 源码。

## 1. 先明确算子契约

输入在本作业中的连续布局为：

- `Q: [B, Tq, Hq, D]`
- `K: [B, Tk, Hkv, D]`
- `V: [B, Tk, Hkv, D]`
- `O: [B, Tq, Hq, D]`

对第 `b` 个 batch、第 `hq` 个 query head、第 `i` 个 query，先映射 GQA head：

```text
group_size = Hq / Hkv
hkv        = hq / group_size
```

然后计算：

```text
s(i,j) = dot(Q[b,i,hq,:], K[b,j,hkv,:]) / sqrt(D)
p(i,j) = exp(s(i,j)) / sum_j exp(s(i,j))
O[b,i,hq,:] = sum_j p(i,j) * V[b,j,hkv,:]
```

causal 使用左上角对齐语义：合法 key 满足 `j <= i`，所以：

```text
key_end = min(Tk, i + 1)
```

这点对 `Tq != Tk` 尤其重要。它不是右下角对齐，也不是按 `Tq - Tk` 平移后的三角形。

Host 包装还承担以下契约：

1. 负维度、非法 GQA 比例、输入容器过小和 `D > 256` 会报错。
2. `B/Tq/Hq/D` 任一为零时输出清空并返回。
3. `Tk == 0` 时输出与 Q 等大且全零。
4. 元素数和字节数的乘加都检查 `size_t` 溢出。
5. Q/K/V/O 放入一次 `cudaMalloc` 得到的连续区域，每段起点按 256 字节对齐。
6. 三次 H2D、一次 kernel、一次 D2H 和释放都检查 CUDA 返回值。

## 2. 为什么朴素 attention 昂贵

朴素实现常分成三步：

```text
S = Q * K^T
P = softmax(S)
O = P * V
```

若将 `S` 和 `P` 写入 HBM，它们的大小都是 `O(Tq * Tk)`。长序列时，中间矩阵的读写
会远大于 Q/K/V/O 本身。FlashAttention 的核心不是改变数学结果，而是把 key 维分块，
在片上保存 softmax 状态，避免把完整 `S` 和 `P` 落到 HBM。

粗略计算量（忽略 softmax 标量操作）是：

```text
QK: 2 * B * Hq * Tq * Tk * D FLOP
PV: 2 * B * Hq * Tq * Tk * D FLOP
总计约 4 * B * Hq * Tq * Tk * D FLOP
```

如果每个 query row 都从 HBM 重读 K/V，算法仍可能受缓存和访问形状限制。GQA 提供了
额外复用机会：属于同一个 KV head 的多个 query head 可以共享同一块 K/V。

## 3. 稳定 softmax

直接计算 `exp(score)` 容易溢出。稳定三遍形式是：

```text
m = max_j score_j
l = sum_j exp(score_j - m)
o = sum_j exp(score_j - m) * V_j
O = o / l
```

它需要重算 score，但计算顺序清晰、与严格参考容易对齐。代码中的 float 通用路径和
shared-KV 路径采用这个形式。

### 3.1 Online softmax 的合并公式

为了只扫描 key tile 一次，对已经处理的前缀保存三项状态：

```text
m: 当前最大 score
l: 以 m 为基准的指数和
o: 以 m 为基准、尚未除以 l 的输出向量
```

一个新 tile 独立得到 `(mt, lt, ot)` 后，令：

```text
m_new = max(m, mt)
alpha = exp(m  - m_new)
beta  = exp(mt - m_new)
l_new = alpha * l + beta * lt
o_new = alpha * o + beta * ot
```

最后 `O = o / l`。证明只需把旧、新两组指数都换成以 `m_new` 为基准：

```text
exp(x - m_new) = exp(x - m) * exp(m - m_new)
```

代码中的 half WMMA 路径按 16 个 key 为一个 tile 执行该合并。通用 warp 路径是退化到
单 key tile 的同一公式。

### 3.2 全 mask tile

若一个 tile 对当前 query 完全不可见，就不能机械计算 `-inf - -inf`，否则得到 NaN。
实现通过 `key_end` 和 `row_key_count` 直接跳过该 tile。Host 已单独处理 `Tk == 0`。

## 4. CUDA 映射的逐步构造

### 4.1 Warp-online 通用路径

最容易保证通用性的映射是“一条 warp 负责一行输出”：

```text
CTA: 8 warps = 256 threads
warp: 一个 (batch, query_position, query_head)
lane: D 维中的若干元素
```

每个 lane 把 Q 的元素和输出累加器保存在寄存器中。每遇到一个 key：

1. lane 计算自己的局部 dot。
2. 完整 warp 用 shuffle 规约为 score。
3. lane 0 更新 `(m,l)` 并广播缩放因子和概率。
4. 各 lane 更新自己的输出分量。

`ValuesPerLane` 随 D 为 `1/2/4/8`，因此覆盖 `D <= 256`。CTA 固定为完整 warp，shuffle
mask 为 `0xffffffff`，不存在 partial-warp 中读取未参与 lane 的问题。

优点是无需大块 shared memory、尾部自然处理、online softmax 直观。缺点是每个 query
row 独立读取 K/V，同一 KV head 下的 GQA 复用没有实现。

### 4.2 Shared-KV 路径

第二层路径按同一个 `(batch, kv_head)` 聚合 8 个 query row：

```text
CTA = 8 warps
每 warp = 一个 query row
K/V tile = 32 keys
shared Q = 8 * D 个 float
shared K/V = 2 * 32 * D 个 float
```

这样，同一 CTA 的 8 行共享 K/V tile。它对 float 和短序列 half 很实用，dispatch 条件是：

```text
D <= 64
Tk >= 16
每个 KV head 至少有 8 个 query row
half 还要求 Tk <= 256
```

当前实现为满足严格 float 参考顺序，采用三遍：第一遍 max，第二遍 sum，第三遍 PV。
三遍都会重算 QK，但 K/V 不形成二次方 HBM 中间矩阵，且同一 CTA 内复用 K/V tile。

这是一个重要工程判断：论文中的“online”是算法首选，但作业 tester 的 float 容差非常
紧。数学等价不意味着浮点逐位等价，改变 reduction 和 softmax 合并顺序会改变末位。

### 4.3 Half WMMA 路径

`D=32/64`、`Tk>=64` 且每 KV head 至少 16 个 query row 时启用 WMMA：

```text
CTA                    4 warps / 128 threads
query tile             64 rows
key tile               16 keys
QK                      16x16x16 WMMA
WMMA accumulator       FP32
softmax state          FP32
PV                     FP32 SIMT accumulation
output                  half
```

每个 warp 计算 16 个 query 与 16 个 key 的 score tile。Q 以 row-major 装入 fragment；
K 在 shared memory 中按 `[key, dim]` 存放，但以 col-major fragment 解释，从而得到
`Q * K^T`。`D=32` 做两次 K 维 MMA，`D=64` 做四次。

WMMA fragment 是 warp collective：同一 warp 的 32 个 lane 必须一起执行 load、MMA 和
store，不能只让部分 lane 进入。score 写到 shared memory 后，两个 lane 合作处理一个
query row，并按奇偶 D 分量做 FP32 PV 累加。

为什么没有把概率转成 half 后再用第二次 WMMA 做 PV？因为 softmax 概率的 half 量化会
在极端 logits 和长序列上扩大误差。当前方案把 Tensor Core 用在最重的 QK，softmax 和
PV 保持 FP32，取得了更稳妥的精度/速度平衡。

### 4.4 Float 的特殊路径

float 不启用 TF32，因为任务要求严格 FP32 结果。`D=32` 使用编译期固定数组并完全展开
内层循环；其他尺寸使用 shared-KV 或 `D<=256` 的通用三遍路径。

固定 D32 路径的 Q 和 O 共 64 个 float，编译结果为 94 个寄存器、零 spill。试验过的
warp 并行 QK 改变了累加顺序，在官方紧容差 case 上失败，因此没有保留。

## 5. GQA 为什么影响 CTA 设计

若 `Hq/Hkv = g`，连续的 `g` 个 query head 读取同一个 K/V head。实现把一个 KV head
下的行展平为：

```text
query_in_kv = query_position * g + head_in_group
query_pos   = query_in_kv / g
query_head  = kv_head * g + query_in_kv % g
```

一个 CTA 取连续的 `query_in_kv`，从而同时覆盖位置和 GQA head，并复用 shared K/V。
这比按全局 query row 随意分组更能稳定命中同一个 KV head。

## 6. Causal 与非方形序列

对左上角 causal，第 `i` 行只读 `[0, min(Tk, i+1))`。因此：

- `Tq < Tk`：靠后的 key 对所有 query 都可能不可见。
- `Tq > Tk`：当 `i >= Tk-1` 后可见全部 key。
- CTA 内各 query 的 `key_end` 不同，加载上界取 CTA 中最大值，各行计算时再裁剪。

这种语义已用 `16x32`、`32x16`、GQA 和尾 tile 测试覆盖。不要根据方形 self-attention
的直觉写成右下角 causal。

## 7. Shared memory、合并访存和 bank conflict

K/V 的源布局是 `[Tk,Hkv,D]`。固定 kv head、相邻 key 的数据之间隔着 `Hkv*D`，因此
“相邻线程各加载相邻 key 的同一 dim”通常不连续。实现改为把 tile 展平成
`key_in_tile * D + dim`，线程沿这个一维空间分工；在每个 key 内，连续线程读连续 D。

shared 数组保持简单的线性布局。对当前 `D=32/64`，WMMA load 访问规则比手工 XOR
swizzle 更重要。NCU 仍报告部分 excessive shared wavefront，说明后续可以研究 swizzle；
第一版没有为一个尚未证实的收益引入难以验证的地址变换。

## 8. 数值精度为什么是设计约束

浮点加法不满足结合律：

```text
(a + b) + c != a + (b + c)
```

因此以下变化都会影响结果：

- 一个线程串行 dot 改成 warp tree reduction。
- 三遍稳定 softmax 改成 tile online 合并。
- `sum += x` 改成不同位置的 FMA。
- 概率或 V 提前量化为 half。
- float QK 改用 TF32 Tensor Core。

本实现的原则是：half 输入、dot、softmax、PV 都在 FP32 累加；只有 Q/K/V 存储和最后
输出是 half。float 路径坚持 FP32 并保留能通过 tester 的累加顺序。

扩展测试使用独立 CPU double reference，覆盖零值、常量、随机正负、大动态范围、极端
logits、尾块和 `D=7/31/33/128/256`。判定不是只看相对误差：参考值接近零时，相对误差
会失真，所以使用“绝对误差或相对误差满足其一”。

## 9. 三种“异步”不能混为一谈

### 9.1 Kernel 内 `cp.async`

`cp.async`/`cuda::memcpy_async` 把 global-to-shared copy 与当前 tile 的计算重叠，适用于：

1. 下一 tile 的地址与大小可提前确定。
2. global/shared 对齐满足 4/8/16 字节，最好有更强对齐保证。
3. 有足够计算覆盖 copy 延迟。
4. 双缓冲增加的 shared memory 和寄存器不显著降低 occupancy。
5. NCU 表明 Long Scoreboard 与 DRAM 等待确实是主瓶颈。

本项目 SASS 中 `LDGSTS=0`，即正式路径没有使用 `cp.async`。这不是遗漏，而是数据驱动
的取舍：最大 half WMMA kernel 的 DRAM 吞吐只有 3.48%，L2 hit 较高，kernel 只有
0.410 ms；完整函数主要被 host 传输限制。加双缓冲无法带来端到端 5% 的稳定收益。

### 9.2 Host pinned memory 与多 stream

pinned host buffer 允许真正的异步 DMA；Q/K/V 可以放到两个 H2D stream，再用 event
让计算 stream 等待。但公开接口传入 `std::vector`，它通常是 pageable memory。若每次
先把 vector 复制到临时 pinned buffer，CPU staging 会吃掉绝大部分收益。

### 9.3 Managed Memory、advise 与 prefetch

`cudaMemAdvise` 告诉驱动偏好位置，`cudaMemPrefetchAsync` 主动迁移页面，但它们不消除
迁移。当前每次函数调用都由 CPU 初始化输入、GPU 读取、CPU 再读取输出，这正好产生
CPU/GPU 往返；大页迁移和同步比显式 memcpy 更慢。

### 9.4 CUDA Graph

Graph 擅长压缩固定序列的重复 launch/API 开销。这里每次调用仍要处理新的 host vector、
分配设备内存并返回同步结果。只有调用者能持久持有 pinned buffer、device workspace 和
已实例化 Graph 时才有明显意义；在当前无全局状态的函数接口里不适合合入。

## 10. 从 FA1 到 FA4，哪些思想能迁移到 5090

### FA1

可直接迁移的是 IO-aware 视角、K/V 分块和 online softmax：不物化二次方的 S/P 矩阵。

### FA2

可直接迁移的是 query 方向并行、同一 head 内更多 CTA、warp 间减少 shared 通信，以及
让工作划分匹配尺寸。本实现的 query tile 和 GQA KV-head 聚合属于这一层思想。

### FA3

FA3 在 H100 上利用异步 Tensor Core、TMA、warp specialization，将 GEMM 与 softmax
流水交叠。5090 可以学习“生产者/消费者与阶段重叠”的方法，但不能照搬 Hopper 专用
调度；而且本项目 NCU 没有证明数据搬运是首要问题。

### FA4

FA4 面向 B200/GB200 的非对称硬件扩展，重点包括完全异步 MMA、更大 tile、降低非
matmul 开销、Tensor Memory 和 2-CTA MMA。RTX 5090 虽同属 Blackwell、计算能力为
12.0，但 FA4 所依赖的 `tcgen05`/Tensor Memory 编程模型面向数据中心 Blackwell 的
`sm_100/101`，不能把“Blackwell”这个名字等同为指令集完全相同。

5090 第一版采用成熟 WMMA/HMMA，SASS 已确认 `HMMA.16816.F32`。TMA、`tcgen05`、
跨 CTA Tensor Memory 和 split-K 均未进入正式路径。

## 11. 如何读 NCU

先问“这个 kernel 是否填满 GPU”：

- `waves/SM` 很小：尺寸太小，优化单 CTA 往往不会改善端到端。
- achieved occupancy 远低于 theoretical：可能有尾波、依赖链或活跃 CTA 不足。
- registers/shared memory 限制 theoretical occupancy：先评估 tile 是否过大。

再问“等待什么”：

- `Long Scoreboard`：常见于 global/local/L1TEX 长延迟依赖，但需同时看 DRAM/L2/L1。
- `Short Scoreboard`：常见于 shared memory 或较短数据依赖。
- `LG Throttle`：global/local 指令发射管线压力，不等同于 HBM 带宽打满。
- `Barrier`：同步等待，需检查 warp 工作量是否不均。

最后看内存层次：

- DRAM 高且 Long Scoreboard 高：才是 `cp.async`/更强复用的典型候选。
- L1 接近峰值而 DRAM 很低：问题更可能是访问形状、请求数量或 L1 依赖。
- excessive sectors/wavefronts 高：检查 global 合并和 shared bank conflict。
- local load/store 与 ptxas spill：检查寄存器数组是否落到 local memory。

本项目的最大 float 路径就是“L1 99.43%、DRAM 0.16%、Long Scoreboard 62.26%、
LG Throttle 36.51%”，因此它不是 HBM 带宽问题。最大 half 则是 WMMA 吞吐与片上数据流
问题，DRAM 同样不高。

## 12. 如何读 NSYS

NSYS 回答的是时间线问题，而非单 kernel 微架构问题：

1. `cudaMalloc/cudaFree` 是否频繁出现。
2. H2D、kernel、D2H 是否串行。
3. 多 stream 是否真的重叠，而不只是 API 名字带 Async。
4. Graph 是否减少 CPU launch 空泡。
5. Unified Memory 是否发生大量小块迁移。

本接口是同步函数，正确的端到端模型是：

```text
host vector -> H2D(Q,K,V) -> kernel -> D2H(O) -> return
```

调用返回前输出必须可供 CPU 使用，所以单次调用不能把 D2H 隐藏到调用之后。多流最多
重叠三份输入的 H2D；若 pageable-to-pinned staging 仍在关键路径，收益会明显缩水。

## 13. 代码阅读路线

建议按以下顺序阅读 `src/kernels.cu`：

1. `flashAttention`：尺寸验证、连续分配、H2D/kernel/D2H 生命周期。
2. `launchFlashAttention(float)` 与 `launchFlashAttention(half)`：理解 dispatch。
3. `flashAttentionWarpOnlineKernel`：最小 online softmax。
4. `flashAttentionSharedKvKernel`：GQA 复用与 shared K/V tile。
5. `flashAttentionHalfWmmaKernel`：fragment 布局和 tile 合并。
6. `flashAttentionFloatD32Kernel`：严格 float 顺序与寄存器权衡。

读每个 kernel 时，在纸上写出一个线程、一个 warp、一个 CTA 分别拥有哪部分 Q/K/V/O；
再标记每次 `__syncthreads` 前后谁写、谁读 shared memory。这个方法比只跟着循环变量读
更不容易迷失。

## 14. 可复现实验

默认正确性：

```bash
make clean
make build EXTRA_LIBS=/tmp/cuda13_abi_shim.o
CUDA_VISIBLE_DEVICES=0 ./test_kernels --verbose
```

5090 优化构建：

```bash
make clean
make build \
  CFLAGS='-std=c++17 -O3 -lineinfo -arch=sm_120 -Xptxas=-v' \
  EXTRA_LIBS=/tmp/cuda13_abi_shim.o
```

Sanitizer 的四类检查分别解决不同问题：

```bash
compute-sanitizer --tool memcheck  ./flash_correctness
compute-sanitizer --tool initcheck ./flash_correctness
compute-sanitizer --tool racecheck ./flash_correctness
compute-sanitizer --tool synccheck ./flash_correctness
```

不要把 NCU 与 Event 时间直接混为一张“速度表”：NCU replay、基础时钟锁定和计数器采集
会扰动时间。NCU 用于解释瓶颈，Event 多组中位数用于比较路径，NSYS 用于解释端到端。

## 15. 学习资料

本地 PDF：

- `并行编程与CUDA编程入门.pdf`
- `2_性能模型与逐元素优化.pdf`
- `2026夏季_内存模型与规约优化.pdf`
- `向量规约.pdf`
- `4_分块与不规则访存.pdf`
- `2026夏季_异步并行-底层控制与系统优化.pdf`

论文与官方文档：

- [FlashAttention，NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)
- [FlashAttention-2，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/98ed250b203d1ac6b24bbcf263e3d4a7-Abstract-Conference.html)
- [FlashAttention-3，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html)
- [FlashAttention-4，arXiv 2026](https://arxiv.org/abs/2603.05451)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [PyTorch scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)

## 16. 最后形成的判断习惯

1. 先固定语义，再谈 tile。
2. 先写稳定、通用的路径，再用 dispatch 加专用路径。
3. Online softmax 是代数工具，不是自动保证逐位相同的魔法。
4. Tensor Core、`cp.async`、多 stream、UM、Graph 分属不同层次，不能互相替代。
5. NCU 先定位瓶颈，Event 再判断收益，NSYS 最后解释端到端。
6. 一个优化若需要改变接口和生命周期，就应把成本写入结论，而不是只展示理想 microbenchmark。
