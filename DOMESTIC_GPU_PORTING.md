# 摩尔线程与沐曦 GPU 双算子移植说明

本文记录 `src/kernels.mu` 和 `src/kernels.maca` 中 RMSNorm 与 Flash
Attention 前向实现的设计依据、数值推导、平台差异和后续真机验收方法。

当前版本已经在 MTT S5000 上完成 Moore Threads 第一阶段真机验证与按尺寸调优；
MetaX 仍是等待凭据恢复的正确性优先实现。文中严格区分 S5000 真机、RTX 5090
CUDA 映射和尚未验证的 MetaX，不用一个平台的数据代替另一个平台的结论。

## 1. 当前状态

| 项目 | Moore Threads | MetaX |
|---|---|---|
| 源文件 | `src/kernels.mu` | `src/kernels.maca` |
| 编译器 | `mcc`, C++11 | `mxcc`, C++17 |
| Runtime | MUSA Runtime | MXMACA Runtime |
| RMSNorm | 已实现 | 已实现 |
| Flash Attention float | 已实现 | 已实现 |
| Flash Attention half | 已实现 | 已实现 |
| 厂商编译器验证 | MUSA 4.3.5 已通过 | 尚无可用登录凭据 |
| 厂商 GPU 正确性 | S5000 官方与扩展测试通过 | 尚无真机 |
| 厂商 GPU 性能 | Event、完整函数与资源报告已采集 | 尚无真机 |

NVIDIA 代码位于另一套工作环境中。本次没有修改 `src/kernels.cu`。S5000 快路径是
依据该设备实测 `warpSize=32` 独立实现，并在运行时检查 warp 宽度；其他 Moore 设备
和 MetaX 仍保留不依赖固定 warp 宽度的 shared-memory/单线程 query 路径。

## 2. 为什么不能直接复制 NVIDIA 内核

三个平台都提供类似的 Grid/Block/Thread 编程模型，但 warp 或 wave 大小并不相同：

- NVIDIA CUDA 的 warp 固定为 32 个线程。
- Moore Threads 的设备相关：S5000 系列为 32，M1000/S4000 系列为 128。
- MetaX C500 的 wavefront 为 64 个线程。

因此，下面这些写法不能在未知目标卡时直接复用：

- `lane = threadIdx.x & 31`；
- 固定使用 `0xffffffff` 作为完整 warp mask；
- 只执行 16、8、4、2、1 五步的 shuffle reduction；
- 假设一个 warp 正好计算一个 32 维 query；
- 假设 CUDA WMMA fragment、PTX 或 `cp.async` 在另一平台有相同语义。

所以不能把 32-lane 假设写进通用路径。本版在 S5000 上增加了使用 MUSA shuffle 的
32-lane 快路径，但每次 dispatch 都先用 `musaGetDeviceProperties` 核实实际 `warpSize`；
查询结果按当前 device 在线程局部缓存。非 32-lane Moore 设备继续使用原 shared-memory
RMSNorm 和一线程一 query Attention。代码不使用 ballot、WMMA 或平台汇编。

参考资料：

- MUSA Warp 函数：
  <https://docs.mthreads.com/en/musa-sdk/musa-sdk-doc-online/programming_guide/musa_cpp_syntax/warp_functions/>
- MUSA Runtime API：
  <https://docs.mthreads.com/en/musa-sdk/musa-sdk-doc-online/libraries/core_api/runtime_api_reference/>
- MXMACA C++ 编程指南：
  <https://developer.metax-tech.com/doc/257>
- MXMACA PyTorch 移植指南中的 warpSize 说明：
  <https://developer.metax-tech.com/api/client/document/file/229/preview/?file_type=pdf>

## 3. Runtime API 对照

公开模板函数保持不变，平台差异只留在各自源文件中。

| 操作 | Moore Threads | MetaX |
|---|---|---|
| half 头文件 | `<musa_fp16.h>` | `<common/maca_fp16.h>` |
| 分配显存 | `musaMalloc` | `mcMalloc` |
| 主机到设备 | `musaMemcpyHostToDevice` | `mcMemcpyHostToDevice` |
| 设备到主机 | `musaMemcpyDeviceToHost` | `mcMemcpyDeviceToHost` |
| 查询 launch 错误 | `musaGetLastError` | `mcGetLastError` |
| 释放显存 | `musaFree` | `mcFree` |
| 错误检查 | `RUNTIME_CHECK` | `RUNTIME_CHECK` |

两种算子都使用一次连续显存分配。各张量的偏移按 256 字节对齐，随后执行同步
H2D、kernel、同步 D2H 和释放。这样与当前 `std::vector` 接口的所有权和生命周期一致，
也减少了多次分配带来的额外错误路径。

## 4. RMSNorm

### 4.1 数学定义

对每一行独立计算：

```text
sum_square = sum_j x[j]^2
inv_rms = rsqrt(sum_square / hidden_dim + eps)
y[j] = x[j] * inv_rms * weight[j]
```

float 与 half 均用 FP32 计算平方和与缩放，half 只在读入和最终写回时转换。

### 4.2 通用线程映射

- 一个 block 对应一行。
- 每个线程以 `blockDim.x` 为步长读取本行元素并计算局部平方和。
- 局部平方和写入 shared memory。
- 使用二叉树规约，每一级后执行 `__syncthreads()`。
- 线程 0 计算 `inv_rms`，再由全 block 共同写输出。

这里的规约只要求 block 大小是 2 的幂，不依赖硬件 warp 宽度。

Moore Threads 的通用 fallback：

```text
hidden_dim <= 128 -> 128 threads
otherwise         -> 256 threads
```

MetaX 的 dispatch：

```text
hidden_dim <= 64  -> 64 threads
hidden_dim <= 128 -> 128 threads
otherwise         -> 256 threads
```

### 4.3 S5000 warp 规约

S5000 的 `warpSize` 实测为 32。快路径先在线程内累加平方和，再执行 32-lane shuffle；
每个 warp 的 lane 0 把结果写入很小的 shared 数组，首 warp 完成第二级规约。与通用路径
相比，它减少了 shared-memory 往返和多次 block 同步。

线程数取能覆盖当前维度的较小整 warp 配置：

```text
D <= 32  -> 32 threads
D <= 64  -> 64 threads
D <= 128 -> 128 threads
otherwise -> 256 threads
```

完整同步函数的固定分配与拷贝成本会掩盖小 kernel 的收益，最终 dispatch 结合 Event 和
官方完整函数数据：float 在 `D<=512` 或 `rows>=8` 时启用；half 在 `D<=256`、
`D>=768` 或 `rows>=128` 时启用。其余尺寸和非 32-lane 设备走通用 fallback。

### 4.4 为什么没有合入 float4/half8

向量访存需要同时确认：

1. 编译器是否生成真正的宽 load/store；
2. 行首和每一行步长是否满足对齐；
3. half2/向量类型的 ABI 是否与预期一致；
4. 寄存器缓存是否造成 spill；
5. 目标卡上收益是否超过标量路径。

本轮已经有 `mcc` 与 S5000 数据，但标量 warp 规约在代表尺寸已获得稳定收益，而当前
容器没有可用的指令级 profiler 来证明向量类型生成了预期宽 load/store。为避免引入
对齐尾部、half ABI 和寄存器缓存风险，正式文件没有合入 `float4/half8`；它们仍是获得
匹配版本 Moore Perf 容器后应继续测试的实验项。

## 5. Flash Attention 契约

输入布局：

```text
Q: [B, Tq, Hq, D]
K: [B, Tk, Hkv, D]
V: [B, Tk, Hkv, D]
O: [B, Tq, Hq, D]
```

每个输出 query row 的线性编号是：

```text
query_row = ((batch * Tq + query_pos) * Hq + query_head)
```

GQA 映射：

```text
queries_per_kv = Hq / Hkv
kv_head = query_head / queries_per_kv
```

score 定义：

```text
score(q, k) = dot(Q[q], K[k]) / sqrt(D)
```

causal 使用左上三角语义：

```text
key_pos <= query_pos
key_end = min(Tk, query_pos + 1)
```

它适用于 `Tq != Tk` 的非方形输入，不采用右下对齐的 causal mask。

## 6. Float Attention：稳定三遍算法

每个 GPU 线程负责一个完整 query row。这样不需要 warp 内通信，且相同 query 的
key 顺序固定，有利于复现严格 float 累加顺序。

第一遍只求最大 score：

```text
m = max_k score_k
```

第二遍重新计算 score 并求指数和：

```text
l = sum_k exp(score_k - m)
```

第三遍再次计算 score 并累加输出：

```text
O[d] = sum_k exp(score_k - m) / l * V[k, d]
```

代价是 QK dot 被计算三次，但不会分配 `Tq x Tk` score 矩阵，也避免了跨线程规约
改变 float 求和顺序。它也是非 32-lane 设备和较小工作量的正确性 fallback。

S5000 的 `D>=64` 且 `Tk*D>=4096` 路径改为一 warp 处理一个 query。lane 0 仍按原来
完全相同的 key、dim 顺序完成三遍 QK 与 softmax，因此不会改变严格 float 结果；第三遍
的 probability 用 shuffle 广播，其余 lane 并行累加不同输出维度。官方最大 `D64`
用例最大误差为 0，同时消除了原来每线程的 `D64` 私有输出数组。

## 7. Half Attention：Online Softmax

half 输入仍全部转为 FP32 后参与 dot、指数、归一化和 PV 累加。每处理一个 key，维护：

```text
m_new = max(m, score)
alpha = exp(m - m_new)
p = exp(score - m_new)
l_new = alpha * l + p
O_new[d] = alpha * O[d] + p * V[k, d]
```

初始 `l=0`，因此第一个 key 的 `alpha` 显式设为 0，避免用有限哨兵值参与无意义的
指数计算。处理完所有 key 后输出 `O/l`。

half 的容差通常比 float 宽，online softmax 可以把 QK 从三遍降为一遍，同时保持稳定性。
最终结果只转换一次 half，避免中间概率和输出反复量化。

S5000 对 `D>=64` 且 `Tk*D>=4096` 使用一 warp 一 query：每个 lane 缓存最多 8 个 Q
和输出 FP32 值，warp shuffle 完成 QK 规约，所有 lane 同步更新 online softmax 状态。
这条路径将 key 方向的串行工作分摊给 32 个 lane，并去掉旧 `D64` 路径的 512 字节线程栈。
`D32` 大用例实测会回退，因此明确保留原 kernel。

## 8. Head Dimension Dispatch

官方常见维度生成独立模板实例：

```text
D = 1, 2, 4, 8, 16, 32, 64
```

通用 fallback 中，这些实例的 `q_cache` 和 `output_cache` 大小是编译期常量，编译器
可以更好地展开和分配寄存器。其他维度走最大 256 的通用实例。

S5000 的 warp 路径每 lane 最多持有 8 项缓存，`mcc -resource-usage` 确认固定 `D64`
和通用 warp 实例均为 0 字节 local memory、0 spill。旧 `D64` fallback 有 512 字节
线程栈，`D=0` 通用 fallback 有 2048 字节线程栈，因此 `D=128/256` 仍是下一轮重点。
功能上明确拒绝 `D>256`。

## 9. 参数与边界处理

Host 端在分配显存前检查：

- 所有 attention 维度非负；
- `kv_heads` 非零且整除 `query_heads`；
- `head_dim <= 256`；
- 所有元素数量、字节数、对齐偏移和总分配大小无 `size_t` 溢出；
- 输入 vector 至少包含 shape 所需元素；
- RMSNorm 行数与 Attention 一维 grid 的 block 数可以用 `unsigned int` 表示。

输出 vector 尺寸不一致时会调整到正确大小。RMSNorm 任一维为零以及空 query 都返回
空输出；`Tk=0` 时直接把输出内存清零，不启动会产生 `1/0` 的 softmax kernel。

## 10. 为什么没有加入系统级异步方案

课程中异步加载、流水、多 stream、Managed Memory、prefetch 和 Graph 都是重要工具，
但它们解决的是不同层级的问题：

- kernel 内异步拷贝用于隐藏 global-to-shared 延迟；
- pinned memory 与多 stream 用于重叠 PCIe 传输和计算；
- Managed Memory 的 advise/prefetch 用于控制页面放置和迁移；
- Graph 用于固定工作流的重复 launch。

当前公开接口每次接收普通 `std::vector`，并在函数结束前同步返回结果。它没有传入
stream、device tensor、pinned buffer 或持久 workspace，也不能保证地址和 shape 重复。
在这种契约下加入多流或 Graph 会引入 staging、缓存和生命周期状态，未必降低完整函数
时间。

此外，当前 Attention 快路径直接从 global memory 消费 K/V，没有 global-to-shared tile，
直接加入异步 shared copy 没有对应的消费流水。本轮 `mcc` 资源报告已证明旧 kernel 的
私有数组和 query 内串行工作是明确问题，而优化 kernel 为 0 local memory、0 spill；
Event 也显示仅改变线程映射就有 3 到 13 倍收益。当前容器又无法读取 stall/HBM 计数器，
所以没有证据支持把正式代码重构为更复杂的 K/V 分块双缓冲。

同理，pinned、多 stream、内存池和 Graph 需要持久 host/device 缓冲区或可复用工作流。
公开接口没有这些所有权条件。本轮没有用未经验证的全局缓存改变接口语义，也没有把
Managed Memory、advise 或 prefetch 当作普通 `std::vector` 同步调用的默认优化。

## 11. CUDA 映射验证的含义

由于本地没有 `mcc`、`mxcc` 或国产 GPU，可以在 RTX 5090 的 `/tmp` 中建立临时镜像：

1. 把 half 头文件映射为 `<cuda_fp16.h>`；
2. 把 `musa*` 或 `mc*` Runtime 名称映射为 `cuda*`；
3. 保持 kernel、索引、模板和 host 生命周期不变；
4. 链接 NVIDIA tester，运行官方和扩展正确性；
5. 对映射副本运行 Compute Sanitizer。

这能发现大部分算法、索引、同步和显存生命周期错误，但不能证明：

- 原文件能被 `mcc` 或 `mxcc` 编译；
- half ABI 和数学 intrinsic 在目标版本完全一致；
- 国产 GPU 上没有编译器 bug；
- 当前 block size、寄存器数量或访存方式性能良好。

因此映射结果必须标为“CUDA 等价路径验证”，不能写成 Moore/MetaX 实测。

### 11.1 本轮实际验证记录

验证日期为 2026-08-11，设备是 RTX 5090，工具链是 CUDA 13.0。临时映射、驱动和
原始日志位于远端 `/tmp/domestic_cuda_smoke.DjD3f5/`，没有写入 NVIDIA 仓库。
预编译 tester 所需的 CUDA 13 ABI shim 也只存在于该临时目录。

| 验证项 | Moore 映射 | MetaX 映射 |
|---|---:|---:|
| 默认优化级别 RMSNorm float/half | 26/26 | 26/26 |
| 默认优化级别 Attention float/half | 28/28 | 28/28 |
| `-O3 -lineinfo -arch=sm_120` RMSNorm | 26/26 | 26/26 |
| `-O3 -lineinfo -arch=sm_120` Attention | 28/28 | 28/28 |
| 扩展 RMSNorm | float/half 各 7 组 | float/half 各 7 组 |
| 扩展 Attention | float/half 各 5 组 | float/half 各 5 组 |
| 空 KV、空 RMS、超限维度/网格 | 通过 | 通过 |
| memcheck/initcheck/synccheck | 0 error | 0 error |
| racecheck | 0 hazard | 0 hazard |

扩展 Attention 覆盖 `D=7/31/33/128/256`、2 倍和 3 倍 GQA、causal、非 causal、
`Tq<Tk`、`Tq>Tk`、大动态范围输入和空 KV。扩展 RMSNorm 覆盖
`D=1/31/33/127/129/257/768`，并检查输出自动调整。CPU 参考使用双精度计算，GPU
输入与输出仍按被测类型量化。

RTX 5090 的 `ptxas` 结果只能帮助理解算法的资源形态：`D=32/64` 固定实例没有
stack frame 或 spill；float 分别使用 95/168 个寄存器，half 分别使用 96/167 个
寄存器。`D=0` 通用模板因为保留两组 256 项 FP32 私有数组，产生 2048 字节 stack
frame，虽然报告为 0 个显式 spill store/load，仍提示 `D=128/256` 是国产真机首先要
profile 的路径。这里不记录 CUDA 时间，因为它不能代表国产设备性能。

临时映射副本建立后，核心复现命令如下；Moore 映射保留其 C++11 约束，MetaX 映射
保留 C++17 约束：

```bash
nvcc -std=c++11 -O0 -DPLATFORM_NVIDIA -c src/kernels_mu_cuda.cu
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA -c src/kernels_maca_cuda.cu

nvcc -std=c++11 -O3 -lineinfo -arch=sm_120 -Xptxas=-v \
  -DPLATFORM_NVIDIA -c src/kernels_mu_cuda.cu
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 -Xptxas=-v \
  -DPLATFORM_NVIDIA -c src/kernels_maca_cuda.cu

compute-sanitizer --tool memcheck  ./extended_mu
compute-sanitizer --tool initcheck ./extended_mu
compute-sanitizer --tool racecheck ./extended_mu
compute-sanitizer --tool synccheck ./extended_mu
# 对 extended_maca 重复相同四项。
```

## 12. Moore Threads S5000 真机结果

### 12.1 环境与升级边界

验证日期为 2026-08-11，干净远端仓库基于 `master` 的 `4c7bf8f`。环境如下：

| 项目 | 真机值 |
|---|---|
| GPU | MTT S5000，80 GiB |
| SM 数 | 60 |
| Compute Capability | 3.1，对应 `mp_31` |
| `warpSize` | 32 |
| 每 block 最大 shared memory | 196608 bytes |
| 驱动 | `3.3.5-server` |
| `mthreads-gmi` | 2.3.2 |
| MUSA Runtime / SDK / `mcc` | 4.3.5 |
| `mcc` 基础 | Clang 14.0.0 |

环境变量固定为：

```bash
export PATH=/usr/local/musa/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH
```

登录环境是 GPU 容器，PID 1 为 `entrypoint.sh`，没有 systemd，也没有
`CAP_SYS_MODULE`、`CAP_SYS_ADMIN`、`PERFMON` 或 `SYS_PTRACE`。容器内 root 能替换
用户态文件，不能替换宿主机内核驱动。当前可工作的 4.3.5 安装因此没有被覆盖。

本轮查到的新版 `mt-compute-sanitizer` 1.1 要求 Driver/SDK 5.2；只把 5.2 SDK 装进
3.3.5 驱动容器会得到版本不匹配环境，不能用于正式结论。正确升级方法是平台提供
Driver、Runtime、SDK、Moore Perf 与 Sanitizer 成套 5.2 镜像，或由管理员先升级宿主机。
官方 Sanitizer 要求见：
<https://docs.mthreads.com/en/mooreperf/mooreperf-doc-online/musa_compute_sanitizer/install/>。

当前镜像没有 `mcu`、`msys`、`mt-compute-sanitizer`。因此本节没有声称 MUSA
Sanitizer 通过，也没有伪造 HBM、stall 或 occupancy 计数器；回退证据是 MUSA Event、
官方完整函数计时、`mcc -resource-usage` 和 `mthreads-gmi`。Moore Perf 工具说明：
<https://docs.mthreads.com/en/musa-sdk/musa-sdk-doc-online/programming_guide/performance_tuning/perf_tools/>。

### 12.2 编译与正确性

原始正确性基线和最终优化版都使用公开 Makefile 编译：

```bash
make PLATFORM=moore clean
make PLATFORM=moore build
./test_kernels --verbose
```

`mcc 4.3.5` 不支持真正的 `-O0`，传入后会警告并提升到至少 `-O2`；所以不能把这次
调试构建写成 O0 结果。默认 Makefile 的 `-O3` 是正式验证配置。

| 验证项 | 原始基线 | 最终版 |
|---|---:|---:|
| 官方 RMSNorm float/half | 26/26 | 26/26 |
| 官方 Attention float/half | 28/28 | 28/28 |
| 扩展 RMSNorm | float/half 各 7 组通过 | float/half 各 7 组通过 |
| 扩展 Attention | float/half 各 5 组通过 | float/half 各 5 组通过 |
| 空 KV、空输入、非法 GQA/维度/网格 | 通过 | 通过 |

扩展 RMSNorm 包含非整 warp 与非向量尾部；扩展 Attention 包含
`D=7/31/33/128/256`、2/3 倍 GQA、causal、非方形序列、大动态范围和空 KV。
参考结果由 CPU double 独立计算。S5000 上的正式源码而非 CUDA 映射副本完成了这些测试。

### 12.3 RMSNorm 性能

kernel-only 使用 MUSA Event，预热 20 次、计时 200 次。下表是 shared 二叉树基线与
32-lane warp 规约的代表结果；速度提升是 `baseline / warp`：

| rows, D | float | half |
|---|---:|---:|
| 1, 31 | 1.207x | 1.189x |
| 64, 256 | 1.618x | 1.605x |
| 128, 1024 | 1.655x | 1.654x |
| 512, 256 | 2.689x | 2.685x |
| 512, 4096 | 1.484x | 1.474x |

Event 矩阵中的所有测试尺寸都提升，但完整函数还包含分配、两次 H2D、一次 D2H 和释放。
10 组基线/最终版交替先后顺序的中位数显示：官方 `rows=128,D=1024` half 从
`0.279208 ms` 降到 `0.241053 ms`（1.158x），`rows=32,D=2048` float 从
`0.278506 ms` 降到 `0.240656 ms`（1.157x）。小尺寸的 kernel 收益大多被约
`0.13-0.21 ms` 的固定成本淹没。

`rows=64,D=512` half 的 kernel-only 虽快，但早期完整函数采样慢约 6%，所以最终
dispatch 已改回 shared fallback。最终两个不同代码体积的测试二进制即使执行相同 kernel，
该项仍有一次 10 组中位数差异 5.1%；这不是优化 kernel 的回退，文档仍保留该噪声而不
把它写成性能提升。

### 12.4 Attention 性能

kernel-only Event 结果：

| 类型与 shape `(B,Tq,Tk,Hq,Hkv,D,causal)` | 基线 ms | warp ms | 提升 |
|---|---:|---:|---:|
| float `(1,64,64,16,4,64,0)` | 1.920305 | 0.302043 | 6.358x |
| float `(2,256,256,32,32,64,0)` | 23.920208 | 7.775418 | 3.076x |
| float `(1,256,1024,32,8,64,0)` | 55.069691 | 17.809114 | 3.092x |
| half `(2,256,256,32,32,64,0)` | 15.860320 | 1.274339 | 12.446x |
| half `(1,256,1024,32,8,64,0)` | 38.571468 | 2.992286 | 12.890x |
| half `(1,512,1024,64,16,64,1)` | 27.664106 | 2.325215 | 11.897x |

官方最大 `D64` 用例的完整函数从 float `26.450363 ms`、half `17.659838 ms`
降到 float `10.220700 ms`、half `2.598487 ms`，分别为 2.588x 和 6.796x。
float 最大误差为 0；half 最大误差 `0.0078125`，在官方 `0.0469141` 容差内。

官方最大 `D32` 用例不满足 `D>=64`，继续走原路径：float `21.458184 -> 20.751852 ms`，
half `9.226470 -> 9.285083 ms`。实验中强制 half warp 路径会从 `1.657144 ms`
恶化到 `8.231820 ms`，所以按维度回退是必要的，不是遗漏优化。

### 12.5 编译资源证据

使用 `mcc -resource-usage` 编译正式文件，并只读取 S5000 的 `mp_31` 条目：

| kernel | Temp registers | Shared memory | Local/stack | Spill load/store |
|---|---:|---:|---:|---:|
| RMS float 256-thread shared | 15 | 1040 B | 0 B | 0/0 B |
| RMS float 256-thread warp | 15 | 48 B | 0 B | 0/0 B |
| float `D64` 三遍 fallback | 54 | 12 B | 512 B | 0/0 B |
| float `D64` warp exact | 57 | 12 B | 0 B | 0/0 B |
| half `D64` online fallback | 51 | 12 B | 512 B | 0/0 B |
| half `D64` warp online | 30 | 12 B | 0 B | 0/0 B |

这里的 local/stack 是线程私有栈，不等同于编译器报告的显式 spill。warp 路径同时消除
512 字节私有数组并降低 half 临时寄存器数；这与 Event 的大幅提升方向一致。

### 12.6 报告与复现

原始报告保存在仓库外：

```text
~/mtt_profiles_s5000/musa_4_3_5/
  environment.log
  official_baseline.log
  official_final.log
  extended_baseline.log
  extended_final.log
  event_ab.csv
  float_attention_event.csv
  attention_threshold.csv
  compiler_resource_usage.log
  rms_full_baseline_dispatch_10x.log
  rms_full_final_dispatch_10x.log
```

一次性 benchmark 与实验源码位于 `~/mtt_kernel_lab/`，没有进入提交。正式仓库只保留
`src/kernels.mu` 和本文的 Moore 第一阶段变化。

## 13. MetaX 待验证阶段

2026-08-11 使用现有 `沐曦.txt` 进行了密码和 keyboard-interactive 两种登录尝试，
服务器均拒绝凭据。因此 `src/kernels.maca` 本轮没有经过 `mxcc` 或沐曦真机验证，
也没有把 S5000 dispatch 照搬过去。恢复访问后从包含 Moore 提交的最新 master 开始，
按下面步骤独立探测和调参。

环境记录：

```bash
mx-smi
mxcc --version
macainfo
which mcProfiler mcTracer
```

构建与官方测试：

```bash
make clean
make PLATFORM=metax VERBOSE=true
```

使用安装版本自带的 `mcProfiler --help` 与 `mcTracer --help` 确认参数后，分别采集
kernel 计数器和系统时间线。不同 MXMACA 版本的命令行选项可能变化，所以本文不硬编码
未经环境确认的采集参数。

重点确认 64-thread wave 下的 block 利用率、线程私有数组是否落入 private/local memory、
H2D/D2H 是否占据完整函数主要时间，以及 `mcMemcpy` 是否带来隐式同步。

## 14. 真机正确性矩阵

RMSNorm：

```text
rows: 1, 7, 64, 512
D: 1, 31, 32, 33, 127, 128, 255, 256, 257,
   768, 1024, 4096, 8192, 12288
data: zero, constant, random signed, large dynamic range
```

Attention：

```text
D: 1, 2, 4, 7, 8, 16, 31, 32, 33, 64, 128, 256
causal: false, true
sequence: square and Tq != Tk
heads: MHA and GQA ratios 2, 3, 4
data: zero, constant, random signed, large logits
```

建议阈值：

- RMSNorm float：`abs <= 2e-5` 或 `rel <= 2e-4`；
- RMSNorm half：`abs/rel <= 2e-3`；
- Attention float：`abs <= 2e-4` 或 `rel <= 2e-3`；
- Attention half：`abs <= 5e-3` 或 `rel <= 1e-2`。

## 15. 后续优化顺序

Moore 第一阶段已完成 Event、资源报告、warp 规约和 Attention warp-per-query。下一轮
在获得匹配 5.2 工具镜像后继续；MetaX 则从正确性基线开始，按以下顺序推进：

1. 记录 kernel-only 与完整函数时间，分离传输、分配和计算。
2. 查看编译器资源报告，确认寄存器、private/local memory、shared memory 和 occupancy。
3. RMSNorm 对比标量、float2/float4、half2/half8，并检查真实宽 load/store。
4. Attention 先为同一 KV head 聚合多个 query，测试 K/V shared-memory 复用。
5. 再比较三遍、online softmax、block 内 key 并行的数值和性能。
6. 厂商矩阵指令只用于 half 常见 D，并始终保留 FP32 softmax/PV 基线。
7. 仅在 profiler 证明 global-memory latency 是主要瓶颈时评估异步双缓冲。
8. 只有公开接口或调用方能提供 pinned buffer、stream 和 workspace 时再评估多流与 Graph。

任何优化路径都必须满足：官方正确性全通过、代表尺寸中位数提升至少 5%，且任一官方
尺寸的优化路径不能回退超过 5%。Sanitizer 可用时还必须零错误；工具缺失时必须像本轮
一样明确记录，不能用普通正确性测试代替。当前 Moore 代码可称为“S5000 真机验证并
调优版本”，MetaX 代码仍只能称为“等待真机验收的可移植正确性基线”。
