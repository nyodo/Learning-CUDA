# 天数智芯平台移植、验证与性能记录

本文解释为什么 NVIDIA 与天数智芯可以共用 `src/kernels.cu`，真正需要隔离的内容是什么，
以及 RMSNorm、Flash Attention 在 BI-V150 真机上的编译、正确性、Sanitizer 和性能结果。

结论先行：`.cu` 只是 CUDA 风格设备源码的扩展名，不代表它只能由 NVIDIA `nvcc`
编译。仓库的 Makefile 在 NVIDIA 平台调用 `nvcc`，在天数智芯平台调用 CoreX
`clang++`。两者共用文件没有链接冲突；必须用编译宏隔离的是 WMMA、PTX、warp 宽度和
设备架构等平台专属语义。

## 1. 同一个 `.cu` 为什么不冲突

Makefile 根据 `PLATFORM` 选择不同编译器和宏：

| 命令 | 编译器 | 宏 | 源文件 |
|---|---|---|---|
| `make PLATFORM=nvidia` | `nvcc` | `PLATFORM_NVIDIA` | `src/kernels.cu` |
| `make PLATFORM=iluvatar` | CoreX `clang++` | `PLATFORM_ILUVATAR` | `src/kernels.cu` |

一次构建只产生一个 `src/kernels.o`，因此不会把两个平台的目标代码链接到一起。CoreX
实现了 CUDA Runtime 兼容接口，所以两边都能使用 `cudaMalloc`、`cudaMemcpy`、
`cudaGetLastError` 和 `cudaFree`。`tester/utils.h` 也有意把这两个平台映射到 CUDA
Runtime API。

真正会冲突的是以下源码假设：

1. NVIDIA `<mma.h>` 和 `nvcuda::wmma` fragment 不能直接交给当前 CoreX WMMA 头文件。
2. NVIDIA warp 固定为 32 lane，BI-V150 实测为 64 lane。
3. `lane = threadIdx.x & 31`、32-lane shuffle reduction 和 partial-warp mask 不能照搬。
4. NVIDIA 内联 PTX、`cp.async` 和针对 `sm_120` 的指令不属于 CoreX ivcore ISA。
5. 预编译 tester 的设备镜像必须与实际天数智芯 GPU 架构匹配。

因此解决方法不是复制出另一个后缀，而是在同一接口下建立清楚的平台边界：

```cpp
#if defined(PLATFORM_NVIDIA)
#include <mma.h>
#endif

#if defined(PLATFORM_ILUVATAR)
// CoreX / 64-lane 实现
#else
// NVIDIA / 32-lane / WMMA 实现
#endif
```

这样公共的参数检查、连续显存分配和模板实例化仍只有一份，设备算法则分别编译。

## 2. 真机环境

本轮实际登录并检测到：

| 项目 | 结果 |
|---|---|
| GPU | Iluvatar BI-V150，32 GiB |
| GPU 数量 | 1 |
| CoreX | 4.4.0 |
| Driver | 4.4.0 |
| CUDA compatibility | 10.2 |
| CoreX 编译器 | clang++ 18.1.8 |
| 计算架构 | MR，`ivcore11` |
| CU 数量 | 16 |
| warpSize | 64 |
| maxThreadsPerBlock | 4096 |
| shared memory/block | 131072 B |
| registers/block | 262144 |
| 工具 | `ixsmi`、`ixobjdump`、`ixsan`、`ixsys`、`ixkn-cli` |

用 128 个线程组成两个真实 warp，每个 lane 输入 `lane + 1`，再执行宽度为 64 的
`__shfl_down_sync` 规约。两个 warp 的 lane 0 都得到 2080，等于
`1 + 2 + ... + 64`。这一步证明当前设备和工具链的 64-lane shuffle 语义可用，之后才
允许 Attention 默认走 warp 快路径。

## 3. 实现结构

### 3.1 RMSNorm

RMSNorm 保留一行一个 block、FP32 累加和 shared-memory 二叉树规约：

```text
每线程累加若干 x^2
        -> shared-memory block reduction
        -> rsqrt(sum / hidden_dim + eps)
        -> x * inv_rms * weight
```

没有把 64-lane shuffle 规约合入正式 RMSNorm。完整函数 A/B 在代表尺寸上只有
`0.98x` 到 `1.02x` 波动，达不到 5% 准入条件；最大的 case 还略慢。RMSNorm 当前接口
每次都执行分配、两次 H2D、kernel 和一次 D2H，几次 block 同步并不是完整函数主瓶颈。

所有类型都以 FP32 累加。标量路径覆盖任意隐藏维度；既有 `float4`/`half8` 路径继续
使用，但 Iluvatar 的 block reduction 不依赖 32-lane 假设。

宿主包装层还补齐了以下边界：

- 空形状清空输出并返回；
- 输出尺寸不正确时自动 resize；
- 检查 input/weight 长度；
- 检查元素数和字节数乘法溢出；
- 拒绝超过一维 grid 上限的 row 数。

### 3.2 Flash Attention fallback

CoreX 分支保留单线程一个 query row 的可靠 fallback，实验时可用
`ILUVATAR_FORCE_THREAD_ATTENTION` 强制启用：

- float 使用稳定三遍算法：max、sum、PV；
- half 使用 FP32 online softmax 和 FP32 输出累加；
- `D=1/2/4/8/16/32/64` 有编译期数组尺寸；
- 其他尺寸由通用 `D <= 256` 路径覆盖；
- 支持左上 causal、GQA 和非方形 Q/K 序列。

fallback 的意义是提供独立、容易审计的数值基线，不是大尺寸默认路径。

### 3.3 Flash Attention 64-lane 快路径

正式 CoreX Attention 使用一个 64-lane warp 处理一个 query row，四个 warp 组成
256-thread block：

1. 各 lane 缓存维度 `lane + item * 64` 的 Q 和输出累加器。
2. 各 lane 计算一部分 QK dot，再以 64-lane shuffle 得到完整 dot。
3. lane 0 更新 FP32 running max、running sum，并广播缩放系数和当前概率。
4. 各 lane 并行累加自己的 V 维度，不生成完整 attention matrix。
5. 最后广播 `1 / running_sum`，每个 lane 写回自己的输出维度。

`D <= 64/128/256` 分别使用每 lane 1/2/4 个 FP32 Q 和输出值。NCU 风格的资源检查
由 `ixkn-cli` 完成，代表 kernel 是 19 个 vector registers/thread、0 B local memory，
说明没有数组 spill。

online softmax 的更新公式为：

```text
m_new = max(m_old, score)
alpha = exp(m_old - m_new)          // 第一项时为 0
p     = exp(score - m_new)
l_new = alpha * l_old + p
o_new = alpha * o_old + p * v
```

最终输出 `o / l`。float 和 half 的点积、softmax 状态和输出累加均为 FP32。

## 4. 正确性结果

独立测试器使用 CPU double reference，不调用厂商 Attention 库，也不依赖课程预编译
tester 中的 GPU reference kernel。

| 测试 | float | half |
|---|---:|---:|
| RMSNorm 扩展尺寸 | 7/7 | 7/7 |
| Attention 官方形状 | 14/14 | 14/14 |
| Attention 额外形状 | 5/5 | 5/5 |
| 空 KV、空 RMS、溢出检查 | 通过 | 通过 |

Attention 覆盖 `D=1/2/4/7/8/16/31/32/33/64/128/256`，GQA 比例 2/3/4、
causal/non-causal、`Tq != Tk`，以及最大官方形状：

```text
B=4, Tq=512, Tk=2048, Hq=64, Hkv=64, D=32, causal=true
```

在 RTX 5090 隔离副本中又验证了同一候选 `kernels.cu`：

| NVIDIA 构建 | RMSNorm | Attention |
|---|---:|---:|
| 默认 `-O0` | float/half 26/26 | float/half 28/28 |
| `-O3 -lineinfo -arch=sm_120` | 全通过 | 全通过 |

这说明 `PLATFORM_ILUVATAR` 分支不会被 NVCC 实例化，NVIDIA 路径未被污染。

## 5. Attention A/B 性能

表中是完整函数中位数，包含分配、H2D、kernel 和 D2H。fallback 与 warp 版本在同一
进程内交替调用，输出也逐元素比较。

| 类型与形状 | warp ms | fallback ms | 加速 |
|---|---:|---:|---:|
| float `16x32 H8/H4 D16 causal` | 0.0981 | 0.1133 | 1.15x |
| half 同形状 | 0.0945 | 0.1029 | 1.09x |
| float `64x64 H16/H4 D8 causal` | 0.1431 | 0.1533 | 1.07x |
| half 同形状 | 0.1385 | 0.1384 | 1.00x |
| float `128x128 H16 D64` | 0.5658 | 6.0702 | 10.73x |
| half 同形状 | 0.4021 | 3.0486 | 7.58x |
| float `64x256 H8/H2 D128` | 0.4613 | 10.8740 | 23.57x |
| half 同形状 | 0.3633 | 6.9281 | 19.07x |
| float `32x128 H8/H2 D256 causal` | 0.2190 | 2.7376 | 12.50x |
| half 同形状 | 0.1385 | 1.8494 | 13.35x |
| float `256x512 H16/H4 D64` | 1.7141 | 5.9356 | 3.46x |
| half 同形状 | 1.5323 | 3.1076 | 2.03x |

小 case 被同步分配和传输延迟主导，所以加速有限；中大 case 的点积和 PV 并行度足以
覆盖 host 固定开销。所有 A/B 的最大绝对误差都不超过 `2.83e-7`（float）和
`1.22e-4`（half）。

## 6. IXKN 与 IXSYS 证据

代表 case 为 float `B1,Tq128,Tk128,Hq16,Hkv16,D64`，对应 512 blocks：

| IXKN 指标 | 数值 |
|---|---:|
| kernel duration | 202.930 us |
| block / grid | 256 threads / 512 blocks |
| achieved occupancy | 83.07% |
| theoretical occupancy | 100% |
| waves/CU | 1.00 |
| vector registers | 19/thread |
| scalar registers | 43/warp |
| local memory | 0 B/thread |
| global load throughput | 618.38 GB/s |
| L1 load hit rate | 91.33% |
| L2 load hit rate | 96.63% |
| LSU utilization | 73.46% |

IXKN 给出 “High Throughput，超过 80% compute 或 memory performance” 的提示。
因此第一版已经消除了 fallback 最明显的串行瓶颈，继续优化应以 shared K/V 复用或厂商
矩阵指令的真机 A/B 为依据，不能只凭理论增加复杂度。

报告同时显示 shared-store bank-conflict 计数，但源码没有显式 shared memory，
LaunchStats 也报告 static/dynamic shared memory 都是 0 B，shared load/store throughput
也是 0 B/s。最稳妥的解释是该计数包含 CoreX 对 subgroup/shuffle 的内部实现或工具统计
口径，不能把它当作源码中的 shared tile 冲突。若以后替换 subgroup 原语，应重新采集
同一 case 验证。

`ixsys` 已成功生成 CUDA API、memcpy、kernel 时间线。当前 `std::vector` 同步接口每次
都分配和回传，pinned buffer、多 H2D stream、Graph 或持久 workspace 需要改变资源生命
周期或引入缓存状态，因此没有仅为天数智芯分支塞进公共接口。

## 7. Sanitizer

用包含 RMSNorm D257 和 Attention GQA/causal/non-square D33 的 float/half smoke case：

| 工具 | 结果 |
|---|---|
| `ixsan --tool memcheck` | 0 errors，0 B leaked |
| `ixsan --tool initcheck` | 0 errors |
| `ixsan --tool racecheck` | 0 hazards，0 warnings |

当前 IXSAN 版本没有列出单独的 synccheck 模式，所以不能声称 synccheck 已运行。

## 8. 预编译 tester 的架构限制

BI-V150 上，仓库原始命令能够编译和链接：

```bash
make PLATFORM=iluvatar
```

`ixobjdump --lelf` 显示学生对象是：

```text
src/kernels.o.MR-Linux.ixbin
```

但 `tester/tester_iluvatar.o` 内嵌的是 `QS-Linux.ixbin`。运行时先启动 tester 自己的 GPU
reference kernel，于 `tester_iluvatar.cu:643` 报：

```text
no kernel image is available for execution on the device
```

这不是学生 kernel 错误，而是预编译 QS tester 与 MR/BI-V150 不匹配。只链接学生对象
和独立 CPU reference driver 时，全部测试可以运行并通过。提交环境如果是 QS 卡，应直接
使用官方 tester；BI-V150 的结果不能冒充 QS 官方 tester 通过。

显式面向本机编译的复现命令是：

```bash
export PATH=/usr/local/corex/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib64:$LD_LIBRARY_PATH

clang++ -x ivcore -std=c++17 -O3 --offload-arch=ivcore11 \
  -DPLATFORM_ILUVATAR -c src/kernels.cu -o src/kernels.o
```

没有把 `ivcore11` 写死进 Makefile，因为这样会破坏课程 QS 目标的可移植性。应由真实
部署环境或构建参数选择架构，而不是用 BI-V150 的实验值替代所有天数智芯设备。

## 9. 工具限制与后续方向

- CoreX 4.4 的 `ixkn-cli --set default` 可直接输出完整指标。
- `--set detailed -o ...` 已完成 replay 并写出报告，但进程在收尾时段错误；导入该报告
  又发生浮点异常。这是 profiler 工具链限制，本文只引用成功直接输出的 default 指标。
- 当前 warp 路径没有 local-memory spill，occupancy 和带宽利用率健康。
- 下一步最有价值的实验是让同一 KV head 的多个 query warp 共享 K/V tile，并和当前
  高 L1/L2 hit-rate 路径做严格 A/B；只有完整函数稳定提升至少 5% 才应进入 dispatch。
- 厂商矩阵指令需要针对 CoreX 支持的 fragment shape 单独设计，不能复用 NVIDIA WMMA
  类型。softmax 和输出累加仍应保持 FP32。

原始 `.ixsys`、`.ixkn`、CSV、A/B 和 Sanitizer 日志保存在真机仓库外的：

```text
~/iluvatar_profiles_biv150/
```

文档中的数字全部来自 BI-V150 真机；没有使用尚未获得的沐曦资源，也没有虚构 QS 平台
测试结果。
