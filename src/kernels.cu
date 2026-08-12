#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cuda_fp16.h>
#include <math_constants.h>
#if defined(PLATFORM_NVIDIA)
#include <mma.h>
#endif

#include "../tester/utils.h"

namespace {

constexpr int kWarpSize = 32;
constexpr size_t kDeviceAlignment = 256;

template <typename T>
__device__ __forceinline__ float toFloat(T value);

template <>
__device__ __forceinline__ float toFloat<float>(float value) {
  return value;
}

template <>
__device__ __forceinline__ float toFloat<half>(half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float value);

template <>
__device__ __forceinline__ float fromFloat<float>(float value) {
  return value;
}

template <>
__device__ __forceinline__ half fromFloat<half>(float value) {
  return __float2half_rn(value);
}

__device__ __forceinline__ float warpReduceSum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warpBroadcast(float value, int source_lane) {
  return __shfl_sync(0xffffffffu, value, source_lane);
}

template <int BlockSize>
__device__ __forceinline__ float blockReduceSum(float value) {
#if defined(PLATFORM_ILUVATAR)
  __shared__ float partial_sums[BlockSize];
  partial_sums[threadIdx.x] = value;
  __syncthreads();
  for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return partial_sums[0];
#else
  __shared__ float warp_sums[BlockSize / kWarpSize];

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  value = warpReduceSum(value);

  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  float block_sum = 0.0f;
  if (warp == 0) {
    block_sum = lane < BlockSize / kWarpSize ? warp_sums[lane] : 0.0f;
    block_sum = warpReduceSum(block_sum);
  }
  return block_sum;
#endif
}

template <typename T, int BlockSize>
__global__ void rmsNormScalarKernel(const T* __restrict__ input,
                                    const T* __restrict__ weight,
                                    T* __restrict__ output,
                                    size_t hidden_dim, float eps) {
  const size_t row_offset = static_cast<size_t>(blockIdx.x) * hidden_dim;
  float sum_squares = 0.0f;

  for (size_t col = threadIdx.x; col < hidden_dim; col += BlockSize) {
    const float value = toFloat(input[row_offset + col]);
    sum_squares = fmaf(value, value, sum_squares);
  }

  const float total = blockReduceSum<BlockSize>(sum_squares);
  __shared__ float inv_rms;
  if (threadIdx.x == 0) {
    inv_rms = rsqrtf(total / static_cast<float>(hidden_dim) + eps);
  }
  __syncthreads();

  for (size_t col = threadIdx.x; col < hidden_dim; col += BlockSize) {
    const float value = toFloat(input[row_offset + col]);
    const float scale = toFloat(weight[col]);
    output[row_offset + col] = fromFloat<T>(value * inv_rms * scale);
  }
}

template <int BlockSize, int PacksPerThread>
__global__ void rmsNormFloat4Kernel(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    float* __restrict__ output,
                                    size_t hidden_dim, float eps) {
  constexpr int kValuesPerPack = 4;
  const size_t vectors_per_row = hidden_dim / kValuesPerPack;
  const size_t vector_row_offset =
      static_cast<size_t>(blockIdx.x) * vectors_per_row;
  const float4* input4 = reinterpret_cast<const float4*>(input);
  const float4* weight4 = reinterpret_cast<const float4*>(weight);
  float4* output4 = reinterpret_cast<float4*>(output);

  float4 cached[PacksPerThread];
  float sum_squares = 0.0f;
#pragma unroll
  for (int pack = 0; pack < PacksPerThread; ++pack) {
    const size_t vec = threadIdx.x + static_cast<size_t>(pack) * BlockSize;
    float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (vec < vectors_per_row) {
      value = input4[vector_row_offset + vec];
    }
    cached[pack] = value;
    sum_squares = fmaf(value.x, value.x, sum_squares);
    sum_squares = fmaf(value.y, value.y, sum_squares);
    sum_squares = fmaf(value.z, value.z, sum_squares);
    sum_squares = fmaf(value.w, value.w, sum_squares);
  }

  const float total = blockReduceSum<BlockSize>(sum_squares);
  __shared__ float inv_rms;
  if (threadIdx.x == 0) {
    inv_rms = rsqrtf(total / static_cast<float>(hidden_dim) + eps);
  }
  __syncthreads();

#pragma unroll
  for (int pack = 0; pack < PacksPerThread; ++pack) {
    const size_t vec = threadIdx.x + static_cast<size_t>(pack) * BlockSize;
    if (vec < vectors_per_row) {
      const float4 value = cached[pack];
      const float4 scale = weight4[vec];
      output4[vector_row_offset + vec] =
          make_float4(value.x * inv_rms * scale.x,
                      value.y * inv_rms * scale.y,
                      value.z * inv_rms * scale.z,
                      value.w * inv_rms * scale.w);
    }
  }
}

union Half2Bits {
  uint32_t bits;
  half2 value;
};

__device__ __forceinline__ float2 half2BitsToFloat2(uint32_t bits) {
  Half2Bits packed;
  packed.bits = bits;
  return __half22float2(packed.value);
}

__device__ __forceinline__ uint32_t float2ToHalf2Bits(float x, float y) {
  Half2Bits packed;
  packed.value = __floats2half2_rn(x, y);
  return packed.bits;
}

template <int BlockSize, int PacksPerThread>
__global__ void rmsNormHalf8Kernel(const half* __restrict__ input,
                                   const half* __restrict__ weight,
                                   half* __restrict__ output,
                                   size_t hidden_dim, float eps) {
  constexpr int kValuesPerPack = 8;
  const size_t vectors_per_row = hidden_dim / kValuesPerPack;
  const size_t vector_row_offset =
      static_cast<size_t>(blockIdx.x) * vectors_per_row;
  const uint4* input8 = reinterpret_cast<const uint4*>(input);
  const uint4* weight8 = reinterpret_cast<const uint4*>(weight);
  uint4* output8 = reinterpret_cast<uint4*>(output);

  uint4 cached[PacksPerThread];
  float sum_squares = 0.0f;
#pragma unroll
  for (int pack = 0; pack < PacksPerThread; ++pack) {
    const size_t vec = threadIdx.x + static_cast<size_t>(pack) * BlockSize;
    uint4 raw = make_uint4(0u, 0u, 0u, 0u);
    if (vec < vectors_per_row) {
      raw = input8[vector_row_offset + vec];
    }
    cached[pack] = raw;

    const float2 xy = half2BitsToFloat2(raw.x);
    const float2 zw = half2BitsToFloat2(raw.y);
    const float2 uv = half2BitsToFloat2(raw.z);
    const float2 pq = half2BitsToFloat2(raw.w);
    sum_squares = fmaf(xy.x, xy.x, sum_squares);
    sum_squares = fmaf(xy.y, xy.y, sum_squares);
    sum_squares = fmaf(zw.x, zw.x, sum_squares);
    sum_squares = fmaf(zw.y, zw.y, sum_squares);
    sum_squares = fmaf(uv.x, uv.x, sum_squares);
    sum_squares = fmaf(uv.y, uv.y, sum_squares);
    sum_squares = fmaf(pq.x, pq.x, sum_squares);
    sum_squares = fmaf(pq.y, pq.y, sum_squares);
  }

  const float total = blockReduceSum<BlockSize>(sum_squares);
  __shared__ float inv_rms;
  if (threadIdx.x == 0) {
    inv_rms = rsqrtf(total / static_cast<float>(hidden_dim) + eps);
  }
  __syncthreads();

#pragma unroll
  for (int pack = 0; pack < PacksPerThread; ++pack) {
    const size_t vec = threadIdx.x + static_cast<size_t>(pack) * BlockSize;
    if (vec < vectors_per_row) {
      const uint4 raw = cached[pack];
      const uint4 raw_scale = weight8[vec];
      const float2 xy = half2BitsToFloat2(raw.x);
      const float2 zw = half2BitsToFloat2(raw.y);
      const float2 uv = half2BitsToFloat2(raw.z);
      const float2 pq = half2BitsToFloat2(raw.w);
      const float2 scale_xy = half2BitsToFloat2(raw_scale.x);
      const float2 scale_zw = half2BitsToFloat2(raw_scale.y);
      const float2 scale_uv = half2BitsToFloat2(raw_scale.z);
      const float2 scale_pq = half2BitsToFloat2(raw_scale.w);

      uint4 result;
      result.x = float2ToHalf2Bits(xy.x * inv_rms * scale_xy.x,
                                  xy.y * inv_rms * scale_xy.y);
      result.y = float2ToHalf2Bits(zw.x * inv_rms * scale_zw.x,
                                  zw.y * inv_rms * scale_zw.y);
      result.z = float2ToHalf2Bits(uv.x * inv_rms * scale_uv.x,
                                  uv.y * inv_rms * scale_uv.y);
      result.w = float2ToHalf2Bits(pq.x * inv_rms * scale_pq.x,
                                  pq.y * inv_rms * scale_pq.y);
      output8[vector_row_offset + vec] = result;
    }
  }
}

int chooseBlockSize(size_t work_items) {
  if (work_items <= 32) {
    return 32;
  }
  if (work_items <= 64) {
    return 64;
  }
  if (work_items <= 128) {
    return 128;
  }
  return 256;
}

int choosePackCount(size_t vectors_per_row, int block_size) {
  const size_t packs =
      (vectors_per_row + static_cast<size_t>(block_size) - 1) / block_size;
  if (packs <= 1) {
    return 1;
  }
  if (packs <= 2) {
    return 2;
  }
  if (packs <= 4) {
    return 4;
  }
  return packs <= 8 ? 8 : 0;
}

template <typename T, int BlockSize>
void launchScalarKernel(const T* input, const T* weight, T* output, size_t rows,
                        size_t hidden_dim, float eps) {
  rmsNormScalarKernel<T, BlockSize>
      <<<static_cast<unsigned int>(rows), BlockSize>>>(input, weight, output,
                                                       hidden_dim, eps);
}

template <typename T>
void launchScalar(const T* input, const T* weight, T* output, size_t rows,
                  size_t hidden_dim, float eps) {
  switch (chooseBlockSize(hidden_dim)) {
    case 32:
      launchScalarKernel<T, 32>(input, weight, output, rows, hidden_dim, eps);
      break;
    case 64:
      launchScalarKernel<T, 64>(input, weight, output, rows, hidden_dim, eps);
      break;
    case 128:
      launchScalarKernel<T, 128>(input, weight, output, rows, hidden_dim, eps);
      break;
    default:
      launchScalarKernel<T, 256>(input, weight, output, rows, hidden_dim, eps);
      break;
  }
}

template <int BlockSize>
void launchFloat4ByPack(const float* input, const float* weight, float* output,
                        size_t rows, size_t hidden_dim, float eps,
                        int packs_per_thread) {
  const dim3 grid(static_cast<unsigned int>(rows));
  switch (packs_per_thread) {
    case 1:
      rmsNormFloat4Kernel<BlockSize, 1>
          <<<grid, BlockSize>>>(input, weight, output, hidden_dim, eps);
      break;
    case 2:
      rmsNormFloat4Kernel<BlockSize, 2>
          <<<grid, BlockSize>>>(input, weight, output, hidden_dim, eps);
      break;
    case 4:
      rmsNormFloat4Kernel<BlockSize, 4>
          <<<grid, BlockSize>>>(input, weight, output, hidden_dim, eps);
      break;
    default:
      rmsNormFloat4Kernel<BlockSize, 8>
          <<<grid, BlockSize>>>(input, weight, output, hidden_dim, eps);
      break;
  }
}

template <int BlockSize>
void launchHalf8ByPack(const half* input, const half* weight, half* output,
                       size_t rows, size_t hidden_dim, float eps,
                       int packs_per_thread) {
  const dim3 grid(static_cast<unsigned int>(rows));
  switch (packs_per_thread) {
    case 1:
      rmsNormHalf8Kernel<BlockSize, 1>
          <<<grid, BlockSize>>>(input, weight, output, hidden_dim, eps);
      break;
    case 2:
      rmsNormHalf8Kernel<BlockSize, 2>
          <<<grid, BlockSize>>>(input, weight, output, hidden_dim, eps);
      break;
    case 4:
      rmsNormHalf8Kernel<BlockSize, 4>
          <<<grid, BlockSize>>>(input, weight, output, hidden_dim, eps);
      break;
    default:
      rmsNormHalf8Kernel<BlockSize, 8>
          <<<grid, BlockSize>>>(input, weight, output, hidden_dim, eps);
      break;
  }
}

bool launchVectorized(const float* input, const float* weight, float* output,
                      size_t rows, size_t hidden_dim, float eps) {
  constexpr size_t kValuesPerPack = 4;
  constexpr int kVectorBlockSize = 256;
  // Small hidden dimensions do not amortize the register-cached vector path.
  if (hidden_dim < 4096 || hidden_dim % kValuesPerPack != 0) {
    return false;
  }

  const size_t vectors_per_row = hidden_dim / kValuesPerPack;
  const int packs_per_thread =
      choosePackCount(vectors_per_row, kVectorBlockSize);
  if (packs_per_thread == 0) {
    return false;
  }

  launchFloat4ByPack<kVectorBlockSize>(input, weight, output, rows, hidden_dim,
                                       eps, packs_per_thread);
  return true;
}

bool launchVectorized(const half* input, const half* weight, half* output,
                      size_t rows, size_t hidden_dim, float eps) {
  constexpr size_t kValuesPerPack = 8;
  constexpr int kVectorBlockSize = 256;
  // Small hidden dimensions do not amortize the register-cached vector path.
  if (hidden_dim < 4096 || hidden_dim % kValuesPerPack != 0) {
    return false;
  }

  const size_t vectors_per_row = hidden_dim / kValuesPerPack;
  const int packs_per_thread =
      choosePackCount(vectors_per_row, kVectorBlockSize);
  if (packs_per_thread == 0) {
    return false;
  }

  launchHalf8ByPack<kVectorBlockSize>(input, weight, output, rows, hidden_dim,
                                      eps, packs_per_thread);
  return true;
}

constexpr int kMaxAttentionHeadDim = 256;

#if defined(PLATFORM_ILUVATAR)

template <int HeadDim>
__global__ void flashAttentionIluvatarFloatKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ output,
    size_t query_rows, int target_seq_len, int src_seq_len, int query_heads,
    int kv_heads, int runtime_head_dim, bool is_causal) {
  enum { kCacheSize = HeadDim == 0 ? kMaxAttentionHeadDim : HeadDim };
  const size_t query_row = static_cast<size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
  if (query_row >= query_rows) {
    return;
  }

  const int head_dim = HeadDim == 0 ? runtime_head_dim : HeadDim;
  const size_t rows_per_batch =
      static_cast<size_t>(target_seq_len) * query_heads;
  const int batch = static_cast<int>(query_row / rows_per_batch);
  const size_t row_in_batch =
      query_row - static_cast<size_t>(batch) * rows_per_batch;
  const int query_pos = static_cast<int>(row_in_batch / query_heads);
  const int query_head = static_cast<int>(row_in_batch % query_heads);
  const int kv_head = query_head / (query_heads / kv_heads);
  const size_t q_offset = query_row * head_dim;
  const int key_end =
      is_causal ? min(src_seq_len, query_pos + 1) : src_seq_len;
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  float q_cache[kCacheSize];
  float output_cache[kCacheSize];
#pragma unroll
  for (int dim = 0; dim < head_dim; ++dim) {
    q_cache[dim] = q[q_offset + dim];
    output_cache[dim] = 0.0f;
  }

  float row_max = -CUDART_INF_F;
  for (int key_pos = 0; key_pos < key_end; ++key_pos) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + key_pos) * kv_heads +
         kv_head) * head_dim;
    float dot = 0.0f;
#pragma unroll
    for (int dim = 0; dim < head_dim; ++dim) {
      dot = fmaf(q_cache[dim], k[kv_offset + dim], dot);
    }
    row_max = fmaxf(row_max, dot * scale);
  }

  float row_sum = 0.0f;
  for (int key_pos = 0; key_pos < key_end; ++key_pos) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + key_pos) * kv_heads +
         kv_head) * head_dim;
    float dot = 0.0f;
#pragma unroll
    for (int dim = 0; dim < head_dim; ++dim) {
      dot = fmaf(q_cache[dim], k[kv_offset + dim], dot);
    }
    row_sum += expf(dot * scale - row_max);
  }

  const float inverse_sum = 1.0f / row_sum;
  for (int key_pos = 0; key_pos < key_end; ++key_pos) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + key_pos) * kv_heads +
         kv_head) * head_dim;
    float dot = 0.0f;
#pragma unroll
    for (int dim = 0; dim < head_dim; ++dim) {
      dot = fmaf(q_cache[dim], k[kv_offset + dim], dot);
    }
    const float probability = expf(dot * scale - row_max) * inverse_sum;
#pragma unroll
    for (int dim = 0; dim < head_dim; ++dim) {
      output_cache[dim] =
          fmaf(probability, v[kv_offset + dim], output_cache[dim]);
    }
  }

#pragma unroll
  for (int dim = 0; dim < head_dim; ++dim) {
    output[q_offset + dim] = output_cache[dim];
  }
}

template <int HeadDim>
__global__ void flashAttentionIluvatarHalfKernel(
    const half* __restrict__ q, const half* __restrict__ k,
    const half* __restrict__ v, half* __restrict__ output,
    size_t query_rows, int target_seq_len, int src_seq_len, int query_heads,
    int kv_heads, int runtime_head_dim, bool is_causal) {
  enum { kCacheSize = HeadDim == 0 ? kMaxAttentionHeadDim : HeadDim };
  const size_t query_row = static_cast<size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
  if (query_row >= query_rows) {
    return;
  }

  const int head_dim = HeadDim == 0 ? runtime_head_dim : HeadDim;
  const size_t rows_per_batch =
      static_cast<size_t>(target_seq_len) * query_heads;
  const int batch = static_cast<int>(query_row / rows_per_batch);
  const size_t row_in_batch =
      query_row - static_cast<size_t>(batch) * rows_per_batch;
  const int query_pos = static_cast<int>(row_in_batch / query_heads);
  const int query_head = static_cast<int>(row_in_batch % query_heads);
  const int kv_head = query_head / (query_heads / kv_heads);
  const size_t q_offset = query_row * head_dim;
  const int key_end =
      is_causal ? min(src_seq_len, query_pos + 1) : src_seq_len;
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  float q_cache[kCacheSize];
  float output_cache[kCacheSize];
#pragma unroll
  for (int dim = 0; dim < head_dim; ++dim) {
    q_cache[dim] = toFloat(q[q_offset + dim]);
    output_cache[dim] = 0.0f;
  }

  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;
  for (int key_pos = 0; key_pos < key_end; ++key_pos) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + key_pos) * kv_heads +
         kv_head) * head_dim;
    float dot = 0.0f;
#pragma unroll
    for (int dim = 0; dim < head_dim; ++dim) {
      dot = fmaf(q_cache[dim], toFloat(k[kv_offset + dim]), dot);
    }
    const float score = dot * scale;
    const float new_max = fmaxf(running_max, score);
    const float previous_scale =
        running_sum == 0.0f ? 0.0f : expf(running_max - new_max);
    const float probability = expf(score - new_max);
    running_sum = running_sum * previous_scale + probability;
#pragma unroll
    for (int dim = 0; dim < head_dim; ++dim) {
      output_cache[dim] =
          fmaf(probability, toFloat(v[kv_offset + dim]),
               output_cache[dim] * previous_scale);
    }
    running_max = new_max;
  }

  const float inverse_sum = 1.0f / running_sum;
#pragma unroll
  for (int dim = 0; dim < head_dim; ++dim) {
    output[q_offset + dim] = fromFloat<half>(output_cache[dim] * inverse_sum);
  }
}

constexpr int kIluvatarWarpSize = 64;
constexpr int kIluvatarAttentionWarpsPerBlock = 4;
constexpr int kIluvatarAttentionBlockSize =
    kIluvatarWarpSize * kIluvatarAttentionWarpsPerBlock;

__device__ __forceinline__ float iluvatarWarpReduceSum(float value) {
#pragma unroll
  for (int offset = kIluvatarWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset, kIluvatarWarpSize);
  }
  return value;
}

__device__ __forceinline__ float iluvatarWarpBroadcast(float value) {
  return __shfl_sync(0xffffffffu, value, 0, kIluvatarWarpSize);
}

template <typename T, int ValuesPerLane>
__global__ void flashAttentionIluvatarWarpKernel(
    const T* __restrict__ q, const T* __restrict__ k,
    const T* __restrict__ v, T* __restrict__ output, size_t query_rows,
    int target_seq_len, int src_seq_len, int query_heads, int kv_heads,
    int head_dim, bool is_causal) {
  const int lane = threadIdx.x & (kIluvatarWarpSize - 1);
  const int warp = threadIdx.x / kIluvatarWarpSize;
  const size_t query_row =
      static_cast<size_t>(blockIdx.x) * kIluvatarAttentionWarpsPerBlock + warp;
  if (query_row >= query_rows) {
    return;
  }

  const size_t rows_per_batch =
      static_cast<size_t>(target_seq_len) * query_heads;
  const int batch = static_cast<int>(query_row / rows_per_batch);
  const size_t row_in_batch =
      query_row - static_cast<size_t>(batch) * rows_per_batch;
  const int query_pos = static_cast<int>(row_in_batch / query_heads);
  const int query_head = static_cast<int>(row_in_batch % query_heads);
  const int kv_head = query_head / (query_heads / kv_heads);
  const size_t q_offset = query_row * head_dim;

  float q_cache[ValuesPerLane];
  float output_cache[ValuesPerLane];
#pragma unroll
  for (int item = 0; item < ValuesPerLane; ++item) {
    const int dim = lane + item * kIluvatarWarpSize;
    q_cache[item] = dim < head_dim ? toFloat(q[q_offset + dim]) : 0.0f;
    output_cache[item] = 0.0f;
  }

  const int key_end =
      is_causal ? min(src_seq_len, query_pos + 1) : src_seq_len;
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;

  for (int key_pos = 0; key_pos < key_end; ++key_pos) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + key_pos) * kv_heads +
         kv_head) * head_dim;
    float dot = 0.0f;
#pragma unroll
    for (int item = 0; item < ValuesPerLane; ++item) {
      const int dim = lane + item * kIluvatarWarpSize;
      if (dim < head_dim) {
        dot = fmaf(q_cache[item], toFloat(k[kv_offset + dim]), dot);
      }
    }
    dot = iluvatarWarpReduceSum(dot);

    float previous_scale = 0.0f;
    float probability = 0.0f;
    if (lane == 0) {
      const float score = dot * scale;
      const float new_max = fmaxf(running_max, score);
      previous_scale = running_sum == 0.0f
                           ? 0.0f
                           : expf(running_max - new_max);
      probability = expf(score - new_max);
      running_sum = running_sum * previous_scale + probability;
      running_max = new_max;
    }
    previous_scale = iluvatarWarpBroadcast(previous_scale);
    probability = iluvatarWarpBroadcast(probability);

#pragma unroll
    for (int item = 0; item < ValuesPerLane; ++item) {
      const int dim = lane + item * kIluvatarWarpSize;
      if (dim < head_dim) {
        output_cache[item] =
            fmaf(probability, toFloat(v[kv_offset + dim]),
                 output_cache[item] * previous_scale);
      }
    }
  }

  float inverse_sum = lane == 0 && running_sum > 0.0f
                          ? 1.0f / running_sum
                          : 0.0f;
  inverse_sum = iluvatarWarpBroadcast(inverse_sum);
#pragma unroll
  for (int item = 0; item < ValuesPerLane; ++item) {
    const int dim = lane + item * kIluvatarWarpSize;
    if (dim < head_dim) {
      output[q_offset + dim] =
          fromFloat<T>(output_cache[item] * inverse_sum);
    }
  }
}

template <typename T, int ValuesPerLane>
void launchIluvatarWarpSpecialization(
    const T* q, const T* k, const T* v, T* output, size_t query_rows,
    int target_seq_len, int src_seq_len, int query_heads, int kv_heads,
    int head_dim, bool is_causal) {
  const unsigned int blocks = static_cast<unsigned int>(
      (query_rows + kIluvatarAttentionWarpsPerBlock - 1) /
      kIluvatarAttentionWarpsPerBlock);
  flashAttentionIluvatarWarpKernel<T, ValuesPerLane>
      <<<blocks, kIluvatarAttentionBlockSize>>>(
          q, k, v, output, query_rows, target_seq_len, src_seq_len,
          query_heads, kv_heads, head_dim, is_causal);
}

template <typename T>
void launchIluvatarWarp(const T* q, const T* k, const T* v, T* output,
                        size_t query_rows, int target_seq_len,
                        int src_seq_len, int query_heads, int kv_heads,
                        int head_dim, bool is_causal) {
  if (head_dim <= 64) {
    launchIluvatarWarpSpecialization<T, 1>(
        q, k, v, output, query_rows, target_seq_len, src_seq_len, query_heads,
        kv_heads, head_dim, is_causal);
  } else if (head_dim <= 128) {
    launchIluvatarWarpSpecialization<T, 2>(
        q, k, v, output, query_rows, target_seq_len, src_seq_len, query_heads,
        kv_heads, head_dim, is_causal);
  } else {
    launchIluvatarWarpSpecialization<T, 4>(
        q, k, v, output, query_rows, target_seq_len, src_seq_len, query_heads,
        kv_heads, head_dim, is_causal);
  }
}

template <int HeadDim>
void launchIluvatarFloatSpecialization(
    const float* q, const float* k, const float* v, float* output,
    size_t query_rows, int target_seq_len, int src_seq_len, int query_heads,
    int kv_heads, int runtime_head_dim, bool is_causal) {
  constexpr int kBlockSize = 256;
  const unsigned int blocks = static_cast<unsigned int>(
      (query_rows + kBlockSize - 1) / kBlockSize);
  flashAttentionIluvatarFloatKernel<HeadDim><<<blocks, kBlockSize>>>(
      q, k, v, output, query_rows, target_seq_len, src_seq_len, query_heads,
      kv_heads, runtime_head_dim, is_causal);
}

template <int HeadDim>
void launchIluvatarHalfSpecialization(
    const half* q, const half* k, const half* v, half* output,
    size_t query_rows, int target_seq_len, int src_seq_len, int query_heads,
    int kv_heads, int runtime_head_dim, bool is_causal) {
  constexpr int kBlockSize = 256;
  const unsigned int blocks = static_cast<unsigned int>(
      (query_rows + kBlockSize - 1) / kBlockSize);
  flashAttentionIluvatarHalfKernel<HeadDim><<<blocks, kBlockSize>>>(
      q, k, v, output, query_rows, target_seq_len, src_seq_len, query_heads,
      kv_heads, runtime_head_dim, is_causal);
}

#define LAUNCH_ILUVATAR_CASES(LAUNCH)                                         \
  case 1:                                                                     \
    LAUNCH<1>(q, k, v, output, query_rows, target_seq_len, src_seq_len,        \
              query_heads, kv_heads, head_dim, is_causal);                    \
    break;                                                                    \
  case 2:                                                                     \
    LAUNCH<2>(q, k, v, output, query_rows, target_seq_len, src_seq_len,        \
              query_heads, kv_heads, head_dim, is_causal);                    \
    break;                                                                    \
  case 4:                                                                     \
    LAUNCH<4>(q, k, v, output, query_rows, target_seq_len, src_seq_len,        \
              query_heads, kv_heads, head_dim, is_causal);                    \
    break;                                                                    \
  case 8:                                                                     \
    LAUNCH<8>(q, k, v, output, query_rows, target_seq_len, src_seq_len,        \
              query_heads, kv_heads, head_dim, is_causal);                    \
    break;                                                                    \
  case 16:                                                                    \
    LAUNCH<16>(q, k, v, output, query_rows, target_seq_len, src_seq_len,       \
               query_heads, kv_heads, head_dim, is_causal);                   \
    break;                                                                    \
  case 32:                                                                    \
    LAUNCH<32>(q, k, v, output, query_rows, target_seq_len, src_seq_len,       \
               query_heads, kv_heads, head_dim, is_causal);                   \
    break;                                                                    \
  case 64:                                                                    \
    LAUNCH<64>(q, k, v, output, query_rows, target_seq_len, src_seq_len,       \
               query_heads, kv_heads, head_dim, is_causal);                   \
    break

void launchFlashAttention(const float* q, const float* k, const float* v,
                          float* output, int batch_size, int target_seq_len,
                          int src_seq_len, int query_heads, int kv_heads,
                          int head_dim, bool is_causal) {
  const size_t query_rows = static_cast<size_t>(batch_size) * target_seq_len *
                            query_heads;
#if defined(ILUVATAR_FORCE_THREAD_ATTENTION)
  switch (head_dim) {
    LAUNCH_ILUVATAR_CASES(launchIluvatarFloatSpecialization);
    default:
      launchIluvatarFloatSpecialization<0>(
          q, k, v, output, query_rows, target_seq_len, src_seq_len,
          query_heads, kv_heads, head_dim, is_causal);
      break;
  }
#else
  launchIluvatarWarp(q, k, v, output, query_rows, target_seq_len, src_seq_len,
                     query_heads, kv_heads, head_dim, is_causal);
#endif
}

void launchFlashAttention(const half* q, const half* k, const half* v,
                          half* output, int batch_size, int target_seq_len,
                          int src_seq_len, int query_heads, int kv_heads,
                          int head_dim, bool is_causal) {
  const size_t query_rows = static_cast<size_t>(batch_size) * target_seq_len *
                            query_heads;
#if defined(ILUVATAR_FORCE_THREAD_ATTENTION)
  switch (head_dim) {
    LAUNCH_ILUVATAR_CASES(launchIluvatarHalfSpecialization);
    default:
      launchIluvatarHalfSpecialization<0>(
          q, k, v, output, query_rows, target_seq_len, src_seq_len,
          query_heads, kv_heads, head_dim, is_causal);
      break;
  }
#else
  launchIluvatarWarp(q, k, v, output, query_rows, target_seq_len, src_seq_len,
                     query_heads, kv_heads, head_dim, is_causal);
#endif
}

#undef LAUNCH_ILUVATAR_CASES

#else

constexpr int kAttentionWarpsPerBlock = 8;
constexpr int kAttentionBlockSize = kAttentionWarpsPerBlock * kWarpSize;

template <typename T, int ValuesPerLane>
__global__ void flashAttentionWarpOnlineKernel(
    const T* __restrict__ q, const T* __restrict__ k,
    const T* __restrict__ v, T* __restrict__ output, int batch_size,
    int target_seq_len,
    int src_seq_len, int query_heads, int kv_heads, int head_dim,
    bool is_causal) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const size_t query_row =
      static_cast<size_t>(blockIdx.x) * kAttentionWarpsPerBlock + warp;
  const size_t rows_per_batch =
      static_cast<size_t>(target_seq_len) * query_heads;
  const size_t total_rows = static_cast<size_t>(batch_size) * rows_per_batch;
  if (query_row >= total_rows) {
    return;
  }

  const int batch = static_cast<int>(query_row / rows_per_batch);
  const size_t row_in_batch = query_row -
      static_cast<size_t>(batch) * rows_per_batch;
  const int query_pos = static_cast<int>(row_in_batch / query_heads);
  const int query_head = static_cast<int>(row_in_batch % query_heads);
  const int queries_per_kv = query_heads / kv_heads;
  const int kv_head = query_head / queries_per_kv;
  const size_t q_offset = query_row * head_dim;

  float q_cache[ValuesPerLane];
  float out_acc[ValuesPerLane];
#pragma unroll
  for (int item = 0; item < ValuesPerLane; ++item) {
    const int dim = lane + item * kWarpSize;
    q_cache[item] = dim < head_dim ? toFloat(q[q_offset + dim]) : 0.0f;
    out_acc[item] = 0.0f;
  }

  const int key_end = is_causal ? min(src_seq_len, query_pos + 1)
                                : src_seq_len;
  const float scale =
      static_cast<float>(1.0 / sqrt(static_cast<double>(head_dim)));
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;

  for (int key_pos = 0; key_pos < key_end; ++key_pos) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + key_pos) * kv_heads +
         kv_head) * head_dim;
    float dot = 0.0f;
#pragma unroll
    for (int item = 0; item < ValuesPerLane; ++item) {
      const int dim = lane + item * kWarpSize;
      if (dim < head_dim) {
        dot = fmaf(q_cache[item], toFloat(k[kv_offset + dim]), dot);
      }
    }
    dot = warpReduceSum(dot);

    float alpha = 0.0f;
    float probability = 0.0f;
    if (lane == 0) {
      const float score = dot * scale;
      const float new_max = fmaxf(running_max, score);
      alpha = running_max == -CUDART_INF_F
                  ? 0.0f
                  : expf(running_max - new_max);
      probability = expf(score - new_max);
      running_sum = running_sum * alpha + probability;
      running_max = new_max;
    }
    alpha = warpBroadcast(alpha, 0);
    probability = warpBroadcast(probability, 0);

#pragma unroll
    for (int item = 0; item < ValuesPerLane; ++item) {
      const int dim = lane + item * kWarpSize;
      if (dim < head_dim) {
        out_acc[item] =
            fmaf(probability, toFloat(v[kv_offset + dim]),
                 out_acc[item] * alpha);
      }
    }
  }

  float inverse_sum = 0.0f;
  if (lane == 0 && running_sum > 0.0f) {
    inverse_sum = 1.0f / running_sum;
  }
  inverse_sum = warpBroadcast(inverse_sum, 0);
#pragma unroll
  for (int item = 0; item < ValuesPerLane; ++item) {
    const int dim = lane + item * kWarpSize;
    if (dim < head_dim) {
      output[q_offset + dim] = fromFloat<T>(out_acc[item] * inverse_sum);
    }
  }
}

template <typename T, int ValuesPerLane>
void launchFlashWarpOnline(const T* q, const T* k, const T* v, T* output,
                           int batch_size, int target_seq_len,
                           int src_seq_len, int query_heads, int kv_heads,
                           int head_dim, bool is_causal) {
  const size_t query_rows = static_cast<size_t>(batch_size) * target_seq_len *
                            query_heads;
  const unsigned int blocks = static_cast<unsigned int>(
      (query_rows + kAttentionWarpsPerBlock - 1) /
      kAttentionWarpsPerBlock);
  const dim3 grid(blocks);
  flashAttentionWarpOnlineKernel<T, ValuesPerLane>
      <<<grid, kAttentionBlockSize>>>(q, k, v, output, batch_size,
                                      target_seq_len, src_seq_len, query_heads,
                                      kv_heads, head_dim, is_causal);
}

template <typename T>
void launchFlashWarpOnline(const T* q, const T* k, const T* v, T* output,
                           int batch_size, int target_seq_len,
                           int src_seq_len, int query_heads, int kv_heads,
                           int head_dim, bool is_causal) {
  if (head_dim <= 32) {
    launchFlashWarpOnline<T, 1>(q, k, v, output, batch_size, target_seq_len,
                                src_seq_len, query_heads, kv_heads, head_dim,
                                is_causal);
  } else if (head_dim <= 64) {
    launchFlashWarpOnline<T, 2>(q, k, v, output, batch_size, target_seq_len,
                                src_seq_len, query_heads, kv_heads, head_dim,
                                is_causal);
  } else if (head_dim <= 128) {
    launchFlashWarpOnline<T, 4>(q, k, v, output, batch_size, target_seq_len,
                                src_seq_len, query_heads, kv_heads, head_dim,
                                is_causal);
  } else {
    launchFlashWarpOnline<T, 8>(q, k, v, output, batch_size, target_seq_len,
                                src_seq_len, query_heads, kv_heads, head_dim,
                                is_causal);
  }
}

constexpr int kAttentionKeyTile = 32;

template <typename T, int ValuesPerLane>
__global__ void flashAttentionSharedKvKernel(
    const T* __restrict__ q, const T* __restrict__ k,
    const T* __restrict__ v, T* __restrict__ output, int target_seq_len,
    int src_seq_len, int query_heads, int kv_heads, int head_dim,
    bool is_causal, int query_tiles_per_kv) {
  extern __shared__ float shared[];
  float* const shared_q = shared;
  float* const shared_k = shared_q + kAttentionWarpsPerBlock * head_dim;
  float* const shared_v = shared_k + kAttentionKeyTile * head_dim;
  __shared__ int warp_key_ends[kAttentionWarpsPerBlock];
  __shared__ int cta_key_end;

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int blocks_per_batch = kv_heads * query_tiles_per_kv;
  const int batch = static_cast<int>(blockIdx.x) / blocks_per_batch;
  const int block_in_batch = static_cast<int>(blockIdx.x) -
                             batch * blocks_per_batch;
  const int kv_head = block_in_batch / query_tiles_per_kv;
  const int query_tile = block_in_batch - kv_head * query_tiles_per_kv;
  const int queries_per_kv = query_heads / kv_heads;
  const int rows_per_kv = target_seq_len * queries_per_kv;
  const int query_in_kv = query_tile * kAttentionWarpsPerBlock + warp;
  const bool active = query_in_kv < rows_per_kv;
  const int query_pos = active ? query_in_kv / queries_per_kv : 0;
  const int query_head =
      active ? kv_head * queries_per_kv + query_in_kv % queries_per_kv : 0;
  const size_t query_row =
      (static_cast<size_t>(batch) * target_seq_len + query_pos) * query_heads +
      query_head;
  const size_t q_offset = query_row * head_dim;

  const int q_values = kAttentionWarpsPerBlock * head_dim;
  for (int item = threadIdx.x; item < q_values;
       item += kAttentionBlockSize) {
    const int item_warp = item / head_dim;
    const int dim = item - item_warp * head_dim;
    const int item_query_in_kv =
        query_tile * kAttentionWarpsPerBlock + item_warp;
    if (item_query_in_kv < rows_per_kv) {
      const int item_query_pos = item_query_in_kv / queries_per_kv;
      const int item_query_head =
          kv_head * queries_per_kv + item_query_in_kv % queries_per_kv;
      const size_t item_q_offset =
          ((static_cast<size_t>(batch) * target_seq_len + item_query_pos) *
               query_heads +
           item_query_head) * head_dim;
      shared_q[item] = toFloat(q[item_q_offset + dim]);
    } else {
      shared_q[item] = 0.0f;
    }
  }

  const int key_end =
      active ? (is_causal ? min(src_seq_len, query_pos + 1) : src_seq_len) : 0;
  if (lane == 0) {
    warp_key_ends[warp] = key_end;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int maximum = 0;
    for (int item_warp = 0; item_warp < kAttentionWarpsPerBlock;
         ++item_warp) {
      maximum = max(maximum, warp_key_ends[item_warp]);
    }
    cta_key_end = maximum;
  }
  __syncthreads();

  float root;
  const float dimension = static_cast<float>(head_dim);
  asm("sqrt.rn.f32 %0, %1;" : "=f"(root) : "f"(dimension));
  float scale;
  asm("rcp.rn.f32 %0, %1;" : "=f"(scale) : "f"(root));
  float row_max = -CUDART_INF_F;
  float row_sum = 0.0f;
  float output_cache[ValuesPerLane];
#pragma unroll
  for (int item = 0; item < ValuesPerLane; ++item) {
    output_cache[item] = 0.0f;
  }

  for (int pass = 0; pass < 3; ++pass) {
    float inverse_sum = 0.0f;
    if (pass == 2 && lane == 0 && active) {
      asm("rcp.rn.f32 %0, %1;" : "=f"(inverse_sum) : "f"(row_sum));
    }
    inverse_sum = warpBroadcast(inverse_sum, 0);

    for (int key_base = 0; key_base < cta_key_end;
         key_base += kAttentionKeyTile) {
      const int tile_keys = min(kAttentionKeyTile, cta_key_end - key_base);
      const int tile_values = tile_keys * head_dim;
      for (int item = threadIdx.x; item < tile_values;
           item += kAttentionBlockSize) {
        const int key_in_tile = item / head_dim;
        const int dim = item - key_in_tile * head_dim;
        const int key_pos = key_base + key_in_tile;
        const size_t kv_offset =
            ((static_cast<size_t>(batch) * src_seq_len + key_pos) * kv_heads +
             kv_head) * head_dim + dim;
        shared_k[item] = toFloat(k[kv_offset]);
        if (pass == 2) {
          shared_v[item] = toFloat(v[kv_offset]);
        }
      }
      __syncthreads();

      if (pass < 2) {
        if (active && lane == 0) {
          const int warp_key_count = min(tile_keys, max(0, key_end - key_base));
          for (int key_in_tile = 0; key_in_tile < warp_key_count;
               ++key_in_tile) {
            float dot = 0.0f;
            const int key_offset = key_in_tile * head_dim;
            const int query_offset = warp * head_dim;
            for (int dim = 0; dim < head_dim; ++dim) {
              dot = fmaf(shared_q[query_offset + dim],
                         shared_k[key_offset + dim], dot);
            }
            const float score = dot * scale;
            if (pass == 0) {
              row_max = fmaxf(row_max, score);
            } else {
              row_sum = fmaf(expf(score - row_max), 1.0f, row_sum);
            }
          }
        }
      } else if (active) {
        const int warp_key_count = min(tile_keys, max(0, key_end - key_base));
        for (int key_in_tile = 0; key_in_tile < warp_key_count;
             ++key_in_tile) {
          float probability = 0.0f;
          if (lane == 0) {
            float dot = 0.0f;
            const int key_offset = key_in_tile * head_dim;
            const int query_offset = warp * head_dim;
            for (int dim = 0; dim < head_dim; ++dim) {
              dot = fmaf(shared_q[query_offset + dim],
                         shared_k[key_offset + dim], dot);
            }
            const float exponent = dot * scale - row_max;
            probability = expf(exponent) * inverse_sum;
          }
          probability = warpBroadcast(probability, 0);
#pragma unroll
          for (int item = 0; item < ValuesPerLane; ++item) {
            const int dim = lane + item * kWarpSize;
            if (dim < head_dim) {
              output_cache[item] =
                  fmaf(probability,
                       shared_v[key_in_tile * head_dim + dim],
                       output_cache[item]);
            }
          }
        }
      }
      __syncthreads();
    }
  }

  if (active) {
#pragma unroll
    for (int item = 0; item < ValuesPerLane; ++item) {
      const int dim = lane + item * kWarpSize;
      if (dim < head_dim) {
        output[q_offset + dim] = fromFloat<T>(output_cache[item]);
      }
    }
  }
}

template <typename T, int ValuesPerLane>
void launchFlashSharedKv(const T* q, const T* k, const T* v, T* output,
                         int batch_size, int target_seq_len, int src_seq_len,
                         int query_heads, int kv_heads, int head_dim,
                         bool is_causal) {
  const int queries_per_kv = query_heads / kv_heads;
  const int query_tiles_per_kv =
      (target_seq_len * queries_per_kv + kAttentionWarpsPerBlock - 1) /
      kAttentionWarpsPerBlock;
  const unsigned int blocks = static_cast<unsigned int>(
      batch_size * kv_heads * query_tiles_per_kv);
  const size_t shared_values =
      static_cast<size_t>(kAttentionWarpsPerBlock + 2 * kAttentionKeyTile) *
      head_dim;
  flashAttentionSharedKvKernel<T, ValuesPerLane>
      <<<blocks, kAttentionBlockSize, shared_values * sizeof(float)>>>(
          q, k, v, output, target_seq_len, src_seq_len, query_heads, kv_heads,
          head_dim, is_causal, query_tiles_per_kv);
}

template <typename T>
bool launchFlashSharedKv(const T* q, const T* k, const T* v, T* output,
                         int batch_size, int target_seq_len, int src_seq_len,
                         int query_heads, int kv_heads, int head_dim,
                         bool is_causal) {
  const int rows_per_kv = target_seq_len * (query_heads / kv_heads);
  if (head_dim > 64 || src_seq_len < 16 || rows_per_kv < 8) {
    return false;
  }
  if (head_dim <= 32) {
    launchFlashSharedKv<T, 1>(q, k, v, output, batch_size, target_seq_len,
                              src_seq_len, query_heads, kv_heads, head_dim,
                              is_causal);
  } else {
    launchFlashSharedKv<T, 2>(q, k, v, output, batch_size, target_seq_len,
                              src_seq_len, query_heads, kv_heads, head_dim,
                              is_causal);
  }
  return true;
}

constexpr int kWmmaWarps = 4;
constexpr int kWmmaBlockSize = kWmmaWarps * kWarpSize;
constexpr int kWmmaQueriesPerBlock = kWmmaWarps * 16;
constexpr int kWmmaKeyTile = 16;

template <int HeadDim, int ValuesPerLane>
__global__ void flashAttentionHalfWmmaKernel(
    const half* __restrict__ q, const half* __restrict__ k,
    const half* __restrict__ v, half* __restrict__ output,
    int target_seq_len, int src_seq_len, int query_heads, int kv_heads,
    bool is_causal, int query_tiles_per_kv) {
  extern __shared__ unsigned char shared_bytes[];
  half* const shared_q = reinterpret_cast<half*>(shared_bytes);
  half* const shared_k = shared_q + kWmmaQueriesPerBlock * HeadDim;
  half* const shared_v = shared_k + kWmmaKeyTile * HeadDim;
  float* const shared_scores = reinterpret_cast<float*>(
      shared_v + kWmmaKeyTile * HeadDim);

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int blocks_per_batch = kv_heads * query_tiles_per_kv;
  const int batch = static_cast<int>(blockIdx.x) / blocks_per_batch;
  const int block_in_batch = static_cast<int>(blockIdx.x) -
                             batch * blocks_per_batch;
  const int kv_head = block_in_batch / query_tiles_per_kv;
  const int query_tile = block_in_batch - kv_head * query_tiles_per_kv;
  const int queries_per_kv = query_heads / kv_heads;
  const int rows_per_kv = target_seq_len * queries_per_kv;

  const int q_values = kWmmaQueriesPerBlock * HeadDim;
  for (int item = threadIdx.x; item < q_values; item += kWmmaBlockSize) {
    const int local_query = item / HeadDim;
    const int dim = item - local_query * HeadDim;
    const int query_in_kv =
        query_tile * kWmmaQueriesPerBlock + local_query;
    if (query_in_kv < rows_per_kv) {
      const int query_pos = query_in_kv / queries_per_kv;
      const int query_head =
          kv_head * queries_per_kv + query_in_kv % queries_per_kv;
      const size_t q_offset =
          ((static_cast<size_t>(batch) * target_seq_len + query_pos) *
               query_heads +
           query_head) * HeadDim;
      shared_q[item] = q[q_offset + dim];
    } else {
      shared_q[item] = __float2half_rn(0.0f);
    }
  }
  __syncthreads();

  const int pair_row = lane / 2;
  const int lane_in_pair = lane & 1;
  const int local_query = warp * 16 + pair_row;
  const int query_in_kv = query_tile * kWmmaQueriesPerBlock + local_query;
  const bool active = query_in_kv < rows_per_kv;
  const int query_pos = active ? query_in_kv / queries_per_kv : 0;
  const int query_head =
      active ? kv_head * queries_per_kv + query_in_kv % queries_per_kv : 0;
  const size_t query_row =
      (static_cast<size_t>(batch) * target_seq_len + query_pos) * query_heads +
      query_head;
  const size_t output_offset = query_row * HeadDim;
  const int key_end =
      active ? (is_causal ? min(src_seq_len, query_pos + 1) : src_seq_len) : 0;
  const int last_query_in_kv =
      min(rows_per_kv - 1,
          query_tile * kWmmaQueriesPerBlock + kWmmaQueriesPerBlock - 1);
  const int last_query_pos = last_query_in_kv / queries_per_kv;
  const int cta_key_end =
      is_causal ? min(src_seq_len, last_query_pos + 1) : src_seq_len;

  float output_cache[ValuesPerLane];
#pragma unroll
  for (int item = 0; item < ValuesPerLane; ++item) {
    output_cache[item] = 0.0f;
  }
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;
  const float scale = 1.0f / sqrtf(static_cast<float>(HeadDim));

  for (int key_base = 0; key_base < cta_key_end;
       key_base += kWmmaKeyTile) {
    const int tile_keys = min(kWmmaKeyTile, cta_key_end - key_base);
    const int tile_values = kWmmaKeyTile * HeadDim;
    for (int item = threadIdx.x; item < tile_values;
         item += kWmmaBlockSize) {
      const int key_in_tile = item / HeadDim;
      const int dim = item - key_in_tile * HeadDim;
      if (key_in_tile < tile_keys) {
        const int key_pos = key_base + key_in_tile;
        const size_t kv_offset =
            ((static_cast<size_t>(batch) * src_seq_len + key_pos) * kv_heads +
             kv_head) * HeadDim + dim;
        shared_k[item] = k[kv_offset];
        shared_v[item] = v[kv_offset];
      } else {
        shared_k[item] = __float2half_rn(0.0f);
        shared_v[item] = __float2half_rn(0.0f);
      }
    }
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
        score_fragment;
    nvcuda::wmma::fill_fragment(score_fragment, 0.0f);
#pragma unroll
    for (int dim_base = 0; dim_base < HeadDim; dim_base += 16) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                             nvcuda::wmma::row_major>
          q_fragment;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half,
                             nvcuda::wmma::col_major>
          k_fragment;
      nvcuda::wmma::load_matrix_sync(
          q_fragment,
          shared_q + static_cast<size_t>(warp) * 16 * HeadDim + dim_base,
          HeadDim);
      nvcuda::wmma::load_matrix_sync(k_fragment, shared_k + dim_base,
                                     HeadDim);
      nvcuda::wmma::mma_sync(score_fragment, q_fragment, k_fragment,
                             score_fragment);
    }
    nvcuda::wmma::store_matrix_sync(
        shared_scores + static_cast<size_t>(warp) * 16 * 16,
        score_fragment, 16, nvcuda::wmma::mem_row_major);
    __syncthreads();

    if (active && key_base < key_end) {
      const int row_key_count = min(tile_keys, key_end - key_base);
      const float* const row_scores =
          shared_scores + static_cast<size_t>(local_query) * 16;
      float tile_max = -CUDART_INF_F;
      for (int key_in_tile = 0; key_in_tile < row_key_count;
           ++key_in_tile) {
        tile_max = fmaxf(tile_max, row_scores[key_in_tile] * scale);
      }
      const float new_max = fmaxf(running_max, tile_max);
      const float alpha = running_max == -CUDART_INF_F
                              ? 0.0f
                              : expf(running_max - new_max);
      float tile_sum = 0.0f;
      for (int key_in_tile = 0; key_in_tile < row_key_count;
           ++key_in_tile) {
        tile_sum += expf(row_scores[key_in_tile] * scale - new_max);
      }
      running_sum = running_sum * alpha + tile_sum;
      running_max = new_max;
#pragma unroll
      for (int item = 0; item < ValuesPerLane; ++item) {
        output_cache[item] *= alpha;
      }

      for (int key_in_tile = 0; key_in_tile < row_key_count;
           ++key_in_tile) {
        const float probability =
            expf(row_scores[key_in_tile] * scale - new_max);
#pragma unroll
        for (int item = 0; item < ValuesPerLane; ++item) {
          const int dim = lane_in_pair + item * 2;
          output_cache[item] =
              fmaf(probability,
                   __half2float(shared_v[key_in_tile * HeadDim + dim]),
                   output_cache[item]);
        }
      }
    }
    __syncthreads();
  }

  if (active) {
    const float inverse_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
#pragma unroll
    for (int item = 0; item < ValuesPerLane; ++item) {
      const int dim = lane_in_pair + item * 2;
      output[output_offset + dim] =
          __float2half_rn(output_cache[item] * inverse_sum);
    }
  }
}

bool launchFlashHalfWmma(const half* q, const half* k, const half* v,
                         half* output, int batch_size, int target_seq_len,
                         int src_seq_len, int query_heads, int kv_heads,
                         int head_dim, bool is_causal) {
  const int queries_per_kv = query_heads / kv_heads;
  const int rows_per_kv = target_seq_len * queries_per_kv;
  if ((head_dim != 32 && head_dim != 64) || src_seq_len < 64 ||
      rows_per_kv < 16) {
    return false;
  }
  const int query_tiles_per_kv =
      (rows_per_kv + kWmmaQueriesPerBlock - 1) / kWmmaQueriesPerBlock;
  const unsigned int blocks = static_cast<unsigned int>(
      batch_size * kv_heads * query_tiles_per_kv);
  const size_t shared_half_values =
      static_cast<size_t>(kWmmaQueriesPerBlock + 2 * kWmmaKeyTile) *
      head_dim;
  const size_t shared_bytes = shared_half_values * sizeof(half) +
                              kWmmaQueriesPerBlock * kWmmaKeyTile *
                                  sizeof(float);
  if (head_dim == 32) {
    flashAttentionHalfWmmaKernel<32, 16>
        <<<blocks, kWmmaBlockSize, shared_bytes>>>(
            q, k, v, output, target_seq_len, src_seq_len, query_heads,
            kv_heads, is_causal, query_tiles_per_kv);
  } else {
    flashAttentionHalfWmmaKernel<64, 32>
        <<<blocks, kWmmaBlockSize, shared_bytes>>>(
            q, k, v, output, target_seq_len, src_seq_len, query_heads,
            kv_heads, is_causal, query_tiles_per_kv);
  }
  return true;
}

__global__ void flashAttentionFloatThreePassKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ output,
    size_t query_rows, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  const size_t query_row = static_cast<size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
  if (query_row >= query_rows) {
    return;
  }

  const size_t rows_per_batch =
      static_cast<size_t>(target_seq_len) * query_heads;
  const int batch = static_cast<int>(query_row / rows_per_batch);
  const size_t row_in_batch = query_row -
      static_cast<size_t>(batch) * rows_per_batch;
  const int query_pos = static_cast<int>(row_in_batch / query_heads);
  const int query_head = static_cast<int>(row_in_batch % query_heads);
  const int kv_head = query_head / (query_heads / kv_heads);
  const size_t q_offset = query_row * head_dim;
  const int key_end = is_causal ? min(src_seq_len, query_pos + 1)
                                : src_seq_len;

  float q_cache[kMaxAttentionHeadDim];
  float output_cache[kMaxAttentionHeadDim];
  for (int dim = 0; dim < head_dim; ++dim) {
    q_cache[dim] = q[q_offset + dim];
    output_cache[dim] = 0.0f;
  }

  float root;
  const float dimension = static_cast<float>(head_dim);
  asm("sqrt.rn.f32 %0, %1;" : "=f"(root) : "f"(dimension));
  float scale;
  asm("rcp.rn.f32 %0, %1;" : "=f"(scale) : "f"(root));

  float row_max = -CUDART_INF_F;
  for (int key_pos = 0; key_pos < key_end; ++key_pos) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + key_pos) * kv_heads +
         kv_head) * head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot = fmaf(q_cache[dim], k[kv_offset + dim], dot);
    }
    row_max = fmaxf(row_max, dot * scale);
  }

  float row_sum = 0.0f;
  for (int key_pos = 0; key_pos < key_end; ++key_pos) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + key_pos) * kv_heads +
         kv_head) * head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot = fmaf(q_cache[dim], k[kv_offset + dim], dot);
    }
    row_sum = fmaf(expf(dot * scale - row_max), 1.0f, row_sum);
  }

  float inverse_sum;
  asm("rcp.rn.f32 %0, %1;" : "=f"(inverse_sum) : "f"(row_sum));
  for (int key_pos = 0; key_pos < key_end; ++key_pos) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + key_pos) * kv_heads +
         kv_head) * head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot = fmaf(q_cache[dim], k[kv_offset + dim], dot);
    }
    const float probability = expf(dot * scale - row_max) * inverse_sum;
    for (int dim = 0; dim < head_dim; ++dim) {
      output_cache[dim] =
          fmaf(probability, v[kv_offset + dim], output_cache[dim]);
    }
  }

  for (int dim = 0; dim < head_dim; ++dim) {
    output[q_offset + dim] = output_cache[dim];
  }
}

__global__ void flashAttentionFloatD32Kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ output,
    size_t query_rows, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, bool is_causal) {
  constexpr int kHeadDim = 32;
  const size_t query_row = static_cast<size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
  if (query_row >= query_rows) {
    return;
  }

  const size_t rows_per_batch =
      static_cast<size_t>(target_seq_len) * query_heads;
  const int batch = static_cast<int>(query_row / rows_per_batch);
  const size_t row_in_batch = query_row -
      static_cast<size_t>(batch) * rows_per_batch;
  const int query_pos = static_cast<int>(row_in_batch / query_heads);
  const int query_head = static_cast<int>(row_in_batch % query_heads);
  const int kv_head = query_head / (query_heads / kv_heads);
  const size_t q_offset = query_row * kHeadDim;
  const int key_end = is_causal ? min(src_seq_len, query_pos + 1)
                                : src_seq_len;

  float q_cache[kHeadDim];
  float output_cache[kHeadDim];
#pragma unroll
  for (int dim = 0; dim < kHeadDim; ++dim) {
    q_cache[dim] = q[q_offset + dim];
    output_cache[dim] = 0.0f;
  }

  float root;
  asm("sqrt.rn.f32 %0, %1;" : "=f"(root) : "f"(32.0f));
  float scale;
  asm("rcp.rn.f32 %0, %1;" : "=f"(scale) : "f"(root));

  float row_max = -CUDART_INF_F;
  for (int key_pos = 0; key_pos < key_end; ++key_pos) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + key_pos) * kv_heads +
         kv_head) * kHeadDim;
    float dot = 0.0f;
#pragma unroll
    for (int dim = 0; dim < kHeadDim; ++dim) {
      dot = fmaf(q_cache[dim], k[kv_offset + dim], dot);
    }
    row_max = fmaxf(row_max, dot * scale);
  }

  float row_sum = 0.0f;
  for (int key_pos = 0; key_pos < key_end; ++key_pos) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + key_pos) * kv_heads +
         kv_head) * kHeadDim;
    float dot = 0.0f;
#pragma unroll
    for (int dim = 0; dim < kHeadDim; ++dim) {
      dot = fmaf(q_cache[dim], k[kv_offset + dim], dot);
    }
    row_sum = fmaf(expf(dot * scale - row_max), 1.0f, row_sum);
  }

  float inverse_sum;
  asm("rcp.rn.f32 %0, %1;" : "=f"(inverse_sum) : "f"(row_sum));
  for (int key_pos = 0; key_pos < key_end; ++key_pos) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + key_pos) * kv_heads +
         kv_head) * kHeadDim;
    float dot = 0.0f;
#pragma unroll
    for (int dim = 0; dim < kHeadDim; ++dim) {
      dot = fmaf(q_cache[dim], k[kv_offset + dim], dot);
    }
    const float probability = expf(dot * scale - row_max) * inverse_sum;
#pragma unroll
    for (int dim = 0; dim < kHeadDim; ++dim) {
      output_cache[dim] =
          fmaf(probability, v[kv_offset + dim], output_cache[dim]);
    }
  }

#pragma unroll
  for (int dim = 0; dim < kHeadDim; ++dim) {
    output[q_offset + dim] = output_cache[dim];
  }
}

void launchFlashAttention(const float* q, const float* k, const float* v,
                          float* output, int batch_size,
                          int target_seq_len, int src_seq_len,
                          int query_heads, int kv_heads, int head_dim,
                          bool is_causal) {
  const size_t query_rows = static_cast<size_t>(batch_size) * target_seq_len *
                            query_heads;
  if (head_dim == 32) {
    constexpr int kD32BlockSize = 128;
    const unsigned int blocks = static_cast<unsigned int>(
        (query_rows + kD32BlockSize - 1) / kD32BlockSize);
    flashAttentionFloatD32Kernel<<<blocks, kD32BlockSize>>>(
        q, k, v, output, query_rows, target_seq_len, src_seq_len,
        query_heads, kv_heads, is_causal);
    return;
  }
  if (launchFlashSharedKv(q, k, v, output, batch_size, target_seq_len,
                          src_seq_len, query_heads, kv_heads, head_dim,
                          is_causal)) {
    return;
  }
  constexpr int kThreePassBlockSize = 256;
  const unsigned int blocks = static_cast<unsigned int>(
      (query_rows + kThreePassBlockSize - 1) / kThreePassBlockSize);
  flashAttentionFloatThreePassKernel<<<blocks, kThreePassBlockSize>>>(
      q, k, v, output, query_rows, target_seq_len, src_seq_len, query_heads,
      kv_heads, head_dim, is_causal);
}

void launchFlashAttention(const half* q, const half* k, const half* v,
                          half* output, int batch_size, int target_seq_len,
                          int src_seq_len, int query_heads, int kv_heads,
                          int head_dim, bool is_causal) {
  if (launchFlashHalfWmma(q, k, v, output, batch_size, target_seq_len,
                          src_seq_len, query_heads, kv_heads, head_dim,
                          is_causal)) {
    return;
  }
  const int rows_per_kv = target_seq_len * (query_heads / kv_heads);
  if (src_seq_len <= 256 && rows_per_kv >= 8 &&
      launchFlashSharedKv(q, k, v, output, batch_size, target_seq_len,
                          src_seq_len, query_heads, kv_heads, head_dim,
                          is_causal)) {
    return;
  }
  launchFlashWarpOnline(q, k, v, output, batch_size, target_seq_len,
                        src_seq_len, query_heads, kv_heads, head_dim,
                        is_causal);
}

#endif

size_t checkedMultiply(size_t lhs, size_t rhs, const char* label) {
  if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
    throw std::overflow_error(label);
  }
  return lhs * rhs;
}

size_t checkedAdd(size_t lhs, size_t rhs, const char* label) {
  if (lhs > std::numeric_limits<size_t>::max() - rhs) {
    throw std::overflow_error(label);
  }
  return lhs + rhs;
}

size_t alignDeviceOffset(size_t value) {
  return (value + kDeviceAlignment - 1) & ~(kDeviceAlignment - 1);
}

}  // namespace

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  if (rows == 0 || hidden_dim == 0) {
    h_output.clear();
    return;
  }
  if (rows > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error("RMSNorm row count exceeds the grid limit");
  }

  const size_t element_count =
      checkedMultiply(rows, hidden_dim, "RMSNorm element count overflow");
  if (h_input.size() < element_count || h_weight.size() < hidden_dim) {
    throw std::invalid_argument("RMSNorm input vector is smaller than its shape");
  }
  if (h_output.size() != element_count) {
    h_output.resize(element_count);
  }

  const size_t input_bytes =
      checkedMultiply(element_count, sizeof(T), "RMSNorm input size overflow");
  const size_t weight_bytes =
      checkedMultiply(hidden_dim, sizeof(T), "RMSNorm weight size overflow");
  const size_t output_bytes = element_count * sizeof(T);
  const size_t weight_offset = alignDeviceOffset(input_bytes);
  const size_t output_offset = alignDeviceOffset(weight_offset + weight_bytes);
  const size_t allocation_bytes = output_offset + output_bytes;

  unsigned char* storage = nullptr;
  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&storage), allocation_bytes));
  T* d_input = reinterpret_cast<T*>(storage);
  T* d_weight = reinterpret_cast<T*>(storage + weight_offset);
  T* d_output = reinterpret_cast<T*>(storage + output_offset);

  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes,
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_bytes,
                           cudaMemcpyHostToDevice));

#if defined(RMSNORM_FORCE_SCALAR)
  launchScalar(d_input, d_weight, d_output, rows, hidden_dim, eps);
#else
  if (!launchVectorized(d_input, d_weight, d_output, rows, hidden_dim, eps)) {
    launchScalar(d_input, d_weight, d_output, rows, hidden_dim, eps);
  }
#endif
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output, output_bytes,
                           cudaMemcpyDeviceToHost));
  RUNTIME_CHECK(cudaFree(storage));
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  if (batch_size < 0 || target_seq_len < 0 || src_seq_len < 0 ||
      query_heads < 0 || kv_heads < 0 || head_dim < 0) {
    throw std::invalid_argument("attention dimensions must be non-negative");
  }
  if (batch_size == 0 || target_seq_len == 0 || query_heads == 0 ||
      head_dim == 0) {
    h_o.clear();
    return;
  }
  if (kv_heads == 0 || query_heads % kv_heads != 0) {
    throw std::invalid_argument(
        "query_heads must be divisible by non-zero kv_heads");
  }
  if (head_dim > kMaxAttentionHeadDim) {
    throw std::invalid_argument("head_dim exceeds supported maximum of 256");
  }

  size_t query_count = checkedMultiply(static_cast<size_t>(batch_size),
                                       static_cast<size_t>(target_seq_len),
                                       "query element count overflow");
  query_count = checkedMultiply(query_count, static_cast<size_t>(query_heads),
                                "query element count overflow");
  query_count = checkedMultiply(query_count, static_cast<size_t>(head_dim),
                                "query element count overflow");
  size_t kv_count = checkedMultiply(static_cast<size_t>(batch_size),
                                    static_cast<size_t>(src_seq_len),
                                    "key/value element count overflow");
  kv_count = checkedMultiply(kv_count, static_cast<size_t>(kv_heads),
                             "key/value element count overflow");
  kv_count = checkedMultiply(kv_count, static_cast<size_t>(head_dim),
                             "key/value element count overflow");
  if (h_q.size() < query_count || h_k.size() < kv_count ||
      h_v.size() < kv_count) {
    throw std::invalid_argument("attention input vector is smaller than its shape");
  }
  if (h_o.size() != query_count) {
    h_o.resize(query_count);
  }
  if (src_seq_len == 0) {
    std::fill(h_o.begin(), h_o.end(), T{});
    return;
  }

  const size_t query_bytes = checkedMultiply(query_count, sizeof(T),
                                             "query byte size overflow");
  const size_t kv_bytes = checkedMultiply(kv_count, sizeof(T),
                                          "key/value byte size overflow");
  const size_t key_offset = alignDeviceOffset(query_bytes);
  const size_t value_offset = alignDeviceOffset(
      checkedAdd(key_offset, kv_bytes, "attention allocation overflow"));
  const size_t output_offset = alignDeviceOffset(
      checkedAdd(value_offset, kv_bytes, "attention allocation overflow"));
  const size_t allocation_bytes = checkedAdd(
      output_offset, query_bytes, "attention allocation overflow");

  unsigned char* storage = nullptr;
  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&storage), allocation_bytes));
  T* d_q = reinterpret_cast<T*>(storage);
  T* d_k = reinterpret_cast<T*>(storage + key_offset);
  T* d_v = reinterpret_cast<T*>(storage + value_offset);
  T* d_o = reinterpret_cast<T*>(storage + output_offset);
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), query_bytes,
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_bytes,
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_bytes,
                           cudaMemcpyHostToDevice));

  launchFlashAttention(d_q, d_k, d_v, d_o, batch_size, target_seq_len,
                        src_seq_len, query_heads, kv_heads, head_dim,
                        is_causal);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, query_bytes,
                           cudaMemcpyDeviceToHost));
  RUNTIME_CHECK(cudaFree(storage));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
