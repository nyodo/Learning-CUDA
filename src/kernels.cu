#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <cuda_fp16.h>

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

template <int BlockSize>
__device__ __forceinline__ float blockReduceSum(float value) {
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
    return;
  }

  const size_t element_count = rows * hidden_dim;
  const size_t input_bytes = element_count * sizeof(T);
  const size_t weight_bytes = hidden_dim * sizeof(T);
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
  // TODO: Implement the flash attention function
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
