// Experimental standalone byte preparation. No TMA, quantization, GEMM or graph hook.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace megartx::experimental {
constexpr int kHidden = 2816, kExperts = 128, kTopK = 8;
constexpr int kAQBytes = kHidden / 2, kSFBlocks = kHidden / 16;
constexpr int kSFBytes = 16384 * kSFBlocks;
struct M1Buffers {
  int const* ids;
  std::uint32_t const* weight_bits;
  unsigned char const* aq;
  unsigned char const* sf;
  int* slot_to_sorted;
  int* sorted_to_slot;
  std::int64_t* offsets;
  unsigned char* expanded_aq;
  std::uint32_t* permuted_weight_bits;
  unsigned char* expanded_sf;
};

__device__ inline bool valid_ids(int const* ids) {
  for (int i = 0; i < kTopK; ++i) {
    if (ids[i] < 0 || ids[i] >= kExperts) return false;
    for (int j = 0; j < i; ++j) if (ids[i] == ids[j]) return false;
  }
  return true;
}
__device__ inline void maps(M1Buffers b, int* sorted) {
  int t = threadIdx.x;
  if (t < kTopK) {
    int rank = 0;
    for (int j = 0; j < kTopK; ++j) rank += b.ids[j] < b.ids[t];
    b.slot_to_sorted[t] = rank;
    b.sorted_to_slot[rank] = t;
    if (sorted) sorted[rank] = t;
  }
  for (int e = t; e <= kExperts; e += blockDim.x) {
    std::int64_t count = 0;
    for (int j = 0; j < kTopK; ++j) count += b.ids[j] < e;
    b.offsets[e] = count;
  }
}
__device__ inline void scale_word(M1Buffers b, int rank, int group, int slot) {
  int expert = b.ids[slot];
  // For distinct M1 routes the selected expert's prefix equals its sorted rank.
  int base = ((rank + 127 * expert + 127) / 128) * 128 * kSFBlocks;
  int column = group * 512;  // Four row-zero bytes in each 128x4 SF atom.
  *reinterpret_cast<std::uint32_t*>(b.expanded_sf + base + column) =
      *reinterpret_cast<std::uint32_t const*>(b.sf + column);
}
__global__ void m1_maps_expand(M1Buffers b) {
  __shared__ int sorted[kTopK];
  __shared__ bool valid;
  if (threadIdx.x == 0) valid = valid_ids(b.ids);
  __syncthreads();
  if (!valid) return;  // Invalid routes leave every output untouched.
  maps(b, sorted);
  __syncthreads();
  auto src = reinterpret_cast<uint4 const*>(b.aq);
  auto dst = reinterpret_cast<uint4*>(b.expanded_aq);
  for (int v = threadIdx.x; v < kTopK * (kAQBytes / 16); v += blockDim.x)
    dst[v] = src[v % (kAQBytes / 16)];
  if (threadIdx.x < kTopK)
    b.permuted_weight_bits[threadIdx.x] = b.weight_bits[sorted[threadIdx.x]];
  for (int task = threadIdx.x; task < kTopK * (kSFBlocks / 4); task += blockDim.x) {
    int rank = task / (kSFBlocks / 4), group = task % (kSFBlocks / 4);
    scale_word(b, rank, group, sorted[rank]);
  }
}

// Same byte contract, two launches: a development timing control, not FlashInfer.
__global__ void m1_maps_control(M1Buffers b) {
  __shared__ bool valid;
  if (threadIdx.x == 0) valid = valid_ids(b.ids);
  __syncthreads();
  if (valid) maps(b, nullptr);
}
__global__ void m1_expand_control(M1Buffers b) {
  __shared__ bool valid;
  if (threadIdx.x == 0) valid = valid_ids(b.ids);
  __syncthreads();
  if (!valid) return;
  int rank = blockIdx.x, slot = b.sorted_to_slot[rank];
  auto src = reinterpret_cast<uint4 const*>(b.aq);
  auto dst = reinterpret_cast<uint4*>(b.expanded_aq) + rank * (kAQBytes / 16);
  for (int v = threadIdx.x; v < kAQBytes / 16; v += blockDim.x) dst[v] = src[v];
  if (threadIdx.x == 0) b.permuted_weight_bits[rank] = b.weight_bits[slot];
  for (int group = threadIdx.x; group < kSFBlocks / 4; group += blockDim.x)
    scale_word(b, rank, group, slot);
}
inline bool aligned(void const* ptr, int n) {
  return ptr && reinterpret_cast<std::uintptr_t>(ptr) % n == 0;
}
inline cudaError_t launch(M1Buffers b, cudaStream_t stream, bool fused,
                         int m = 1, int h = kHidden, int e = kExperts, int top_k = kTopK) {
  if (m != 1 || h != kHidden || e != kExperts || top_k != kTopK ||
      !aligned(b.ids, 4) || !aligned(b.weight_bits, 4) || !aligned(b.aq, 16) ||
      !aligned(b.sf, 4) || !aligned(b.slot_to_sorted, 4) || !aligned(b.sorted_to_slot, 4) ||
      !aligned(b.offsets, 8) || !aligned(b.expanded_aq, 16) ||
      !aligned(b.permuted_weight_bits, 4) || !aligned(b.expanded_sf, 4))
    return cudaErrorInvalidValue;
  if (fused) m1_maps_expand<<<1, 256, 0, stream>>>(b);
  else {
    m1_maps_control<<<1, 32, 0, stream>>>(b);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    m1_expand_control<<<kTopK, 256, 0, stream>>>(b);
  }
  return cudaGetLastError();
}
}  // namespace megartx::experimental
