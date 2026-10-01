// Explicit native opt-in at the maps/expansion boundary. No Python registration.
#pragma once
#include "m1_maps_expand.cuh"
#include "moe_kernels.h"
#include <cuda.h>
#include <array>
#include <cstring>

namespace megartx::experimental {
struct InstalledPreparationCall {
  M1Buffers buffers;
  tensorrt_llm::kernels::cutlass_kernels::QuantParams const& quant;
  cudaStream_t stream;
  int64_t tokens = 1, hidden = kHidden;
  int experts = kExperts, top_k = kTopK, start_expert = 0, end_expert = kExperts;
  int intermediate = 704, tp_size = 1, ep_size = 1;
  bool swizzled_input_sf = true, enable_pdl = false;
  bool correction_runner_active = false, min_latency = false, lora = false;
  bool groupwise = false, all_to_all = false;
};
enum class PreparationBackend { Stock, Fused };

inline bool device_views_valid(M1Buffers b) {
  // Read actual allocation extents; a declared capacity is never a receipt.
  std::array<void const*,10> ptrs{b.ids,b.weight_bits,b.aq,b.sf,b.slot_to_sorted,
    b.sorted_to_slot,b.offsets,b.expanded_aq,b.permuted_weight_bits,b.expanded_sf};
  constexpr std::array<std::size_t,10> sizes{32,32,kAQBytes,128*kSFBlocks,32,32,
    129*8,kTopK*kAQBytes,32,kSFBytes};
  constexpr std::array<int,10> alignments{4,4,16,4,4,4,8,16,4,4};
  int device; if(cudaGetDevice(&device)!=cudaSuccess)return false;
  for(std::size_t i=0;i<ptrs.size();++i) {
    if(!aligned(ptrs[i],alignments[i]))return false;
    cudaPointerAttributes attr{};
    if(cudaPointerGetAttributes(&attr,ptrs[i])!=cudaSuccess ||
       attr.type!=cudaMemoryTypeDevice || attr.device!=device)return false;
    CUdeviceptr base=0;std::size_t extent=0;
    auto p=reinterpret_cast<CUdeviceptr>(ptrs[i]);
    if(cuMemGetAddressRange(&base,&extent,p)!=CUDA_SUCCESS || p<base ||
       p-base>extent || sizes[i]>extent-(p-base))return false;
    for(std::size_t j=0;j<i;++j) {
      auto q=reinterpret_cast<CUdeviceptr>(ptrs[j]);
      if(p<q+sizes[j] && q<p+sizes[i])return false;
    }
  }
  return true;
}

inline bool candidate_eligible(InstalledPreparationCall const& c, bool opt_in) {
  if(!opt_in || c.tokens!=1 || c.hidden!=kHidden || c.experts!=kExperts ||
     c.top_k!=kTopK || c.start_expert!=0 || c.end_expert!=kExperts ||
     c.intermediate!=704 || c.tp_size!=1 || c.ep_size!=1 ||
     !c.swizzled_input_sf || c.enable_pdl || !c.correction_runner_active ||
     c.min_latency || c.lora || c.groupwise || c.all_to_all ||
     c.quant.fp4.fc1.use_per_expert_act_scale || !c.quant.fp4.fc1.weight_block_scale)
    return false;
  cudaStreamCaptureStatus capture;
  if(cudaStreamIsCapturing(c.stream,&capture)!=cudaSuccess || capture!=cudaStreamCaptureStatusNone)
    return false;  // No graph qualification, no D2H or synchronization during capture.
  CUcontext current=nullptr,stream_context=nullptr;
  if(!c.stream || cuCtxGetCurrent(&current)!=CUDA_SUCCESS ||
     cuStreamGetCtx(reinterpret_cast<CUstream>(c.stream),&stream_context)!=CUDA_SUCCESS ||
     !current || current!=stream_context)return false;
  if(!device_views_valid(c.buffers))return false;
  std::array<int,8> ids;
  auto error=cudaMemcpyAsync(ids.data(),c.buffers.ids,32,cudaMemcpyDeviceToHost,c.stream);
  if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
  error=cudaStreamSynchronize(c.stream);
  if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
  for(int i=0;i<8;++i) {
    if(ids[i]<0 || ids[i]>=128)return false;
    for(int j=0;j<i;++j)if(ids[i]==ids[j])return false;
  }
  return true;
}

template <class Incumbent>
inline PreparationBackend prepare(InstalledPreparationCall const& c, Incumbent&& incumbent,
                                  bool opt_in=false) {
  auto b=c.buffers;
  if(candidate_eligible(c,opt_in)) {
    auto error=launch(b,c.stream,true);
    // A launch/runtime error is not an unsupported lane. Do not overwrite partially
    // published output by attempting fallback after a candidate launch.
    if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
    return PreparationBackend::Fused;
  }
  incumbent();  // Preserve the caller's complete stock branch, including its own fallback.
  return PreparationBackend::Stock;
}
}  // namespace megartx::experimental
