// CPU fault injection around an unchanged adapter header; no native/GPU claim.
#include "mock_cuda.hpp"
#include "m1_installed_preparation.cuh"
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

using namespace megartx::experimental;
struct Range { CUdeviceptr begin; std::size_t bytes; };
struct Mock {
  std::string failing_api;
  int failing_call=1, error_code=719;
  bool diagnostic_failure=false, capture=false, wrong_context=false, host_pointer=false, short_range=false;
  int incumbent_calls=0, output=0;
  std::map<std::string,int> calls;
  std::vector<Range> ranges;
  bool fail(char const* api) { return ++calls[api]==failing_call && failing_api==api; }
} mock;

char const* cudaGetErrorString(cudaError_t) { return "mock runtime failure"; }
cudaError_t cudaStreamIsCapturing(cudaStream_t,cudaStreamCaptureStatus* status) {
  if(mock.fail("cudaStreamIsCapturing"))return cudaError_t(mock.error_code);
  *status=mock.capture?cudaStreamCaptureStatusActive:cudaStreamCaptureStatusNone;return cudaSuccess;
}
cudaError_t cudaGetDevice(int* device) {
  if(mock.fail("cudaGetDevice"))return cudaError_t(mock.error_code);
  *device=0;return cudaSuccess;
}
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attr,void const*) {
  if(mock.fail("cudaPointerGetAttributes"))return cudaError_t(mock.error_code);
  attr->type=mock.host_pointer?cudaMemoryTypeHost:cudaMemoryTypeDevice;attr->device=0;return cudaSuccess;
}
CUresult cuGetErrorString(CUresult,char const** message) {
  if(mock.diagnostic_failure)return CUDA_ERROR_INVALID_VALUE;
  *message="mock driver failure";return CUDA_SUCCESS;
}
CUresult cuCtxGetCurrent(CUcontext* context) {
  if(mock.fail("cuCtxGetCurrent"))return CUresult(mock.error_code);
  *context=reinterpret_cast<CUcontext>(2);return CUDA_SUCCESS;
}
CUresult cuStreamGetCtx(CUstream,CUcontext* context) {
  if(mock.fail("cuStreamGetCtx"))return CUresult(mock.error_code);
  *context=reinterpret_cast<CUcontext>(mock.wrong_context?3:2);return CUDA_SUCCESS;
}
CUresult cuMemGetAddressRange(CUdeviceptr* base,std::size_t* extent,CUdeviceptr ptr) {
  if(mock.fail("cuMemGetAddressRange"))return CUresult(mock.error_code);
  for(auto r:mock.ranges)if(ptr>=r.begin && ptr<r.begin+r.bytes) {
    *base=r.begin;*extent=mock.short_range?0:r.bytes;return CUDA_SUCCESS;
  }
  return CUDA_ERROR_INVALID_VALUE;
}
cudaError_t cudaMemcpyAsync(void* dest,void const* src,std::size_t size,cudaMemcpyKind,cudaStream_t) {
  if(mock.fail("cudaMemcpyAsync"))return cudaError_t(mock.error_code);
  std::memcpy(dest,src,size);return cudaSuccess;
}
cudaError_t cudaStreamSynchronize(cudaStream_t) {
  if(mock.fail("cudaStreamSynchronize"))return cudaError_t(mock.error_code);
  return cudaSuccess;
}
namespace megartx::experimental {
cudaError_t launch(M1Buffers,cudaStream_t,bool) {
  // Model a submission that could publish output before its error is observed.
  mock.output=3;
  return mock.fail("launch")?cudaError_t(mock.error_code):cudaSuccess;
}
}

struct Fixture {
  alignas(16) std::array<int,8> ids{127,0,82,42,126,7,89,12};
  alignas(16) std::array<uint32_t,8> weights{},rank{},sorted{},permuted_weights{};
  alignas(16) std::array<unsigned char,1408> aq{};
  alignas(16) std::array<unsigned char,22528> sf{};
  alignas(16) std::array<int64_t,129> offsets{};
  alignas(16) std::array<unsigned char,11264> expanded_aq{};
  alignas(16) std::array<unsigned char,2883584> expanded_sf{};
  M1Buffers buffers() {
    return {ids.data(),weights.data(),aq.data(),sf.data(),reinterpret_cast<int*>(rank.data()),
      reinterpret_cast<int*>(sorted.data()),offsets.data(),expanded_aq.data(),permuted_weights.data(),expanded_sf.data()};
  }
  void register_ranges() {
    auto b=buffers();
    std::array<void const*,10> ptrs{b.ids,b.weight_bits,b.aq,b.sf,b.slot_to_sorted,b.sorted_to_slot,
      b.offsets,b.expanded_aq,b.permuted_weight_bits,b.expanded_sf};
    constexpr std::array<std::size_t,10> sizes{32,32,1408,22528,32,32,1032,11264,32,2883584};
    for(std::size_t i=0;i<ptrs.size();++i)mock.ranges.push_back({reinterpret_cast<CUdeviceptr>(ptrs[i]),sizes[i]});
  }
};

int main(int argc,char** argv) {
  try {
    if(argc<2)throw std::runtime_error("test case required");
    bool lean=std::string(argv[1])=="lean";
    if(lean) { --argc;++argv; }
    auto fixture=std::make_unique<Fixture>();fixture->register_ranges();
    tensorrt_llm::kernels::cutlass_kernels::QuantParams quant{};
    quant.fp4.fc1.weight_block_scale=fixture->sf.data();
    InstalledPreparationCall call{fixture->buffers(),quant,reinterpret_cast<cudaStream_t>(1)};
    call.correction_runner_active=true;
    std::string test=argv[1];bool opt_in=true,expected_fault=false;
    if(test=="fault") {
      if(argc<4)throw std::runtime_error("fault API and occurrence required");
      mock.failing_api=argv[2];mock.failing_call=std::stoi(argv[3]);expected_fault=true;
      if(argc>4 && std::string(argv[4])=="invalid_value")mock.error_code=1;
      if(argc>4 && std::string(argv[4])=="diagnostic_failure")mock.diagnostic_failure=true;
    } else if(test=="disabled")opt_in=false;
    else if(test=="geometry")call.tokens=2;
    else if(test=="capture")mock.capture=true;
    else if(test=="context")mock.wrong_context=true;
    else if(test=="host_pointer")mock.host_pointer=true;
    else if(test=="short_range")mock.short_range=true;
    else if(test=="alias")call.buffers.sorted_to_slot=call.buffers.slot_to_sorted;
    else if(test=="duplicate")fixture->ids[0]=fixture->ids[1];
    else if(test=="per_expert_unverified")quant.fp4.fc1.use_per_expert_act_scale=true;
    else if(test=="per_expert_copy") {
      quant.fp4.fc1.use_per_expert_act_scale=true;
      call.fc1_input_lane=Fc1InputLane::InstalledPrequantizedFP4;
    }
    else if(test=="stock_supported")opt_in=false;
    else if(test=="changed_after_previous") {
      if(!lean)throw std::runtime_error("stale decision test requires lean helper");
      auto previous=prepare_capture_free(call,[]{throw std::runtime_error("unexpected first fallback");},true);
      if(!previous.qualified || previous.backend!=PreparationBackend::Fused)
        throw std::runtime_error("first dynamic check was not qualified");
      fixture->ids[0]=fixture->ids[1];mock.output=0;
      opt_in=previous.qualified;  // A previous result is only a request, never admission.
    }
    else if(test=="supported"){}
    else if(test=="incumbent_failure")opt_in=false;
    else throw std::runtime_error("unknown control");
    auto incumbent=[&] {
      if(mock.output)throw std::runtime_error("output changed before incumbent callback");
      ++mock.incumbent_calls;mock.output=2;
      if(test=="incumbent_failure")throw std::runtime_error("incumbent failure");
    };
    bool threw=false,qualified=false;PreparationBackend backend=PreparationBackend::Stock;
    try {
      if(lean) {
        auto decision=prepare_capture_free(call,incumbent,opt_in);
        qualified=decision.qualified;backend=decision.backend;
      } else backend=prepare(call,incumbent,opt_in);
    }
    catch(std::runtime_error const& e) {
      threw=true;
      auto operation=mock.failing_api=="launch"?"candidate launch":mock.failing_api;
      if(expected_fault && std::string(e.what()).find(operation+":")==std::string::npos)
        throw std::runtime_error("lost primary operation in diagnostic");
      if(!expected_fault && test!="incumbent_failure")throw;
    }
    if(expected_fault) {
      auto launches=mock.calls["launch"];
      if(!threw || mock.incumbent_calls ||
         (mock.failing_api=="launch" ? launches!=1 || mock.output!=3 : launches!=0 || mock.output!=0))
        throw std::runtime_error("failure invoked fallback or changed preflight output");
    } else if(test=="supported" || test=="per_expert_copy") {
      if(threw || backend!=PreparationBackend::Fused || mock.incumbent_calls || mock.calls["launch"]!=1 || mock.output!=3)
        throw std::runtime_error("supported control did not launch once");
    } else if(test=="stock_supported") {
      if(threw || backend!=PreparationBackend::Stock || !qualified || mock.incumbent_calls!=1 ||
          mock.calls["launch"] || mock.output!=2)
        throw std::runtime_error("stock path lost fresh descriptor qualification");
    } else if(test=="changed_after_previous") {
      if(threw || qualified || backend!=PreparationBackend::Stock || mock.incumbent_calls!=1 ||
          mock.calls["launch"]!=1 || mock.output!=2 || mock.calls["cudaMemcpyAsync"]!=2 ||
          mock.calls["cudaStreamSynchronize"]!=2 || mock.calls["cudaPointerGetAttributes"]!=20)
        throw std::runtime_error("previous decision bypassed the current dynamic check");
    } else if(test=="incumbent_failure") {
      if(!threw || mock.incumbent_calls!=1 || mock.calls["launch"] || mock.output!=2)
        throw std::runtime_error("incumbent error did not propagate once");
    } else if(threw || backend!=PreparationBackend::Stock || mock.incumbent_calls!=1 || mock.calls["launch"] || mock.output!=2)
      throw std::runtime_error("unsupported control did not invoke incumbent exactly once");
    if(lean && (test=="supported" || test=="per_expert_copy" || test=="stock_supported")) {
      if(!qualified || mock.calls["cudaMemcpyAsync"]!=1 || mock.calls["cudaStreamSynchronize"]!=1 ||
          mock.calls["cudaPointerGetAttributes"]!=10 || mock.calls["cuMemGetAddressRange"]!=10)
        throw std::runtime_error("lean dispatch did not perform exactly one complete fresh check");
    }
    std::cout<<"prepare control flow passed\n";return 0;
  } catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
}
