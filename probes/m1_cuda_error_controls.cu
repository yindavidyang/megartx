// Deliberately fault only this disposable process. Never load a model.
#include "m1_installed_bridge.cuh"
#include "../kernels/m1_installed_preparation.cuh"
#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace mx = megartx::experimental;
namespace stock = tensorrt_llm::kernels::cutlass_kernels;
__global__ void illegal_address(volatile int* pointer) { *pointer = 1; }

std::string quoted(std::string const& text) {
  std::string result="\"";
  for(char c:text) { if(c=='\\' || c=='\"')result+='\\';if(c=='\n')result+="\\n";else result+=c; }
  return result+'\"';
}
[[noreturn]] void finish(std::string const& control,int incumbent,int errors,
                       std::string const& message,int trigger=0,size_t scratch=0,
                       int preparation_calls=0) {
  std::cout << "{\"control\":" << quoted(control) << ",\"incumbent_calls\":" << incumbent
      << ",\"propagated_errors\":" << errors << ",\"observed_error\":" << quoted(message)
      << ",\"trigger_status\":" << trigger
      << ",\"scratch_bytes\":" << scratch
      << ",\"preparation_calls\":" << preparation_calls
      << ",\"device_buffers_reused_after_error\":false,"
         "\"process_exit_without_cuda_cleanup_after_error\":true}" << std::endl;
  // A poisoned context must not run destructors that fence, copy or recycle
  // CUDA buffers. Process termination releases only this child's resources.
  std::_Exit(0);
}

int main(int argc,char** argv) {
  try {
    if(argc!=2)throw std::runtime_error("one fixed control required");
    std::string control=argv[1];
    if(control!="supported" && control!="illegal_address" && control!="driver_no_context")
      throw std::runtime_error("undeclared control");
    mx::require_cuda_success(cudaSetDevice(0),"initial device");
    mx::require_cuda_success(cudaFree(nullptr),"initialize owned context");
    size_t free=0,total=0;mx::require_cuda_success(cudaMemGetInfo(&free,&total),"headroom");
    if(free<(size_t{2}<<30))throw std::runtime_error("2 GiB GPU headroom required");
    if(control=="driver_no_context") {
      mx::require_driver_success(cuCtxSetCurrent(nullptr),"clear only child current context");
      CUdevice device;auto status=cuCtxGetDevice(&device);
      if(status!=CUDA_ERROR_INVALID_CONTEXT)throw std::runtime_error("real driver invalid-context control did not fail");
      try { mx::require_driver_success(status,"real cuCtxGetDevice"); }
      catch(std::exception const& error) { finish(control,0,1,error.what(),int(status)); }
      throw std::runtime_error("driver context error was swallowed");
    }
    cudaStream_t stream{};mx::require_cuda_success(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"owned stream");
    constexpr std::array<size_t,10> sizes{32,32,1408,22528,32,32,1032,11264,32,2883584};
    std::array<void*,10> pointers{};
    size_t scratch=0;
    for(size_t i=0;i<sizes.size();++i) {
      scratch+=sizes[i];if(scratch>(8<<20))throw std::runtime_error("scratch bound");
      mx::require_cuda_success(cudaMalloc(&pointers[i],sizes[i]),"owned allocation");
      mx::require_cuda_success(cudaMemsetAsync(pointers[i],0x38,sizes[i],stream),"initialize child operands");
    }
    std::array<int,8> ids{11,12,13,14,15,16,17,18};
    mx::require_cuda_success(cudaMemcpyAsync(pointers[0],ids.data(),32,cudaMemcpyHostToDevice,stream),"child IDs");
    mx::require_cuda_success(cudaStreamSynchronize(stream),"initial child fence");
    mx::M1Buffers buffers{static_cast<int const*>(pointers[0]),static_cast<uint32_t const*>(pointers[1]),
        static_cast<unsigned char const*>(pointers[2]),static_cast<unsigned char const*>(pointers[3]),
        static_cast<int*>(pointers[4]),static_cast<int*>(pointers[5]),static_cast<int64_t*>(pointers[6]),
        static_cast<unsigned char*>(pointers[7]),static_cast<uint32_t*>(pointers[8]),static_cast<unsigned char*>(pointers[9])};
    stock::QuantParams quant{};quant.fp4.fc1.weight_block_scale=static_cast<unsigned char const*>(pointers[3]);
    mx::InstalledPreparationCall call{buffers,quant,stream};call.correction_runner_active=true;
    call.fc1_input_lane=mx::Fc1InputLane::InstalledPrequantizedFP4;
    int trigger=0;
    if(control=="illegal_address") {
      illegal_address<<<1,1,0,stream>>>(nullptr);
      mx::require_cuda_success(cudaGetLastError(),"intentional child fault launch");
      auto status=cudaStreamSynchronize(stream);trigger=int(status);
      if(status!=cudaErrorIllegalAddress)throw std::runtime_error("child context was not poisoned by real illegal address");
    }
    int incumbent=0;
    try {
      auto backend=mx::prepare(call,[&]{++incumbent;},true);
      if(control!="supported" || backend!=mx::PreparationBackend::Fused || incumbent)
        throw std::runtime_error("real query error fell through to stock or candidate");
      mx::require_cuda_success(cudaStreamSynchronize(stream),"supported candidate fence");
      std::array<int,8> maps{};mx::require_cuda_success(cudaMemcpy(maps.data(),pointers[4],32,cudaMemcpyDeviceToHost),"supported map read");
      for(int i=0;i<8;++i)if(maps[i]!=i)throw std::runtime_error("supported candidate did not write expected map");
      finish(control,incumbent,0,"",trigger,scratch,1);
    } catch(std::exception const& error) {
      std::string message=error.what();
      if(control=="supported" || incumbent || message.rfind("cu",0)!=0)
        throw;
      finish(control,incumbent,1,message,trigger,scratch,1);
    }
  } catch(std::exception const& error) { std::cerr<<error.what()<<std::endl;std::_Exit(1); }
}
