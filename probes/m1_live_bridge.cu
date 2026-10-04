// Task-local ELF bridge for the hash-pinned installed runner. No installed writes.
// The incumbent runMoe, TMA setup, GEMMs and three-step fallback remain loaded.
#include "../kernels/m1_installed_preparation.cuh"
#include "../kernels/m1_sf_layout_contract.hpp"
#include "m1_installed_bridge.cuh"
#include "m1_live_symbols.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

namespace mx = megartx::experimental;
namespace sk = tensorrt_llm::kernels::cutlass_kernels;
using Runner = sk::CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16,
                                    __nv_fp4_e2m1, __nv_bfloat16>;
using Desc = sk::TmaWarpSpecializedGroupedGemmInput;
// This ABI type must have external linkage: NVCC internalizes a C function
// whose parameter type is declared in an anonymous namespace.
struct View { void* pointer; uint64_t bytes; void* storage; uint64_t storage_bytes; };
static_assert(sizeof(View)==32, "live View ABI changed");
using M1ExternalObserver = int (*)(char const*,char const*,void const*,uint64_t,uint64_t,int);
static std::atomic<M1ExternalObserver> m1_external_observer{nullptr};
static std::atomic<unsigned> m1_active_leases{0};
static std::mutex m1_observer_mutex;

namespace {
mx::M1SfLayoutContract<Desc::NVFP4BlockScaledConfig> const& sf_layout_contract() {
  static const mx::M1SfLayoutContract<Desc::NVFP4BlockScaledConfig> proof;
  return proof;
}
constexpr char runner_symbol[] = "_ZN12tensorrt_llm7kernels15cutlass_kernels18CutlassMoeFCRunnerI13__nv_fp4_e2m1S3_13__nv_bfloat16S3_S4_Lb0ELNS1_21Sm90Wfp4Afp8ScaleModeE0EvE6runMoeEPKvS8_bPKiPKfS8_S8_NS1_16ActivationParamsES8_S8_NS1_11QuantParamsElllliiPcPvPiNS1_20MOEParallelismConfigEbbRNS0_10LoraParamsEbbbRNS1_19MoeMinLatencyParamsEbP11CUstream_st";
constexpr char live_tactic[] = "Cutlass GEMM Tactic\n\tstyle=TMA Warp Specialized\n\tsm: 120\n"
    "\ttile shape ID: 128x128x128\n\tcluster shape ID: 1x1x1\n"
    "\tdynamic cluster shape ID: undefined\n\tfallback cluster shape ID: undefined\n"
    "\tmainloop sched: 0\n\tepi sched: 0\n\tenable cuda kernel: false\n"
    "\tepilogue fusion type: 0\n\tswap_ab: false\n";
struct Lease {
  // AQ, SF, IDs, route weights, output, workspace, private unused route scratch.
  // Two original weight tensors and six original typed quantization tensors
  // follow the seven preparation/output owners; none is copied or rewritten.
  View views[15]{};
  cudaStream_t stream{};
  bool active=false, fused_requested=false, runner_seen=false, maps_seen=false;
  bool qualified=false, candidate=false, expand_seen=false;
  bool capture_enabled=true;
  M1ExternalObserver observer=nullptr;
  std::string directory, error, metadata;
};
thread_local Lease lease;
struct Invocation {
  mx::InstalledPreparationCall call;
  std::map<std::string,std::pair<size_t,size_t>> regions;
};
thread_local Invocation* invocation=nullptr;
// Untimed attribution only. Each pair is accumulated host nanoseconds/count.
// Thread-local state cannot affect eligibility, arithmetic or stream ordering.
thread_local bool attribution_enabled=false;
thread_local std::array<uint64_t,10> attribution{};
struct AttributionPhase {
  int index;
  std::chrono::steady_clock::time_point start;
  explicit AttributionPhase(int i):index(attribution_enabled?i:-1) {
    if(index>=0)start=std::chrono::steady_clock::now();
  }
  ~AttributionPhase() {
    if(index>=0) {
      attribution[2*index]+=std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now()-start).count();
      ++attribution[2*index+1];
    }
  }
};

void* stock_symbol(char const* name) {
  static void* module=[] {
    auto path=std::getenv("MEGARTX_M1_STOCK_MODULE");
    if(!path)throw std::runtime_error("missing pinned incumbent module");
    auto p=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if(!p)throw std::runtime_error(dlerror());
    return p;
  }();
  auto result=dlsym(module,name);
  if(!result)throw std::runtime_error(std::string("missing incumbent symbol: ")+name);
  Dl_info origin{};
  if(!dladdr(result,&origin) || !origin.dli_fname ||
     !std::filesystem::equivalent(origin.dli_fname,std::getenv("MEGARTX_M1_STOCK_MODULE")))
    throw std::runtime_error("incumbent symbol module identity mismatch");
  return result;
}
bool contains(View const& v, void const* p, size_t bytes) {
  auto start=reinterpret_cast<uintptr_t>(v.pointer),q=reinterpret_cast<uintptr_t>(p);
  return q>=start && q-start<=v.bytes && bytes<=v.bytes-(q-start);
}
void require(bool yes,char const* error) { if(!yes)throw std::runtime_error(error); }
void observe(char const* event,char const* name,void const* data,size_t bytes,
             cudaStream_t stream,int value) {
  if(!lease.observer)return;
  require(lease.observer(event,name,data,bytes,reinterpret_cast<uint64_t>(stream),value)==0,
          "external observer callback failed");
}
bool pinned_callsite(void* pc) {
  Dl_info origin{};
  auto path=std::getenv("MEGARTX_M1_STOCK_MODULE");
  return path && dladdr(pc,&origin) && origin.dli_fname &&
      std::filesystem::equivalent(origin.dli_fname,path);
}
void capture(char const* name,void const* p,size_t bytes,cudaStream_t stream) {
  if(!lease.capture_enabled && !lease.observer)return;
  require(bytes<=8u<<20,"capture exceeds 8 MiB");
  std::vector<unsigned char> data(bytes);
  mx::require_cuda_success(cudaMemcpyAsync(data.data(),p,bytes,cudaMemcpyDeviceToHost,stream),"live capture copy");
  mx::require_cuda_success(cudaStreamSynchronize(stream),"live capture fence");
  if(lease.observer)observe("payload",name,data.data(),data.size(),stream,0);
  if(!lease.capture_enabled)return;
  auto path=std::filesystem::path(lease.directory)/name;
  require(!std::filesystem::exists(path),"refusing to replace live capture");
  std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<char*>(data.data()),data.size());
  require(bool(out),"live capture write failed");
}
void capture_preparation(bool before) {
  if(!lease.capture_enabled && !lease.observer)return;
  auto const& b=invocation->call.buffers;
  if(before) {
    capture("sf-before.bin",b.expanded_sf,mx::kSFBytes,lease.stream);
    capture("input-aq.bin",b.aq,mx::kAQBytes,lease.stream);
    capture("input-sf.bin",b.sf,128*mx::kSFBlocks,lease.stream);
    capture("ids.bin",b.ids,32,lease.stream);
    capture("route-weights.bin",b.weight_bits,32,lease.stream);
    capture("fc1-act-global.bin",lease.views[9].pointer,128*4,lease.stream);
    capture("fc1-global.bin",lease.views[11].pointer,128*4,lease.stream);
    capture("fc2-act-global.bin",lease.views[12].pointer,128*4,lease.stream);
    capture("fc2-global.bin",lease.views[14].pointer,128*4,lease.stream);
  } else {
    capture("sf-after.bin",b.expanded_sf,mx::kSFBytes,lease.stream);
    capture("expanded-aq.bin",b.expanded_aq,mx::kTopK*mx::kAQBytes,lease.stream);
    capture("slot-to-sorted.bin",b.slot_to_sorted,32,lease.stream);
    capture("sorted-to-slot.bin",b.sorted_to_slot,32,lease.stream);
    capture("offsets.bin",b.offsets,129*8,lease.stream);
  }
}
}

// C ABI used only by the verified Python request context. Errors never cross ctypes.
extern "C" __attribute__((visibility("default"))) char const* megartx_m1_contract_v2() {
  return M1_LIVE_CONTRACT_JSON;
}
extern "C" __attribute__((visibility("default"))) int megartx_m1_set_external_observer_v1(
    M1ExternalObserver callback) {
  std::lock_guard<std::mutex> lock(m1_observer_mutex);
  if(m1_active_leases.load(std::memory_order_acquire)!=0)return -1;
  m1_external_observer.store(callback,std::memory_order_release);
  return 0;
}
namespace {
int begin_lease(
    uint32_t abi_version,View const* views,uint32_t view_count,uint32_t view_bytes,
    uint64_t stream,int fused,char const* directory,bool capture_enabled) {
  bool owns_lease=false;
  try {
    require(!lease.active && !invocation,"nested live lease");
    // Validate framing before any array access (including bad pointer tests).
    require(abi_version==2 && view_count==15 && view_bytes==sizeof(View),
            "live lease ABI/version/view framing mismatch");
    // Prove the fixed source layouts once before any admitted device call.
    // Initialization errors are handled by this same C ABI error boundary.
    sf_layout_contract();
    lease=Lease{};
    require(views && stream && (!capture_enabled || directory) && (fused==0 || fused==1),"invalid live lease");
    for(int i=0;i<15;++i) {
      auto const& v=views[i];
      require(v.pointer && v.storage && v.bytes && v.storage_bytes,"missing exact owner extent");
      View storage{v.storage,v.storage_bytes,v.storage,v.storage_bytes};
      require(contains(storage,v.pointer,v.bytes),"view exceeds retained storage");
      lease.views[i]=v;
    }
    require(views[5].pointer==views[5].storage && views[5].bytes==views[5].storage_bytes,
            "workspace must be a complete dedicated retained owner");
    require(views[5].bytes<=8u<<20 && views[6].bytes<=(8u<<20)-views[5].bytes,
            "live workspace exceeds 8 MiB scope");
    for(int i=0;i<7;++i)for(int j=i+1;j<7;++j) {
      auto a=reinterpret_cast<uintptr_t>(views[i].pointer);
      auto b=reinterpret_cast<uintptr_t>(views[j].pointer);
      require(a<b ? views[i].bytes<=b-a : views[j].bytes<=a-b,
              "live operand views overlap before preparation");
    }
    for(int i=7;i<15;++i)for(int j=4;j<7;++j) {
      auto a=reinterpret_cast<uintptr_t>(views[i].pointer);
      auto b=reinterpret_cast<uintptr_t>(views[j].pointer);
      require(a<b ? views[i].bytes<=b-a : views[j].bytes<=a-b,
              "original quantization owner overlaps a mutable live view");
    }
    lease.stream=reinterpret_cast<cudaStream_t>(stream);
    lease.fused_requested=fused;lease.capture_enabled=capture_enabled;
    if(capture_enabled)lease.directory=directory;
    {
      // Serialize callback registration against lease admission so the setter
      // cannot succeed in the interval before this live lease is counted.
      std::lock_guard<std::mutex> lock(m1_observer_mutex);
      lease.observer=m1_external_observer.load(std::memory_order_acquire);
      require(!capture_enabled || !lease.observer,
              "external observer is limited to capture-free validation");
      lease.active=true;
      m1_active_leases.fetch_add(1,std::memory_order_acq_rel);
      owns_lease=true;
    }
    if(lease.observer)
      observe("lease_begin",fused?"fused":"stock",nullptr,0,lease.stream,capture_enabled?1:0);
    return 0;
  } catch(std::exception const& e) {
    // A rejected nested begin must not erase or rewrite the pre-existing
    // thread-local lease. Only roll back state admitted by this invocation.
    if(owns_lease) {
      lease.error=e.what();
      lease.active=false;
      m1_active_leases.fetch_sub(1,std::memory_order_acq_rel);
    } else if(!lease.active) lease.error=e.what();
    return -1;
  }
}
}
// Preserve the captured v2 ABI and its required directory. The distinct export
// is the explicit capture-free capability; no environment flag reaches native
// admission, owner validation, operator selection or CUDA error handling.
extern "C" __attribute__((visibility("default"))) int megartx_m1_begin_v2(
    uint32_t abi_version,View const* views,uint32_t view_count,uint32_t view_bytes,
    uint64_t stream,int fused,char const* directory) {
  return begin_lease(abi_version,views,view_count,view_bytes,stream,fused,directory,true);
}
extern "C" __attribute__((visibility("default"))) int megartx_m1_begin_capture_free_v2(
    uint32_t abi_version,View const* views,uint32_t view_count,uint32_t view_bytes,
    uint64_t stream,int fused) {
  return begin_lease(abi_version,views,view_count,view_bytes,stream,fused,nullptr,false);
}
extern "C" __attribute__((visibility("default"))) char const* megartx_m1_error() { return lease.error.c_str(); }
extern "C" __attribute__((visibility("default"))) int megartx_m1_active() { return lease.active; }
extern "C" __attribute__((visibility("default"))) int megartx_m1_attribution_v1(
    int enabled,uint64_t* counters,uint32_t count) {
  if(lease.active || invocation || (enabled!=0 && enabled!=1) || !counters || count!=10)return -1;
  if(enabled) {
    if(attribution_enabled)return -1;
    attribution.fill(0);attribution_enabled=true;
  } else {
    if(!attribution_enabled)return -1;
    attribution_enabled=false;
  }
  std::copy(attribution.begin(),attribution.end(),counters);
  return 0;
}
extern "C" __attribute__((visibility("default"))) int megartx_m1_end() {
  if(invocation) { lease.error="cannot end an executing native lease";return -2; }
  bool seen=lease.runner_seen;
  int status=seen ? (lease.candidate ? 1 : 0) : -1;
  if(lease.active && lease.observer) {
    try { observe("lease_end",lease.fused_requested?"fused":"stock",nullptr,0,lease.stream,status); }
    catch(std::exception const& e) { lease.error=e.what();status=-2; }
  }
  if(lease.active)m1_active_leases.fetch_sub(1,std::memory_order_acq_rel);
  lease.active=false;
  return status;
}
extern "C" __attribute__((visibility("default"))) char const* megartx_m1_metadata() { return lease.metadata.c_str(); }
extern "C" __attribute__((visibility("default"))) int megartx_m1_verify_bindings(char const* bridge_path) {
  try {
    require(bridge_path && !lease.active && !invocation,"binding check requires an inactive lease");
    auto handle=dlopen(bridge_path,RTLD_NOW|RTLD_NOLOAD);
    require(handle,"live bridge is not loaded");
    struct Close { void* handle;~Close(){dlclose(handle);} } close{handle};
    // These four offsets are generated from the hash-pinned installed ELF.
    // Verify its vtable and PLT relocations, rather than assume interposition.
    for(auto const& hook:m1_live_hook_relocations) {
      Dl_info original{};
      require(dladdr(stock_symbol(hook.symbol),&original) && original.dli_fbase,
              "missing installed relocation owner");
      auto actual=*reinterpret_cast<void* const*>(
          static_cast<unsigned char const*>(original.dli_fbase)+hook.offset);
      auto expected=dlsym(handle,hook.symbol);
      Dl_info bound{};
      require(expected && actual==expected && dladdr(actual,&bound) && bound.dli_fname &&
          std::filesystem::equivalent(bound.dli_fname,bridge_path),
          "pinned native relocation bypasses the live bridge");
    }
    return 0;
  } catch(std::exception const& e) { lease.error=e.what();return -1; }
}

namespace tensorrt_llm::kernels::cutlass_kernels {
// Retain the complete installed runner. Only its exported prep calls are replaced
// while an exact, request-bound lease is live on this same host thread.
template<> __attribute__((visibility("default"))) void Runner::runMoe(void const* input,void const* sf,bool swizzled,
    int const* ids,float const* weights,void const* w1,void const* bias1,
    ActivationParams activation,void const* w2,void const* bias2,QuantParams quant,
    int64_t rows,int64_t hidden,int64_t unpadded,int64_t inter,int experts,int topk,
    char* workspace,void* output,int* source_map,MOEParallelismConfig parallel,
    bool alltoall,bool lora,LoraParams& lp,bool deepseek,bool mxfp8,bool minlat,
    MoeMinLatencyParams& mp,bool pdl,cudaStream_t stream) {
  using Original=void(*)(Runner*,void const*,void const*,bool,int const*,float const*,
      void const*,void const*,ActivationParams,void const*,void const*,QuantParams,
      int64_t,int64_t,int64_t,int64_t,int,int,char*,void*,int*,MOEParallelismConfig,
      bool,bool,LoraParams&,bool,bool,bool,MoeMinLatencyParams&,bool,cudaStream_t);
  auto original=reinterpret_cast<Original>(stock_symbol(runner_symbol));
  auto stock=[&] { original(this,input,sf,swizzled,ids,weights,w1,bias1,activation,w2,bias2,
      quant,rows,hidden,unpadded,inter,experts,topk,workspace,output,source_map,parallel,
      alltoall,lora,lp,deepseek,mxfp8,minlat,mp,pdl,stream); };
  if(!lease.active) { stock();return; }
  AttributionPhase runner_phase(0);
  require(pinned_callsite(__builtin_return_address(0)),"live runner caller is not the pinned module");
  require(!lease.runner_seen,"multiple native runners in one retained lease");
  lease.runner_seen=true;
  if(lease.capture_enabled || lease.observer) {
    std::ostringstream observed;
    observed<<"rows="<<rows<<" hidden="<<hidden<<" unpadded="<<unpadded<<" inter="<<inter
        <<" experts="<<experts<<" topk="<<topk<<" activation="<<int(activation.activation_type)
        <<" expected_activation="<<int(ActivationType::Geglu)<<" parallel="<<parallel
        <<" alltoall="<<alltoall<<" lora="<<lora<<" deepseek="<<deepseek<<" mxfp8="<<mxfp8
        <<" minlat="<<minlat<<" pdl="<<pdl<<" swizzled="<<swizzled
        <<" bias1="<<bool(bias1)<<" bias2="<<bool(bias2)
        <<" per_expert_scale="<<quant.fp4.fc1.use_per_expert_act_scale
        <<" fc2_per_expert_scale="<<quant.fp4.fc2.use_per_expert_act_scale
        <<" groupwise_scale="<<bool(quant.groupwise.fc1.act_scales)
        <<"\nfc1="<<(gemm1_config_?gemm1_config_->toString():"unset")
        <<"\nfc2="<<(gemm2_config_?gemm2_config_->toString():"unset")<<"\n";
    lease.metadata=observed.str();
    observe("runner_identity","runner.json",lease.metadata.data(),lease.metadata.size(),stream,0);
  }
  if(rows!=1 || hidden!=2816 || unpadded!=2816 || inter!=704 || experts!=128 || topk!=8 ||
     parallel.tp_size!=1 || parallel.ep_size!=1 || parallel.cluster_size!=1 ||
     parallel.tp_rank || parallel.ep_rank || parallel.cluster_rank || alltoall || lora ||
     deepseek || mxfp8 || minlat || pdl || !sf || !swizzled || bias1 || bias2 ||
     activation!=ActivationType::Geglu || !gemm1_config_ || !gemm2_config_ ||
     !gemm1_config_->is_tma_warp_specialized || !gemm2_config_->is_tma_warp_specialized ||
     gemm1_config_->epilogue_fusion_type!=tensorrt_llm::cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::NONE ||
     gemm2_config_->epilogue_fusion_type!=tensorrt_llm::cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::NONE ||
     gemm1_config_->toString()!=live_tactic || gemm2_config_->toString()!=live_tactic ||
     quant.groupwise.fc1.act_scales) {
    stock();return;
  }
  require(stream==lease.stream,"producer/consumer stream lease mismatch");
  require(input==lease.views[0].pointer && sf==lease.views[1].pointer &&
      ids==lease.views[2].pointer && weights==lease.views[3].pointer &&
      output==lease.views[4].pointer && workspace==lease.views[5].pointer,
      "actual runner operands differ from retained lease");
  require(w1==lease.views[7].pointer && w2==lease.views[8].pointer &&
      quant.fp4.fc1.act_global_scale==lease.views[9].pointer &&
      quant.fp4.fc1.weight_block_scale==lease.views[10].pointer &&
      quant.fp4.fc1.global_scale==lease.views[11].pointer &&
      quant.fp4.fc2.act_global_scale==lease.views[12].pointer &&
      quant.fp4.fc2.weight_block_scale==lease.views[13].pointer &&
      quant.fp4.fc2.global_scale==lease.views[14].pointer,
      "original weights/typed QuantParams differ from retained quantization owners");
  constexpr size_t bytes[15]={1408,128*176,32,32,2816*2,0,32,
      128u*1408*1408,128u*2816*352,128*4,128u*1408*176,128*4,
      128*4,128u*2816*44,128*4};
  for(int i=0;i<15;++i)if(bytes[i])require(contains(lease.views[i],lease.views[i].pointer,bytes[i]),
      "actual operand exceeds retained subview extent");
  auto regions=getWorkspaceDeviceBufferSizes(rows,hidden,inter,experts,topk,activation,
      false,false,false,false,false);
  auto ptr=[&](char const* name,size_t bytes) {
    auto r=regions.at(name);
    require(bytes<=r.first && contains(lease.views[5],workspace+r.second,r.first),"workspace reserved subview extent mismatch");
    return workspace+r.second;
  };
  mx::M1Buffers b{ids,reinterpret_cast<uint32_t const*>(weights),
      static_cast<unsigned char const*>(input),static_cast<unsigned char const*>(sf),
      source_map,reinterpret_cast<int*>(ptr("permuted_row_to_unpermuted_row",32)),
      reinterpret_cast<int64_t*>(ptr("expert_first_token_offset",129*8)),
      reinterpret_cast<unsigned char*>(ptr("overlapped_gemm1_gemm2_inputs",8*1408)),
      reinterpret_cast<uint32_t*>(lease.views[6].pointer),
      reinterpret_cast<unsigned char*>(ptr("fp4_act_scale",mx::kSFBytes))};
  size_t moe_bytes=0;for(auto const& r:regions)moe_bytes+=r.second.first;
  require(reinterpret_cast<char*>(source_map)==workspace+moe_bytes &&
      contains(lease.views[5],source_map,32),"source map differs from exact FFI workspace partition");
  mx::InstalledPreparationCall call{b,quant,stream};call.correction_runner_active=true;
  call.fc1_input_lane=mx::Fc1InputLane::InstalledPrequantizedFP4;
  Invocation current{call,regions};
  require(!invocation,"recursive native runner");invocation=&current;
  struct ClearInvocation { ~ClearInvocation(){invocation=nullptr;} } clear_invocation;
  if(lease.capture_enabled || lease.observer) {
    std::ostringstream meta;
    meta << "workspace_bytes=" << lease.views[5].bytes << "\nfc1=" << gemm1_config_->toString()
         << "\nfc2=" << gemm2_config_->toString() << "\n";
    for(auto const& r:regions)meta<<r.first<<":"<<r.second.second<<":"<<r.second.first<<"\n";
    auto layout=meta.str();
    lease.metadata+=layout;
    observe("runner_workspace","workspace.json",layout.data(),layout.size(),stream,0);
  }
  stock();
  if(lease.qualified)capture("routed-output.bin",output,2816*2,stream);
  require(lease.maps_seen && lease.expand_seen,"installed preparation call sites bypassed bridge");
}

__attribute__((visibility("default"))) bool fusedBuildExpertMapsSortFirstToken(int const* ids,int* inverse,int* source,
    int64_t* offsets,int64_t rows,int experts,int topk,int start,int end,bool pdl,cudaStream_t stream) {
  auto original=reinterpret_cast<megartx::installed_probe::Map>(stock_symbol(megartx::installed_probe::map_symbol));
  if(!invocation) return original(ids,inverse,source,offsets,rows,experts,topk,start,end,pdl,stream);
  require(pinned_callsite(__builtin_return_address(0)),"map caller is not the pinned module");
  auto& c=invocation->call;auto const& b=c.buffers;
  require(!lease.maps_seen,"duplicate map call in live invocation");lease.maps_seen=true;
  require(ids==b.ids && inverse==b.sorted_to_slot && source==b.slot_to_sorted && offsets==b.offsets &&
      rows==1 && experts==128 && topk==8 && start==0 && end==128 && !pdl && stream==c.stream,
      "actual map ABI differs from source-bound workspace ledger");
  if(!lease.capture_enabled && !lease.observer) {
    bool map_result=false;
    mx::PreparationDecision decision;
    // The helper performs one fresh check followed by immediate dispatch. The
    // phase includes eligibility here; the original diagnostic phases below
    // remain separate. No capture/observer callback can mutate the checked state.
    { AttributionPhase dispatch_phase(2);decision=mx::prepare_capture_free(c,[&] {
      map_result=original(ids,inverse,source,offsets,rows,experts,topk,start,end,pdl,stream);
    },lease.fused_requested); }
    lease.qualified=decision.qualified;
    lease.candidate=decision.backend==mx::PreparationBackend::Fused;
    return lease.candidate || map_result;
  }
  // Successful unsupported queries delegate unchanged to the incumbent. Errors
  // escape before the map callback, capture, or candidate output mutation.
  bool eligible;
  { AttributionPhase eligibility_phase(1);eligible=mx::candidate_eligible(c,true); }
  if(!eligible) {
    observe("installed_map_call","map",nullptr,0,stream,0);
    return original(ids,inverse,source,offsets,rows,experts,topk,start,end,pdl,stream);
  }
  lease.qualified=true;
  capture_preparation(true);
  bool map_result=false;
  mx::PreparationBackend backend;
  { AttributionPhase dispatch_phase(2);backend=mx::prepare(c,[&] {
    observe("installed_map_call","map",nullptr,0,stream,0);
    map_result=original(ids,inverse,source,offsets,rows,experts,topk,start,end,pdl,stream);
  },lease.fused_requested); }
  lease.candidate=backend==mx::PreparationBackend::Fused;
  auto decision=std::string("{\"backend\":\"")+(lease.candidate?"fused":"stock")+
      "\",\"incumbent_map_result\":"+(map_result?"true":"false")+"}";
  observe("candidate_status","preparation.json",decision.data(),decision.size(),stream,
          lease.candidate?1:0);
  return lease.candidate || map_result; // Preserve the caller's three-step fallback.
}

template<> __attribute__((visibility("default"))) void expandInputRowsKernelLauncher<__nv_fp4_e2m1,__nv_fp4_e2m1>(
    __nv_fp4_e2m1 const* input,__nv_fp4_e2m1* expanded,float const* weights,float* permuted,
    int const* inverse,int const* experts,int64_t rows,int64_t hidden,int topk,int expert_count,
    QuantParams const& quant,bool per_expert,int64_t* offsets,unsigned char* expanded_sf,
    unsigned char const* sf,bool swizzled,void const* pre_scale,float* dequant,
    float const* residual,float const** alpha,bool pdl,cudaStream_t stream) {
  auto original=reinterpret_cast<megartx::installed_probe::Expand>(stock_symbol(megartx::installed_probe::expand_symbol));
  auto stock=[&] { original(input,expanded,weights,permuted,inverse,experts,rows,hidden,topk,
      expert_count,quant,per_expert,offsets,expanded_sf,sf,swizzled,pre_scale,dequant,residual,alpha,pdl,stream); };
  if(!invocation) { stock();return; }
  require(pinned_callsite(__builtin_return_address(0)),"expand caller is not the pinned module");
  auto const& b=invocation->call.buffers;
  require(!lease.expand_seen,"duplicate expand call");lease.expand_seen=true;
  if(!lease.qualified) {
    observe("installed_expand_call","expand",nullptr,0,stream,0);
    stock();return;
  }
  require(input==reinterpret_cast<__nv_fp4_e2m1 const*>(b.aq) &&
      expanded==reinterpret_cast<__nv_fp4_e2m1*>(b.expanded_aq) &&
      weights==reinterpret_cast<float const*>(b.weight_bits) && inverse==b.sorted_to_slot &&
      offsets==b.offsets && expanded_sf==b.expanded_sf && sf==b.sf && rows==1 && hidden==2816 &&
      topk==8 && expert_count==128 && per_expert==invocation->call.quant.fp4.fc1.use_per_expert_act_scale &&
      swizzled && !pre_scale && !dequant &&
      !residual && !alpha && !pdl && stream==lease.stream && !permuted,
      "actual expansion ABI/epilogue differs from bound no-finalize lane");
  require(quant.fp4.fc1.act_global_scale==invocation->call.quant.fp4.fc1.act_global_scale &&
      quant.fp4.fc1.weight_block_scale==invocation->call.quant.fp4.fc1.weight_block_scale,
      "typed quantization identity changed");
  if(!lease.candidate) {
    observe("installed_expand_call","expand",nullptr,0,stream,0);
    stock();
  }
  capture_preparation(false);
}

template<> __attribute__((visibility("default"))) std::pair<Desc,Desc> Runner::setupTmaWarpSpecializedInputs(
    int64_t rows,int64_t expanded_rows,ActivationParams activation,int64_t hidden,
    int64_t unpadded,int64_t inter,int64_t expert_count,void const* input,
    unsigned char const* sf,void* output,__nv_fp4_e2m1 const* w1,__nv_fp4_e2m1 const* w2,
    QuantParams quant,__nv_bfloat16 const* bias1,__nv_bfloat16 const* bias2,bool minlat,
    MoeMinLatencyParams& mp,bool lora,int start,MOEParallelismConfig parallel,
    bool pdl,cudaStream_t stream) {
  using Original=std::pair<Desc,Desc>(*)(Runner*,int64_t,int64_t,ActivationParams,
      int64_t,int64_t,int64_t,int64_t,void const*,unsigned char const*,void*,
      __nv_fp4_e2m1 const*,__nv_fp4_e2m1 const*,QuantParams,__nv_bfloat16 const*,
      __nv_bfloat16 const*,bool,MoeMinLatencyParams&,bool,int,MOEParallelismConfig,bool,cudaStream_t);
  auto original=reinterpret_cast<Original>(stock_symbol(M1_TMA_SETUP_SYMBOL));
  // BEGIN M1 SOURCE-BOUND TMA INPUT CONTRACT
  Desc expected_tables[2];
  auto check_tables=[&](Desc const& d,int stage,bool prepared) {
    auto const& e=expected_tables[stage];
    require(d.shape_info.num_groups==128 && !d.shape_info.host_problem_shapes &&
        d.shape_info.problem_shapes==e.shape_info.problem_shapes &&
        d.stride_act==e.stride_act && d.stride_weight==e.stride_weight &&
        d.ptr_act==e.ptr_act && d.ptr_weight==e.ptr_weight &&
        d.stride_d==e.stride_d && d.ptr_d==e.ptr_d &&
        d.alpha_scale_ptr_array==e.alpha_scale_ptr_array &&
        d.fpX_block_scaling_factors_act==e.fpX_block_scaling_factors_act &&
        d.fpX_block_scaling_factors_weight==e.fpX_block_scaling_factors_weight &&
        d.fpX_block_scaling_factors_stride_act==e.fpX_block_scaling_factors_stride_act &&
        d.fpX_block_scaling_factors_stride_weight==e.fpX_block_scaling_factors_stride_weight &&
        d.ptr_c==(prepared?nullptr:e.ptr_c) && d.stride_c==(prepared?nullptr:e.stride_c) &&
        d.gemm_workspace==e.gemm_workspace && d.gemm_workspace_size==e.gemm_workspace_size &&
        !d.precomputed_scheduler_workspace && !d.precomputed_scheduler_workspace_size,
        "actual TMA table bindings differ from retained stage workspace");
  };
  if(invocation && lease.qualified) {
    require(pinned_callsite(__builtin_return_address(0)),"TMA setup caller is not pinned");
    require(lease.expand_seen && stream==lease.stream,"consumer setup precedes preparation");
    auto const& b=invocation->call.buffers;
    auto const& q=invocation->call.quant;
    require(rows==1 && expanded_rows==8 && hidden==2816 && unpadded==2816 &&
        inter==704 && expert_count==128 && activation==ActivationType::Geglu &&
        !bias1 && !bias2 && !minlat && !lora && !start && !pdl &&
        parallel.tp_size==1 && parallel.ep_size==1 && parallel.cluster_size==1 &&
        !parallel.tp_rank && !parallel.ep_rank && !parallel.cluster_rank &&
        !quant.groupwise.fc1.act_scales && !quant.groupwise.fc2.act_scales &&
        !quant.mxfp8_mxfp4.fc1.weight_block_scale && !quant.mxfp8_mxfp4.fc2.weight_block_scale &&
        !quant.mxfp8_mxfp8.fc1.weight_block_scale && !quant.mxfp8_mxfp8.fc2.weight_block_scale &&
        gemm1_config_ && gemm2_config_ && gemm1_config_->toString()==live_tactic &&
        gemm2_config_->toString()==live_tactic,
        "actual TMA setup geometry or mode differs");
    require(input==b.aq && sf==b.sf && output==lease.views[4].pointer &&
        w1==lease.views[7].pointer && w2==lease.views[8].pointer &&
        quant.fp4.fc1.act_global_scale==q.fp4.fc1.act_global_scale &&
        quant.fp4.fc1.weight_block_scale==q.fp4.fc1.weight_block_scale &&
        quant.fp4.fc1.global_scale==q.fp4.fc1.global_scale &&
        quant.fp4.fc2.act_global_scale==q.fp4.fc2.act_global_scale &&
        quant.fp4.fc2.weight_block_scale==q.fp4.fc2.weight_block_scale &&
        quant.fp4.fc2.global_scale==q.fp4.fc2.global_scale &&
        quant.fp4.fc1.use_per_expert_act_scale==q.fp4.fc1.use_per_expert_act_scale &&
        quant.fp4.fc2.use_per_expert_act_scale==q.fp4.fc2.use_per_expert_act_scale,
        "actual TMA setup operands differ from retained owners");
    auto region=[&](char const* name,size_t bytes) {
      auto r=invocation->regions.at(name);
      require(r.second<=lease.views[5].bytes && r.first<=lease.views[5].bytes-r.second &&
          bytes<=r.first,"TMA workspace region exceeds retained owner");
      return static_cast<unsigned char*>(lease.views[5].pointer)+r.second;
    };
    auto scratch=region("gemm_workspace",1);
    for(int stage=0;stage<2;++stage) {
      auto tables=region(stage?"tma_ws_gemm2_workspace":"tma_ws_gemm1_workspace",
          Desc::workspaceSize(128,Desc::FpXBlockScalingType::NVFP4));
      // This pinned host routine only partitions addresses. It neither reads nor
      // writes device memory; every result belongs to this fresh retained lease.
      expected_tables[stage].configureWorkspace(reinterpret_cast<int8_t*>(tables),128,
          scratch,invocation->regions.at("gemm_workspace").first,nullptr,0,
          Desc::FpXBlockScalingType::NVFP4);
      check_tables(stage?tma_ws_grouped_gemm2_input_:tma_ws_grouped_gemm1_input_,stage,false);
    }
    auto inputs=region("overlapped_gemm1_gemm2_inputs",8*1408);
    auto outputs=region("overlapped_gemm1_gemm2_outputs",8*2816*2);
    require(permuted_data_==reinterpret_cast<__nv_fp4_e2m1*>(inputs) &&
        fc1_result_==reinterpret_cast<__nv_fp4_e2m1*>(inputs) &&
        glu_inter_result_==outputs && fc2_result_==outputs &&
        expert_first_token_offset_==b.offsets &&
        fc1_fp4_act_scale_==b.expanded_sf && fc2_fp4_act_scale_==b.expanded_sf,
        "actual TMA producer workspace bindings differ");
  }
  // END M1 SOURCE-BOUND TMA INPUT CONTRACT
  auto result=original(this,rows,expanded_rows,activation,hidden,unpadded,inter,expert_count,
      input,sf,output,w1,w2,quant,bias1,bias2,minlat,mp,lora,start,parallel,pdl,stream);
  if(!invocation || !lease.qualified)return result;
  require(pinned_callsite(__builtin_return_address(0)),"TMA setup caller is not pinned");
  require(lease.expand_seen && stream==lease.stream,"consumer setup precedes preparation");
  // BEGIN M1 SOURCE-BOUND TMA OUTPUT CONTRACT
  // The provider's device entries are trusted only in observer-off execution.
  // This is not a device-completion check. Prior/pending launch errors still
  // propagate before GEMM; asynchronous faults may surface at a later boundary.
  mx::require_cuda_success(cudaPeekAtLastError(),"native TMA setup launch");
  for(int stage=0;stage<2;++stage) {
    auto const& d=stage?result.second:result.first;
    check_tables(d,stage,true);
    require(d.fpX_block_scaling_type==Desc::FpXBlockScalingType::NVFP4 &&
        d.fusion==Desc::EpilogueFusion::NONE && !d.enable_pdl && !d.swap_ab &&
        !d.int4_groupwise_params.enabled && !d.int4_groupwise_params.use_wfp4a16 &&
        d.shape_info.num_groups==128 &&
        !d.shape_info.host_problem_shapes && d.precomputed_scheduler_total_routed_tokens==8,
        "unqualified actual TMA host descriptor");

  }
  if(!lease.capture_enabled && !lease.observer)return result;
  // END M1 SOURCE-BOUND TMA OUTPUT CONTRACT
  // Diagnostic descriptor checks remain mandatory. These D2H
  // copies and fences stay fresh. Only the source-invariant dense-layout proof
  // is reused; captured/observer diagnostics also enumerate each actual layout.
  using Shape=Desc::ProblemShape::UnderlyingProblemShape;
  using Layout=Desc::NVFP4BlockScaledConfig::LayoutSF;
  std::ostringstream out;
  std::ostringstream masks;
  if(lease.capture_enabled || lease.observer) {
    out<<"{\"scope\":\"actual_descriptor_sf_carrier_envelopes\",\"stages\":[";
    masks<<"{\"scope\":\"source_bound_actual_sm120_grouped_tma_payload_masks\","
        "\"tile_mn\":[128,128],\"physical_tile_k\":256,\"sf_vector_size\":16,"
        "\"inactive_group_metadata_may_be_read\":true,\"stages\":[";
  }
  Desc descs[2]={result.first,result.second};
  for(int stage=0;stage<2;++stage) {
    auto const& d=descs[stage];
    require(d.fpX_block_scaling_type==Desc::FpXBlockScalingType::NVFP4 &&
        d.fusion==Desc::EpilogueFusion::NONE && !d.enable_pdl,"unqualified actual descriptor lane");
    std::vector<Shape> shapes(128);std::vector<Layout> layouts(128);
    std::vector<unsigned char const*> pointers(128);
    std::vector<void const*> aq(128);std::vector<void*> destinations(128);
    std::vector<Desc::StrideA> aq_strides(128);
    std::vector<Desc::StrideD> output_strides(128);
    { AttributionPhase readback_phase(3);
    mx::require_cuda_success(cudaMemcpyAsync(shapes.data(),d.shape_info.problem_shapes,
        shapes.size()*sizeof(Shape),cudaMemcpyDeviceToHost,stream),"descriptor shape copy");
    mx::require_cuda_success(cudaMemcpyAsync(layouts.data(),d.fpX_block_scaling_factors_stride_act,
        layouts.size()*sizeof(Layout),cudaMemcpyDeviceToHost,stream),"descriptor SF layout copy");
    mx::require_cuda_success(cudaMemcpyAsync(pointers.data(),d.fpX_block_scaling_factors_act,
        pointers.size()*sizeof(void*),cudaMemcpyDeviceToHost,stream),"descriptor SF pointer copy");
    require(contains(lease.views[5],d.ptr_act,128*sizeof(void*)) &&
        contains(lease.views[5],d.ptr_d,128*sizeof(void*)) &&
        contains(lease.views[5],d.stride_act,128*sizeof(Desc::StrideA)) &&
        contains(lease.views[5],d.stride_d,128*sizeof(Desc::StrideD)),
        "actual descriptor pointer/stride tables exceed retained workspace");
    mx::require_cuda_success(cudaMemcpyAsync(aq.data(),d.ptr_act,128*sizeof(void*),
        cudaMemcpyDeviceToHost,stream),"descriptor AQ pointer copy");
    mx::require_cuda_success(cudaMemcpyAsync(destinations.data(),d.ptr_d,128*sizeof(void*),
        cudaMemcpyDeviceToHost,stream),"descriptor output pointer copy");
    mx::require_cuda_success(cudaMemcpyAsync(aq_strides.data(),d.stride_act,128*sizeof(Desc::StrideA),
        cudaMemcpyDeviceToHost,stream),"descriptor AQ stride copy");
    mx::require_cuda_success(cudaMemcpyAsync(output_strides.data(),d.stride_d,128*sizeof(Desc::StrideD),
        cudaMemcpyDeviceToHost,stream),"descriptor output stride copy");
    mx::require_cuda_success(cudaStreamSynchronize(stream),"descriptor consumer fence"); }
    if(lease.capture_enabled || lease.observer) {
      if(stage)out<<',';out<<"{\"stage\":"<<stage+1<<",\"swap_ab\":"<<(d.swap_ab?"true":"false")<<",\"experts\":[";
      if(stage)masks<<',';
      masks<<"{\"stage\":"<<stage+1<<",\"aq_lifetime\":\""
          <<(stage?"after_incumbent_activation_reuses_preparation_owner":"preparation_to_fc1_consumption")
          <<"\",\"sf_lifetime\":\""
          <<(stage?"after_incumbent_activation_overwrites_shared_sf_owner":"preparation_to_fc1_consumption")
          <<"\",\"active_payloads\":[";
    }
    bool first=true;
    AttributionPhase validation_phase(4);
    for(int e=0;e<128;++e) {
      auto m=int64_t(cute::get<0>(shapes[e])),n=int64_t(cute::get<1>(shapes[e])),k=int64_t(cute::get<2>(shapes[e]));
      auto token_rows=d.swap_ab?n:m;
      auto bytes=size_t(cute::cosize(layouts[e]));
      auto base=reinterpret_cast<uintptr_t>(invocation->call.buffers.expanded_sf);
      auto p=reinterpret_cast<uintptr_t>(pointers[e]);
      require(token_rows==0 || token_rows==1,"actual consumer token rows differ");
      // Grouped GEMM schedules no tile for M=0. Its unused layout can have
      // a nonzero atom cosize; that is not a physical consumer read.
      if(token_rows && !(p>=base && p-base<=mx::kSFBytes && bytes<=mx::kSFBytes-(p-base)))
        throw std::runtime_error("actual SF carrier bounds: stage="+std::to_string(stage+1)+
            " expert="+std::to_string(e)+" M="+std::to_string(m)+" N="+std::to_string(n)+
            " K="+std::to_string(k)+" relative_offset="+
            std::to_string(int64_t(p)-int64_t(base))+" bytes="+std::to_string(bytes));
      if(lease.capture_enabled || lease.observer) {
        if(e)out<<',';out<<"["<<m<<','<<n<<','<<k<<','<<(token_rows?int64_t(p)-int64_t(base):-1)
            <<','<<(token_rows?bytes:0)<<"]";
      }
      if(!token_rows)continue;
      require(!d.swap_ab && m==1 && n==(stage?2816:1408) && k==(stage?704:2816) &&
          cute::size<0>(layouts[e])==128 && cute::size<1>(layouts[e])==k &&
          bytes==128*size_t(k/16),"unqualified actual physical TMA SF domain");
      // The pinned SM120 mainloop uses the group LayoutSF as the TMA source
      // domain, including its 128-row atom. Enumerate that *actual* domain;
      // the logical row-zero SF mask alone would miss physical padding reads.
      // Compare every actual semantic shape/stride tuple to the independently
      // proven source layout, before using its dense-carrier conclusion. Never
      // cache an expert pointer, route prefix, owner or returned descriptor.
      sf_layout_contract().validate(layouts[e],stage,lease.capture_enabled || lease.observer);
      require(cute::get<0>(aq_strides[e])==k && cute::get<1>(aq_strides[e])==1 &&
          cute::get<0>(output_strides[e])==n && cute::get<1>(output_strides[e])==1 &&
          contains(lease.views[5],aq[e],k/2) && contains(lease.views[5],destinations[e],n*2),
          "actual AQ/output consumer stride or extent differs");
      if(lease.capture_enabled || lease.observer) {
        auto ws=reinterpret_cast<uintptr_t>(lease.views[5].pointer);
        if(!first)masks<<',';first=false;
        masks<<"{\"expert\":"<<e<<",\"shape\":["<<m<<','<<n<<','<<k<<"],"
            "\"aq_read_range\":["<<(reinterpret_cast<uintptr_t>(aq[e])-ws)<<','<<k/2<<"],"
            "\"sf_read_range\":["<<(p-base)<<','<<bytes<<"],"
            "\"gemm_output_write_range\":["<<(reinterpret_cast<uintptr_t>(destinations[e])-ws)
            <<','<<n*2<<"]}";
      }
    }
    if(lease.capture_enabled || lease.observer) { out<<"]}";masks<<"]}"; }
  }
  if(!lease.capture_enabled) {
    if(lease.observer) {
      out<<"]}";masks<<"]}";
      auto envelopes=out.str(),physical_masks=masks.str();
      observe("json","consumer-envelopes.json",envelopes.data(),envelopes.size(),stream,0);
      observe("json","consumer-masks.json",physical_masks.data(),physical_masks.size(),stream,0);
    }
    return result;
  }
  out<<"]}";
  auto path=std::filesystem::path(lease.directory)/"consumer-envelopes.json";
  require(!std::filesystem::exists(path),"duplicate descriptor capture");
  std::ofstream file(path);file<<out.str();require(bool(file),"descriptor evidence write failed");
  masks<<"]}";
  auto mask_path=std::filesystem::path(lease.directory)/"consumer-masks.json";
  require(!std::filesystem::exists(mask_path),"duplicate physical mask capture");
  std::ofstream mask_file(mask_path);mask_file<<masks.str();require(bool(mask_file),"physical mask evidence write failed");
  return result;
}
}
