// CPU-only mocks around extracted, unchanged live setup code. No CUDA/ABI claim.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "m1_sf_layout_contract.hpp"

using cudaStream_t=void*;
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaMemcpyDeviceToHost=2;
using __nv_fp4_e2m1=unsigned char;
using __nv_bfloat16=uint16_t;
namespace cute {
template<int I,class T,size_t N> auto get(std::array<T,N> const& a){return a[I];}
}
struct Desc {
  using StrideA=std::array<int64_t,3>;
  using StrideB=StrideA;using StrideC=StrideA;using StrideD=StrideA;
  using StrideC_T=StrideA;using StrideD_T=StrideA;
  using NVFP4BlockScaledConfig=cute::Config;
  using ElementSF=unsigned char;
  struct ProblemShape {using UnderlyingProblemShape=std::array<int64_t,3>;};
  struct ShapeInfo {int num_groups=0;ProblemShape::UnderlyingProblemShape* problem_shapes=nullptr;
    ProblemShape::UnderlyingProblemShape* host_problem_shapes=nullptr;} shape_info;
  enum class FpXBlockScalingType {MXFPX,NVFP4,NONE};
  enum class EpilogueFusion {NONE,ACTIVATION,FINALIZE};
  FpXBlockScalingType fpX_block_scaling_type=FpXBlockScalingType::NONE;
  EpilogueFusion fusion=EpilogueFusion::NONE;
  bool swap_ab=false,enable_pdl=false;
  struct {bool enabled=false,use_wfp4a16=false;} int4_groupwise_params;
  void *stride_act=nullptr,*stride_weight=nullptr,*stride_c=nullptr,*stride_d=nullptr;
  void const **ptr_act=nullptr,**ptr_weight=nullptr,**ptr_c=nullptr;
  void **ptr_d=nullptr;
  float const** alpha_scale_ptr_array=nullptr;
  unsigned char const **fpX_block_scaling_factors_act=nullptr,**fpX_block_scaling_factors_weight=nullptr;
  void *fpX_block_scaling_factors_stride_act=nullptr,*fpX_block_scaling_factors_stride_weight=nullptr;
  unsigned char *gemm_workspace=nullptr,*precomputed_scheduler_workspace=nullptr;
  size_t gemm_workspace_size=0,precomputed_scheduler_workspace_size=0;
  int64_t precomputed_scheduler_total_routed_tokens=0;
  static size_t workspaceSize(int,FpXBlockScalingType){return 65536;}
  void configureWorkspace(int8_t* raw,int count,void* scratch,size_t size,void* scheduler,size_t scheduler_size,FpXBlockScalingType) {
    auto* p=reinterpret_cast<unsigned char*>(raw);
    auto alloc=[&](size_t bytes){void* q=p;p+=(bytes+255)/256*256;return q;};
    shape_info={count,static_cast<ProblemShape::UnderlyingProblemShape*>(alloc(count*sizeof(ProblemShape::UnderlyingProblemShape))),nullptr};
    stride_act=alloc(count*sizeof(StrideA));stride_weight=alloc(count*sizeof(StrideB));
    stride_c=alloc(count*sizeof(StrideC));stride_d=alloc(count*sizeof(StrideD));
    ptr_act=static_cast<void const**>(alloc(count*sizeof(void*)));
    ptr_weight=static_cast<void const**>(alloc(count*sizeof(void*)));
    ptr_c=static_cast<void const**>(alloc(count*sizeof(void*)));
    ptr_d=static_cast<void**>(alloc(count*sizeof(void*)));
    alpha_scale_ptr_array=static_cast<float const**>(alloc(count*sizeof(float*)));
    fpX_block_scaling_factors_act=static_cast<unsigned char const**>(alloc(count*sizeof(void*)));
    fpX_block_scaling_factors_weight=static_cast<unsigned char const**>(alloc(count*sizeof(void*)));
    fpX_block_scaling_factors_stride_act=alloc(count*sizeof(cute::Layout));
    fpX_block_scaling_factors_stride_weight=alloc(count*sizeof(cute::Layout));
    gemm_workspace=static_cast<unsigned char*>(scratch);gemm_workspace_size=size;
    precomputed_scheduler_workspace=static_cast<unsigned char*>(scheduler);
    precomputed_scheduler_workspace_size=scheduler_size;
  }
};
enum class ActivationType {Geglu,Identity};
using ActivationParams=ActivationType;
struct QuantParams {
  struct Stage {float const* act_global_scale=nullptr;unsigned char const* weight_block_scale=nullptr;
    float const* global_scale=nullptr;bool use_per_expert_act_scale=false;};
  struct Pair {Stage fc1,fc2;} fp4,mxfp8_mxfp4,mxfp8_mxfp8;
  struct {struct Stage {void const* act_scales=nullptr;}fc1,fc2;}groupwise;
};
struct MOEParallelismConfig {int tp_size=1,ep_size=1,cluster_size=1,tp_rank=0,ep_rank=0,cluster_rank=0;};
struct MoeMinLatencyParams {};
struct View {void* pointer;uint64_t bytes;void* storage;uint64_t storage_bytes;};
struct Lease {View views[15]{};cudaStream_t stream=nullptr;bool qualified=true,expand_seen=true,capture_enabled=false;
  void* observer=nullptr;std::string directory;} lease;
namespace mx {
constexpr size_t kSFBytes=16384*176;
struct M1Buffers {unsigned char const* aq;unsigned char const* sf;int64_t* offsets;unsigned char* expanded_aq;unsigned char* expanded_sf;};
struct Call {M1Buffers buffers;QuantParams quant;cudaStream_t stream;};
inline void require_cuda_success(int e,char const* op){if(e)throw std::runtime_error(std::string(op)+": mock CUDA error "+std::to_string(e));}
}
struct Invocation {mx::Call call;std::map<std::string,std::pair<size_t,size_t>>regions;};
Invocation* invocation=nullptr;
struct AttributionPhase {explicit AttributionPhase(int){}};
struct Config {bool good=true;std::string toString()const{return good?"pinned":"other";}};
char const* live_tactic="pinned";
char const* M1_TMA_SETUP_SYMBOL="setup";
struct Runner {
  Config c1,c2;Config* gemm1_config_=&c1;Config* gemm2_config_=&c2;
  __nv_fp4_e2m1 *permuted_data_=nullptr,*fc1_result_=nullptr;
  void *glu_inter_result_=nullptr,*fc2_result_=nullptr;
  int64_t* expert_first_token_offset_=nullptr;
  unsigned char *fc1_fp4_act_scale_=nullptr,*fc2_fp4_act_scale_=nullptr;
  Desc tma_ws_grouped_gemm1_input_,tma_ws_grouped_gemm2_input_;
  std::pair<Desc,Desc> setupTmaWarpSpecializedInputs(int64_t,int64_t,ActivationParams,int64_t,int64_t,int64_t,int64_t,
    void const*,unsigned char const*,void*,__nv_fp4_e2m1 const*,__nv_fp4_e2m1 const*,QuantParams,__nv_bfloat16 const*,
    __nv_bfloat16 const*,bool,MoeMinLatencyParams&,bool,int,MOEParallelismConfig,bool,cudaStream_t);
};
struct Mock {int provider=0,consumer=0,copies=0,fences=0,peeks=0,error=0,publish_error=0,copy_error=0,fence_error=0;
  bool pinned=true,provider_throw=false;std::function<void(Desc&,int)> drift;} mock;
void require(bool x,char const* msg){if(!x)throw std::runtime_error(msg);}
bool pinned_callsite(void*){return mock.pinned;}
void observe(char const*,char const*,void const*,size_t,cudaStream_t,int){}
int cudaPeekAtLastError(){++mock.peeks;return mock.error;}
int cudaMemcpyAsync(void* dst,void const* src,size_t n,int,cudaStream_t){++mock.copies;if(mock.copy_error)return mock.copy_error;std::memcpy(dst,src,n);return 0;}
int cudaStreamSynchronize(cudaStream_t){++mock.fences;return mock.fence_error;}
mx::M1Buffers active_buffers{};
std::pair<Desc,Desc> provider(Runner* runner,int64_t,int64_t,ActivationParams,int64_t,int64_t,int64_t,int64_t,
    void const*,unsigned char const*,void*,__nv_fp4_e2m1 const*,__nv_fp4_e2m1 const*,QuantParams,__nv_bfloat16 const*,
    __nv_bfloat16 const*,bool,MoeMinLatencyParams&,bool,int,MOEParallelismConfig,bool,cudaStream_t) {
  ++mock.provider;if(mock.provider_throw)throw std::runtime_error("mock provider failure");
  auto result=std::make_pair(runner->tma_ws_grouped_gemm1_input_,runner->tma_ws_grouped_gemm2_input_);
  for(int stage=0;stage<2;++stage){auto& d=stage?result.second:result.first;
    d.ptr_c=nullptr;d.stride_c=nullptr;d.fpX_block_scaling_type=Desc::FpXBlockScalingType::NVFP4;
    d.precomputed_scheduler_total_routed_tokens=8;
    int k=stage?704:2816,n=stage?2816:1408;
    for(int e=0;e<128;++e){
      d.shape_info.problem_shapes[e]={0,n,k};
      // Inactive payload entries deliberately stay stale, including expert 0.
      if(e<1 || e>8)continue;
      int rank=e-1;d.shape_info.problem_shapes[e]={1,n,k};
      static_cast<cute::Layout*>(d.fpX_block_scaling_factors_stride_act)[e]=cute::Config::tile_atom_to_shape_SFA(cute::make_shape(1,n,k,1));
      d.fpX_block_scaling_factors_act[e]=active_buffers.expanded_sf+e*128*(k/16);
      d.ptr_act[e]=active_buffers.expanded_aq+rank*k/2;
      d.ptr_d[e]=static_cast<unsigned char*>(runner->glu_inter_result_)+rank*n*2;
      static_cast<Desc::StrideA*>(d.stride_act)[e]={k,1,0};
      static_cast<Desc::StrideD*>(d.stride_d)[e]={n,1,0};
    }
    if(mock.drift)mock.drift(d,stage);
  }
  if(mock.publish_error)mock.error=mock.publish_error;
  return result;
}
void* stock_symbol(char const*){return reinterpret_cast<void*>(&provider);}
auto const& sf_layout_contract(){static megartx::experimental::M1SfLayoutContract<cute::Config> proof;return proof;}
#include "extracted_tma.inc"

template<class T>T bump(T p){return reinterpret_cast<T>(reinterpret_cast<uintptr_t>(p)+16);}
struct Fixture {
  alignas(256) std::array<unsigned char,4<<20> workspace{};
  std::array<unsigned char,22528> input{},sf{},output{},weight1{},weight2{};
  std::array<float,128> globals{};
  Runner runner;Invocation current;MoeMinLatencyParams mp;
  int64_t rows=1,expanded=8,hidden=2816,unpadded=2816,inter=704,experts=128;
  ActivationParams activation=ActivationType::Geglu;bool minlat=false,lora=false,pdl=false;int start=0;
  MOEParallelismConfig parallel;QuantParams quant;
  void const* in=input.data();unsigned char const* input_sf=sf.data();void* out=output.data();
  __nv_fp4_e2m1 const* w1=weight1.data();__nv_fp4_e2m1 const* w2=weight2.data();
  __nv_bfloat16 const *bias1=nullptr,*bias2=nullptr;cudaStream_t stream=reinterpret_cast<void*>(1);
  Fixture(){reset();}
  void reset(){
    mock={};lease={};lease.stream=stream;lease.qualified=true;lease.expand_seen=true;
    current.regions={{"overlapped_gemm1_gemm2_inputs",{45056,0}},{"overlapped_gemm1_gemm2_outputs",{45056,65536}},
      {"fp4_act_scale",{mx::kSFBytes,131072}},{"tma_ws_gemm1_workspace",{65536,3145728}},
      {"tma_ws_gemm2_workspace",{65536,3211264}},{"gemm_workspace",{65536,3276800}}};
    lease.views[4]={out,output.size(),out,output.size()};lease.views[5]={workspace.data(),workspace.size(),workspace.data(),workspace.size()};
    lease.views[7]={weight1.data(),weight1.size(),weight1.data(),weight1.size()};
    lease.views[8]={weight2.data(),weight2.size(),weight2.data(),weight2.size()};
    quant.fp4.fc1={globals.data(),sf.data(),globals.data(),false};quant.fp4.fc2=quant.fp4.fc1;
    current.call={{input.data(),sf.data(),reinterpret_cast<int64_t*>(workspace.data()+3407872),workspace.data(),workspace.data()+131072},quant,stream};
    active_buffers=current.call.buffers;invocation=&current;
    runner.permuted_data_=runner.fc1_result_=workspace.data();
    runner.glu_inter_result_=runner.fc2_result_=workspace.data()+65536;
    runner.expert_first_token_offset_=current.call.buffers.offsets;
    runner.fc1_fp4_act_scale_=runner.fc2_fp4_act_scale_=current.call.buffers.expanded_sf;
    runner.tma_ws_grouped_gemm1_input_.configureWorkspace(reinterpret_cast<int8_t*>(workspace.data()+3145728),128,workspace.data()+3276800,65536,nullptr,0,Desc::FpXBlockScalingType::NVFP4);
    runner.tma_ws_grouped_gemm2_input_.configureWorkspace(reinterpret_cast<int8_t*>(workspace.data()+3211264),128,workspace.data()+3276800,65536,nullptr,0,Desc::FpXBlockScalingType::NVFP4);
    // Legal inactive slots may retain prior-call values; no payload validity is
    // claimed for them. The provider must still refresh every problem shape.
    for(auto* d:{&runner.tma_ws_grouped_gemm1_input_,&runner.tma_ws_grouped_gemm2_input_})for(int e=0;e<128;++e){
      d->shape_info.problem_shapes[e]={19,23,29};
      d->ptr_act[e]=reinterpret_cast<void*>(16);d->ptr_d[e]=reinterpret_cast<void*>(32);
      d->fpX_block_scaling_factors_act[e]=reinterpret_cast<unsigned char*>(48);
      static_cast<cute::Layout*>(d->fpX_block_scaling_factors_stride_act)[e]={{128,999,1},{17,4,512}};
      static_cast<Desc::StrideA*>(d->stride_act)[e]={111,222,333};
      static_cast<Desc::StrideD*>(d->stride_d)[e]={444,555,666};
    }
  }
  bool run(){try{runner.setupTmaWarpSpecializedInputs(rows,expanded,activation,hidden,unpadded,inter,experts,in,input_sf,out,w1,w2,quant,bias1,bias2,minlat,mp,lora,start,parallel,pdl,stream);++mock.consumer;return true;}
    catch(std::exception const&){return false;}}
};
using Mutation=std::function<void(Desc&)>;
std::vector<Mutation> table_mutations(){return {
  [](Desc&d){d.shape_info.num_groups=127;},[](Desc&d){d.shape_info.host_problem_shapes=d.shape_info.problem_shapes;},
  [](Desc&d){d.shape_info.problem_shapes=bump(d.shape_info.problem_shapes);},
  [](Desc&d){d.stride_act=bump(d.stride_act);},[](Desc&d){d.stride_weight=bump(d.stride_weight);},
  [](Desc&d){d.stride_d=bump(d.stride_d);},[](Desc&d){d.ptr_act=bump(d.ptr_act);},
  [](Desc&d){d.ptr_weight=bump(d.ptr_weight);},[](Desc&d){d.ptr_d=bump(d.ptr_d);},
  [](Desc&d){d.ptr_c=bump(d.ptr_c);},[](Desc&d){d.stride_c=bump(d.stride_c);},
  [](Desc&d){d.alpha_scale_ptr_array=bump(d.alpha_scale_ptr_array);},
  [](Desc&d){d.fpX_block_scaling_factors_act=bump(d.fpX_block_scaling_factors_act);},
  [](Desc&d){d.fpX_block_scaling_factors_weight=bump(d.fpX_block_scaling_factors_weight);},
  [](Desc&d){d.fpX_block_scaling_factors_stride_act=bump(d.fpX_block_scaling_factors_stride_act);},
  [](Desc&d){d.fpX_block_scaling_factors_stride_weight=bump(d.fpX_block_scaling_factors_stride_weight);},
  [](Desc&d){d.gemm_workspace=bump(d.gemm_workspace);},[](Desc&d){--d.gemm_workspace_size;},
  [](Desc&d){d.precomputed_scheduler_workspace=reinterpret_cast<unsigned char*>(16);},
  [](Desc&d){d.precomputed_scheduler_workspace_size=16;}};}
void check(bool ok,char const* message){if(!ok)throw std::runtime_error(message);}
int main(int argc,char**argv){try{
  check(argc>=2,"case required");std::string name=argv[1];auto fresh=[](){return std::make_unique<Fixture>();};
  int cases=0;
  if(name=="production" || name=="observer" || name=="captured"){
    auto f=fresh();if(name=="observer")lease.observer=reinterpret_cast<void*>(1);
    if(name=="captured"){check(argc>=3,"capture directory required");lease.capture_enabled=true;lease.directory=argv[2];std::filesystem::create_directory(lease.directory);}
    check(f->run(),"positive call rejected");check(mock.provider==1 && mock.consumer==1,"provider/consumer count");
    check(mock.copies==(name=="production"?0:14) && mock.fences==(name=="production"?0:2),"copy/fence count");
    check(mock.peeks==1,"launch error check missing");++cases;
  }else if(name=="fresh"){
    for(int kind=0;kind<4;++kind){auto f=fresh();check(f->run(),"first call");
      if(kind==0)f->rows=2;
      if(kind==1)f->runner.tma_ws_grouped_gemm1_input_.ptr_act=bump(f->runner.tma_ws_grouped_gemm1_input_.ptr_act);
      if(kind==2)lease.views[5].bytes=1024;
      if(kind==3)mock.drift=[](Desc&d,int){d.swap_ab=true;};
      check(!f->run(),"cached admission or returned descriptor");
      check(mock.provider==(kind==3?2:1) && mock.consumer==1,"fresh rejection boundary");++cases;}
  }else if(name=="unsupported"){
    for(int mode=0;mode<2;++mode){auto f=fresh();if(mode)lease.qualified=false;else invocation=nullptr;
      mock.error=719;check(f->run(),"unsupported changed");check(mock.provider==1 && mock.consumer==1 && !mock.peeks && !mock.copies && !mock.fences,"unsupported side effects");++cases;}
  }else if(name=="preflight"){
    std::vector<std::function<void(Fixture&)>> changes={
      [](Fixture&f){f.rows=2;},[](Fixture&f){f.expanded=7;},[](Fixture&f){f.hidden=2800;},[](Fixture&f){f.unpadded=2800;},
      [](Fixture&f){f.inter=700;},[](Fixture&f){f.experts=127;},[](Fixture&f){f.activation=ActivationType::Identity;},
      [](Fixture&f){f.bias1=reinterpret_cast<uint16_t*>(16);},[](Fixture&f){f.bias2=reinterpret_cast<uint16_t*>(16);},
      [](Fixture&f){f.minlat=true;},[](Fixture&f){f.lora=true;},[](Fixture&f){f.start=1;},[](Fixture&f){f.pdl=true;},
      [](Fixture&f){f.parallel.tp_size=2;},[](Fixture&f){f.parallel.ep_size=2;},[](Fixture&f){f.parallel.cluster_size=2;},
      [](Fixture&f){f.parallel.tp_rank=1;},[](Fixture&f){f.parallel.ep_rank=1;},[](Fixture&f){f.parallel.cluster_rank=1;},
      [](Fixture&f){f.in=bump(f.in);},[](Fixture&f){f.input_sf=bump(f.input_sf);},[](Fixture&f){f.out=bump(f.out);},
      [](Fixture&f){f.w1=bump(f.w1);},[](Fixture&f){f.w2=bump(f.w2);},[](Fixture&f){f.stream=reinterpret_cast<void*>(2);},
      [](Fixture&){mock.pinned=false;},[](Fixture&){lease.expand_seen=false;},
      [](Fixture&f){f.runner.gemm1_config_=nullptr;},[](Fixture&f){f.runner.gemm2_config_=nullptr;},
      [](Fixture&f){f.runner.c1.good=false;},[](Fixture&f){f.runner.c2.good=false;},
      [](Fixture&f){f.runner.permuted_data_=bump(f.runner.permuted_data_);},[](Fixture&f){f.runner.fc1_result_=bump(f.runner.fc1_result_);},
      [](Fixture&f){f.runner.glu_inter_result_=bump(f.runner.glu_inter_result_);},[](Fixture&f){f.runner.fc2_result_=bump(f.runner.fc2_result_);},
      [](Fixture&f){f.runner.expert_first_token_offset_=bump(f.runner.expert_first_token_offset_);},
      [](Fixture&f){f.runner.fc1_fp4_act_scale_=bump(f.runner.fc1_fp4_act_scale_);},[](Fixture&f){f.runner.fc2_fp4_act_scale_=bump(f.runner.fc2_fp4_act_scale_);},
      [](Fixture&f){f.current.regions["tma_ws_gemm1_workspace"].first=1;},
      [](Fixture&f){f.current.regions["tma_ws_gemm2_workspace"].second=UINT64_MAX;},
      [](Fixture&f){f.current.regions["overlapped_gemm1_gemm2_outputs"].first=1;},
      [](Fixture&f){f.current.regions["overlapped_gemm1_gemm2_inputs"].first=1;},
      [](Fixture&){lease.views[5].bytes=1024;},
      [](Fixture&f){f.quant.groupwise.fc1.act_scales=f.in;},[](Fixture&f){f.quant.groupwise.fc2.act_scales=f.in;},
      [](Fixture&f){f.quant.mxfp8_mxfp4.fc1.weight_block_scale=f.input_sf;},[](Fixture&f){f.quant.mxfp8_mxfp4.fc2.weight_block_scale=f.input_sf;},
      [](Fixture&f){f.quant.mxfp8_mxfp8.fc1.weight_block_scale=f.input_sf;},[](Fixture&f){f.quant.mxfp8_mxfp8.fc2.weight_block_scale=f.input_sf;}};
    for(int stage=0;stage<2;++stage)for(int field=0;field<4;++field)changes.push_back([=](Fixture&f){auto&q=stage?f.quant.fp4.fc2:f.quant.fp4.fc1;
      if(field==0)q.act_global_scale=bump(q.act_global_scale);
      if(field==1)q.weight_block_scale=bump(q.weight_block_scale);
      if(field==2)q.global_scale=bump(q.global_scale);
      if(field==3)q.use_per_expert_act_scale=true;});
    for(auto&change:changes){auto f=fresh();change(*f);check(!f->run(),("preflight accepted case "+std::to_string(cases)).c_str());check(!mock.provider && !mock.consumer && !mock.copies && !mock.fences,"preflight published");++cases;}
    for(int stage=0;stage<2;++stage)for(auto&change:table_mutations()){auto f=fresh();change(stage?f->runner.tma_ws_grouped_gemm2_input_:f->runner.tma_ws_grouped_gemm1_input_);
      check(!f->run(),"member table drift accepted");check(!mock.provider && !mock.consumer,"member drift published");++cases;}
  }else if(name=="returned"){
    auto changes=table_mutations();std::vector<Mutation> modes={[](Desc&d){d.swap_ab=true;},[](Desc&d){d.enable_pdl=true;},
      [](Desc&d){d.fusion=Desc::EpilogueFusion::FINALIZE;},[](Desc&d){d.fpX_block_scaling_type=Desc::FpXBlockScalingType::MXFPX;},
      [](Desc&d){d.int4_groupwise_params.enabled=true;},[](Desc&d){d.int4_groupwise_params.use_wfp4a16=true;},
      [](Desc&d){d.precomputed_scheduler_total_routed_tokens=7;}};changes.insert(changes.end(),modes.begin(),modes.end());
    for(int stage=0;stage<2;++stage)for(auto&change:changes){auto f=fresh();mock.drift=[=](Desc&d,int s){if(s==stage)change(d);};
      check(!f->run(),"returned drift accepted");check(mock.provider==1 && !mock.consumer && !mock.copies && !mock.fences,"returned failure ordering");++cases;}
  }else if(name=="errors"){
    for(int late=0;late<2;++late){auto f=fresh();if(late)mock.publish_error=719;else mock.error=719;
      check(!f->run(),"pending error swallowed");check(mock.error==719 && mock.provider==1 && !mock.consumer && mock.peeks==1 && !mock.copies && !mock.fences,"pending error cleared/fallback");++cases;}
    {auto f=fresh();mock.provider_throw=true;check(!f->run(),"provider exception swallowed");check(mock.provider==1 && !mock.consumer && !mock.peeks,"provider retried");++cases;}
    for(int fence=0;fence<2;++fence){auto f=fresh();lease.observer=reinterpret_cast<void*>(1);if(fence)mock.fence_error=719;else mock.copy_error=719;
      check(!f->run(),"diagnostic CUDA failure swallowed");check(mock.provider==1 && !mock.consumer,"diagnostic CUDA fallback");++cases;}
  }else if(name=="device"){
    std::vector<Mutation> changes={[](Desc&d){d.shape_info.problem_shapes[0][0]=1;},[](Desc&d){d.shape_info.problem_shapes[1][0]=2;},[](Desc&d){d.shape_info.problem_shapes[1][1]=1;},
      [](Desc&d){d.shape_info.problem_shapes[1][2]=1;},[](Desc&d){d.fpX_block_scaling_factors_act[1]=reinterpret_cast<unsigned char*>(16);},
      [](Desc&d){d.ptr_act[1]=reinterpret_cast<void*>(16);},[](Desc&d){d.ptr_d[1]=reinterpret_cast<void*>(16);},
      [](Desc&d){static_cast<Desc::StrideA*>(d.stride_act)[1][0]=1;},[](Desc&d){static_cast<Desc::StrideD*>(d.stride_d)[1][0]=1;},
      [](Desc&d){static_cast<cute::Layout*>(d.fpX_block_scaling_factors_stride_act)[1].stride_[0]=17;}};
    for(int stage=0;stage<2;++stage)for(auto&change:changes){auto f=fresh();lease.observer=reinterpret_cast<void*>(1);mock.drift=[=](Desc&d,int s){if(s==stage)change(d);};
      check(!f->run(),"diagnostic entry drift accepted");check(mock.provider==1 && !mock.consumer && mock.copies==7*(stage+1) && mock.fences==stage+1,"entry checks moved");++cases;}
    // Metadata content is intentionally trusted in production; this is not a GPU proof.
    for(auto&change:changes){auto f=fresh();mock.drift=[=](Desc&d,int){change(d);};check(f->run(),"production introspected entries");check(!mock.copies && !mock.fences,"production readback");++cases;}
  }else throw std::runtime_error("unknown case");
  std::cout<<name<<": "<<cases<<" CPU controls passed\n";return 0;
}catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}}
