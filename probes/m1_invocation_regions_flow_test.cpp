// CPU-only host stubs around byte-extracted Invocation, lease entry, and runMoe.
// No CUDA headers, imports, device calls, native bridge build, or ABI claim.
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using cudaStream_t=void*;
using M1ExternalObserver=int (*)(char const*,char const*,void const*,uint64_t,uint64_t,int);
std::atomic<M1ExternalObserver> m1_external_observer{nullptr};
std::atomic<unsigned> m1_active_leases{0};
std::mutex m1_observer_mutex;
enum class ActivationType {Geglu,Identity};
struct ActivationParams {
  ActivationType activation_type=ActivationType::Geglu;
  bool operator!=(ActivationType other) const {return activation_type!=other;}
};
struct QuantParams {
  struct Stage {float const* act_global_scale=nullptr;unsigned char const* weight_block_scale=nullptr;
    float const* global_scale=nullptr;bool use_per_expert_act_scale=false;};
  struct {Stage fc1,fc2;} fp4;
  struct {struct {void const* act_scales=nullptr;} fc1;} groupwise;
};
struct MOEParallelismConfig {int tp_size=1,ep_size=1,cluster_size=1,tp_rank=0,ep_rank=0,cluster_rank=0;};
std::ostream& operator<<(std::ostream& out,MOEParallelismConfig const& p) {
  return out<<p.tp_size<<":"<<p.ep_size<<":"<<p.cluster_size<<":"<<p.tp_rank<<":"<<p.ep_rank<<":"<<p.cluster_rank;
}
struct LoraParams {};
struct MoeMinLatencyParams {};
namespace tensorrt_llm::cutlass_extensions {
struct CutlassGemmConfig {enum class EpilogueFusionType {NONE,OTHER};};
}
namespace tensorrt_llm::kernels::cutlass_kernels {using QuantParams=::QuantParams;}
namespace mx {
constexpr int kHidden=2816,kExperts=128,kTopK=8,kAQBytes=1408,kSFBytes=2883584;
}
#include "extracted_invocation_calls.inc"
#include "extracted_invocation_declarations.inc"
using Regions=std::map<std::string,std::pair<size_t,size_t>>;
static_assert(std::is_same_v<decltype(Invocation::regions),Regions const&>,
              "Invocation must borrow the exact map as a const lvalue reference");

// Real std::map nodes and long key strings are tracked through global allocation
// hooks. Only allocations made while building the provider's fresh map acquire
// the lifetime check. No allocator/type substitution touches production source.
struct Allocation {
  void* pointer=nullptr;
  Invocation* expected_after_scope=nullptr;
  int generation=0;
  bool live=false,guarded=false;
};
struct AllocationTrace {
  std::array<Allocation,1024> records{};
  size_t total_new=0,total_delete=0;
  int generation=0,released=0,lifetime_errors=0;
  bool collecting=false;
  Invocation* expected_after_scope=nullptr;
} allocations;
void* operator new(std::size_t bytes) {
  void* p=std::malloc(bytes?bytes:1);
  if(!p)throw std::bad_alloc();
  ++allocations.total_new;
  if(allocations.collecting) {
    bool recorded=false;
    for(auto& entry:allocations.records)if(!entry.live) {
      entry={p,allocations.expected_after_scope,allocations.generation,true,false};recorded=true;break;
    }
    if(!recorded)std::abort();
  }
  return p;
}
void operator delete(void* p) noexcept {
  if(!p)return;
  ++allocations.total_delete;
  for(auto& entry:allocations.records)if(entry.live && entry.pointer==p) {
    if(entry.guarded) {
      ++allocations.released;
      if(invocation!=entry.expected_after_scope)++allocations.lifetime_errors;
    }
    entry.live=false;break;
  }
  std::free(p);
}
void operator delete(void* p,std::size_t) noexcept {::operator delete(p);}
void* operator new[](std::size_t bytes) {return ::operator new(bytes);}
void operator delete[](void* p) noexcept {::operator delete(p);}
void operator delete[](void* p,std::size_t) noexcept {::operator delete(p);}

void check(bool yes,char const* error) {if(!yes)throw std::runtime_error(error);}
struct InjectedFailure:std::runtime_error {explicit InjectedFailure(char const* stage):std::runtime_error(stage){}};
struct Config {
  bool is_tma_warp_specialized=true;
  tensorrt_llm::cutlass_extensions::CutlassGemmConfig::EpilogueFusionType epilogue_fusion_type=
      tensorrt_llm::cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::NONE;
  bool good=true;
  std::string toString() const {return good?"pinned":"other";}
};
char const* live_tactic="pinned";
char const* runner_symbol="runner";
struct AttributionPhase {explicit AttributionPhase(int){}};
void sf_layout_contract() {}
struct Runner {
  Config c1,c2;Config* gemm1_config_=&c1;Config* gemm2_config_=&c2;
  Regions getWorkspaceDeviceBufferSizes(int64_t,int64_t,int64_t,int,int,ActivationParams,bool,bool,bool,bool,bool);
  void runMoe(void const*,void const*,bool,int const*,float const*,void const*,void const*,
      ActivationParams,void const*,void const*,QuantParams,int64_t,int64_t,int64_t,int64_t,int,int,
      char*,void*,int*,MOEParallelismConfig,bool,bool,LoraParams&,bool,bool,bool,MoeMinLatencyParams&,bool,cudaStream_t);
};
struct Mock {
  int generation=0,constructed=0,provider_calls=0,maps=0,expansions=0,tma=0,consumers=0,captures=0;
  int rejected_leases=0,rejected_runners=0;
  bool pinned=true,omit_maps=false,omit_expand=false,qualified=true;
  char const* fail=nullptr;
  void const* first_node=nullptr;
  void const* owner_address=nullptr;
  std::array<int64_t,5> geometry{};
  std::function<void()> while_active;
  std::vector<std::string> events;
  std::string workspace_metadata;
} mock;
void fail(char const* stage) {if(mock.fail && std::strcmp(mock.fail,stage)==0)throw InjectedFailure(stage);}
bool pinned_callsite(void*) {return mock.pinned;}
void* stock_symbol(char const*);
void capture(char const*,void const*,size_t,cudaStream_t);
#include "extracted_invocation_functions.inc"

struct RegionSpec {char const* name;size_t bytes;};
constexpr std::array<RegionSpec,8> specs{{
  {"permuted_row_to_unpermuted_row",32}, {"expert_first_token_offset",129*8},
  {"overlapped_gemm1_gemm2_inputs",8*1408}, {"fp4_act_scale",mx::kSFBytes},
  {"overlapped_gemm1_gemm2_outputs",8*2816*2}, {"tma_ws_gemm1_workspace",65536},
  {"tma_ws_gemm2_workspace",65536}, {"gemm_workspace",65536}
}};
size_t region_size(RegionSpec const& spec,int generation) {return spec.bytes+size_t(generation)*256;}
size_t total_regions(int generation) {size_t total=0;for(auto const& spec:specs)total+=region_size(spec,generation);return total;}
Regions make_regions(int generation) {
  Regions result;size_t offset=0;
  for(auto const& spec:specs) {auto bytes=region_size(spec,generation);result.emplace(spec.name,std::make_pair(bytes,offset));offset+=bytes;}
  return result;
}
Regions Runner::getWorkspaceDeviceBufferSizes(int64_t rows,int64_t hidden,int64_t inter,int experts,int topk,
    ActivationParams activation,bool a,bool b,bool c,bool d,bool e) {
  ++mock.constructed;fail("workspace factory");
  check(!(a||b||c||d||e) && !(activation!=ActivationType::Geglu),"workspace provider flags changed");
  mock.geometry={rows,hidden,inter,experts,topk};
  allocations.collecting=true;allocations.generation=mock.generation;
  allocations.expected_after_scope=invocation;
  struct StopCollecting {
    ~StopCollecting() {allocations.collecting=false;for(auto& entry:allocations.records)
      if(entry.live && entry.generation==allocations.generation)entry.guarded=true;}
  } stop_collecting;
  auto result=make_regions(mock.generation);
  mock.first_node=std::addressof(*result.begin());
  return result;
}
void inspect_current_map() {
  check(invocation!=nullptr,"missing current invocation in synchronous consumer");
  auto const& regions=invocation->regions;
  check(regions.size()==specs.size(),"region count changed");
  check(std::addressof(*regions.begin())==mock.first_node,"Invocation copied or substituted fresh map nodes");
  size_t offset=0;
  for(auto const& spec:specs) {
    auto const& r=regions.at(spec.name);
    check(r.first==region_size(spec,mock.generation) && r.second==offset,"stale region values");
    offset+=r.first;
  }
  check(invocation->call.buffers.sorted_to_slot==reinterpret_cast<int*>(
      static_cast<char*>(lease.views[5].pointer)+regions.at("permuted_row_to_unpermuted_row").second),"map owner mismatch");
  check(invocation->call.buffers.offsets==reinterpret_cast<int64_t*>(
      static_cast<char*>(lease.views[5].pointer)+regions.at("expert_first_token_offset").second),"offset owner mismatch");
  check(invocation->call.buffers.expanded_aq==static_cast<unsigned char*>(lease.views[5].pointer)+
      regions.at("overlapped_gemm1_gemm2_inputs").second,"AQ owner mismatch");
  check(invocation->call.buffers.expanded_sf==static_cast<unsigned char*>(lease.views[5].pointer)+
      regions.at("fp4_act_scale").second,"SF owner mismatch");
  mock.owner_address=lease.views[5].pointer;
}
void mock_tma() {inspect_current_map();++mock.tma;fail("TMA hook");}
void mock_expand() {inspect_current_map();++mock.expansions;fail("expand hook");lease.expand_seen=!mock.omit_expand;mock_tma();}
void mock_maps() {inspect_current_map();++mock.maps;fail("map hook");lease.maps_seen=!mock.omit_maps;mock_expand();}
void original_runner(Runner*,void const*,void const*,bool,int const*,float const*,void const*,void const*,
    ActivationParams,void const*,void const*,QuantParams,int64_t,int64_t,int64_t,int64_t,int,int,
    char*,void*,int*,MOEParallelismConfig,bool,bool,LoraParams&,bool,bool,bool,MoeMinLatencyParams&,bool,cudaStream_t) {
  ++mock.provider_calls;
  fail("original runner");
  if(!invocation) {++mock.consumers;return;}
  inspect_current_map();
  if(mock.while_active)mock.while_active();
  mock_maps();lease.qualified=mock.qualified;
  ++mock.consumers;
}
void* stock_symbol(char const* name) {check(std::strcmp(name,runner_symbol)==0,"unexpected symbol");return reinterpret_cast<void*>(&original_runner);}
int observer(char const* event,char const*,void const* data,uint64_t bytes,uint64_t,int) {
  mock.events.emplace_back(event);
  if(std::strcmp(event,"runner_workspace")==0) {
    inspect_current_map();mock.workspace_metadata.assign(static_cast<char const*>(data),bytes);
  }
  fail(event);
  return 0;
}
void capture(char const* name,void const* data,size_t bytes,cudaStream_t stream) {
  if(!lease.capture_enabled && !lease.observer)return;
  inspect_current_map();++mock.captures;
  check(std::strcmp(name,"routed-output.bin")==0 && data==lease.views[4].pointer && bytes==2816*2 && stream==lease.stream,
        "output diagnostic arguments changed");
  fail("output capture");
}

struct Fixture {
  std::vector<unsigned char> workspace=std::vector<unsigned char>(4<<20);
  // Opaque host addresses model retained owner metadata only. Weight data is
  // never accessed by the CPU provider; declared extents are not GPU evidence.
  std::array<std::array<unsigned char,64>,15> owners{};
  Runner runner;LoraParams lp;MoeMinLatencyParams mp;QuantParams quant;
  int64_t rows=1,hidden=2816,unpadded=2816,inter=704;int experts=128,topk=8;
  ActivationParams activation;MOEParallelismConfig parallel;
  bool swizzled=true,pdl=false;int* source_map=nullptr;
  cudaStream_t stream=reinterpret_cast<void*>(1);
  Fixture() {reset();}
  void reset(int generation=0) {
    check(invocation==nullptr,"previous invocation escaped runMoe");
    for(auto const& entry:allocations.records)check(!entry.live,"previous fresh map survived call completion");
    mock={};mock.generation=generation;lease=Lease{};lease.active=true;lease.capture_enabled=false;lease.stream=stream;
    m1_active_leases.store(1);
    for(int i=0;i<15;++i)lease.views[i]={owners[i].data(),size_t(1)<<30,owners[i].data(),size_t(1)<<30};
    lease.views[5]={workspace.data(),workspace.size(),workspace.data(),workspace.size()};
    quant.fp4.fc1={reinterpret_cast<float const*>(owners[9].data()),owners[10].data(),reinterpret_cast<float const*>(owners[11].data()),false};
    quant.fp4.fc2={reinterpret_cast<float const*>(owners[12].data()),owners[13].data(),reinterpret_cast<float const*>(owners[14].data()),false};
    source_map=reinterpret_cast<int*>(workspace.data()+total_regions(generation));
    allocations.released=0;allocations.lifetime_errors=0;
  }
  void run() {
    runner.runMoe(owners[0].data(),owners[1].data(),swizzled,reinterpret_cast<int const*>(owners[2].data()),
      reinterpret_cast<float const*>(owners[3].data()),owners[7].data(),nullptr,activation,owners[8].data(),nullptr,quant,
      rows,hidden,unpadded,inter,experts,topk,reinterpret_cast<char*>(workspace.data()),owners[4].data(),source_map,parallel,
      false,false,lp,false,false,false,mp,pdl,stream);
  }
};
void check_cleanup(bool created=true) {
  check(!allocations.lifetime_errors,"map storage released before restoring expected invocation");
  check(!invocation,"dangling Invocation after call completion");
  for(auto const& entry:allocations.records)check(!entry.live,"map allocation escaped scope");
  if(created)check(allocations.released>=int(specs.size()),"map destruction was not observed");
}
std::string expected_metadata(Fixture const& f) {
  std::ostringstream out;out<<"workspace_bytes="<<f.workspace.size()<<"\nfc1=pinned\nfc2=pinned\n";
  for(auto const& r:make_regions(mock.generation))out<<r.first<<":"<<r.second.second<<":"<<r.second.first<<"\n";
  return out.str();
}
void expect_error(std::function<void()> const& action,char const* expected,bool injected=false) {
  bool failed=false;
  try {action();}catch(std::exception const& error) {
    failed=true;check(std::strcmp(error.what(),expected)==0,"original exception changed");
    if(injected)check(dynamic_cast<InjectedFailure const*>(&error)!=nullptr,"injected exception type changed");
  }
  check(failed,"expected exception was swallowed");
}
int main(int argc,char** argv) {try {
  check(argc==2,"case required");std::string name=argv[1];int cases=0;
  if(name=="identity") {
    QuantParams quant;mx::InstalledPreparationCall call{{},quant,nullptr};
    auto regions=make_regions(0);
    auto before_new=allocations.total_new,before_delete=allocations.total_delete;
    {Invocation current{call,regions};check(std::addressof(current.regions)==std::addressof(regions),"not the named map");
      check(current.regions==regions,"borrowed entries changed");}
    check(allocations.total_new==before_new && allocations.total_delete==before_delete,"Invocation allocated/copied/freed map storage");
    struct OwningControl {mx::InstalledPreparationCall call;Regions regions;};
    {OwningControl control{call,regions};check(std::addressof(control.regions)!=std::addressof(regions),"owning control unexpectedly aliases");}
    check(allocations.total_new>before_new && allocations.total_delete>before_delete,"owning-copy control did not detect node allocations");
    cases=2;
  }else if(name=="normal") {
    Fixture f;f.run();check_cleanup();
    check(mock.constructed==1 && mock.provider_calls==1 && mock.maps==1 && mock.expansions==1 && mock.tma==1 && mock.consumers==1,"normal pipeline counts");
    check((mock.geometry==std::array<int64_t,5>{1,2816,704,128,8}),"workspace factory not called with current geometry");
    ++cases;
  }else if(name=="exceptions") {
    for(char const* stage:{"workspace factory","original runner","map hook","expand hook","TMA hook","runner_identity","runner_workspace","output capture"}) {
      Fixture f;mock.fail=stage;
      if(std::strcmp(stage,"runner_identity")==0 || std::strcmp(stage,"runner_workspace")==0)lease.observer=&observer;
      if(std::strcmp(stage,"output capture")==0)lease.capture_enabled=true;
      expect_error([&]{f.run();},stage,true);
      bool created=std::strcmp(stage,"workspace factory")!=0 && std::strcmp(stage,"runner_identity")!=0;
      check_cleanup(created);check(mock.provider_calls<=1,"error retried incumbent/fallback");++cases;
    }
    for(int mode=0;mode<3;++mode) {
      Fixture f;if(mode==0)lease.views[5].bytes=1;if(mode==1)mock.omit_maps=true;if(mode==2)mock.omit_expand=true;
      expect_error([&]{f.run();},mode==0?"workspace reserved subview extent mismatch":"installed preparation call sites bypassed bridge");
      check_cleanup();check(mock.provider_calls==(mode==0?0:1),"pre-guard/final error dispatch changed");++cases;
    }
    {Fixture f;f.source_map=reinterpret_cast<int*>(f.workspace.data());
      expect_error([&]{f.run();},"source map differs from exact FFI workspace partition");check_cleanup();check(!mock.provider_calls,"partition failure dispatched");++cases;}
    {Fixture f;mock.pinned=false;expect_error([&]{f.run();},"live runner caller is not the pinned module");check_cleanup(false);check(!mock.constructed && !mock.provider_calls,"caller failure crossed map boundary");++cases;}
    {Fixture f;lease.observer=&observer;mock.fail="output capture";expect_error([&]{f.run();},"output capture",true);check_cleanup();check(mock.provider_calls==1,"observer capture retried");++cases;}
  }else if(name=="reentrant") {
    Fixture f;
    mock.while_active=[&] {
      auto* outer=invocation;auto* outer_map=std::addressof(outer->regions);auto prior_metadata=lease.metadata;auto prior_error=lease.error;
      auto owner=lease.views[5].pointer;
      check(begin_lease(0,nullptr,0,0,0,0,nullptr,false)==-1,"nested lease accepted");++mock.rejected_leases;
      check(invocation==outer && std::addressof(invocation->regions)==outer_map && lease.active && lease.runner_seen &&
          lease.metadata==prior_metadata && lease.error==prior_error && lease.views[5].pointer==owner &&
          m1_active_leases.load()==1,"nested lease changed outer state");
      expect_error([&]{f.run();},"multiple native runners in one retained lease");++mock.rejected_runners;
      check(invocation==outer && std::addressof(invocation->regions)==outer_map,"nested runner replaced outer map");
      inspect_current_map();
    };
    f.run();check_cleanup();check(mock.rejected_leases==1 && mock.rejected_runners==1 && mock.constructed==1 && mock.provider_calls==1,"reentrant guard dispatch changed");++cases;
    // Exercise the later invocation guard as well, without removing its exact
    // production check. A sentinel outer invocation represents an active host
    // scope before runner_seen is set; the inner fresh map must unwind safely.
    f.reset();auto regions=make_regions(7);mx::InstalledPreparationCall call{{},f.quant,f.stream};Invocation outer{call,regions};invocation=&outer;
    expect_error([&]{f.run();},"recursive native runner");
    check(invocation==&outer && std::addressof(invocation->regions)==std::addressof(regions),"late recursion guard erased outer invocation");
    check(!allocations.lifetime_errors && !mock.provider_calls,"late recursion dispatched or cleared outer state");invocation=nullptr;check_cleanup();++cases;
  }else if(name=="fresh") {
    Fixture f;void const* first_owner=nullptr;
    for(int generation=0;generation<2;++generation) {
      f.reset(generation);f.run();check_cleanup();
      check(mock.constructed==1 && mock.provider_calls==1,"fresh map was skipped");
      if(!generation)first_owner=mock.owner_address;
      else check(mock.owner_address==first_owner,"same-owner fixture changed workspace address");
      ++cases;
    }
    Fixture other;other.reset(2);other.run();check_cleanup();check(mock.owner_address!=first_owner,"different owner fixture reused storage");++cases;
    // Guaranteed placement reuse independently checks the same map object
    // address after its original lifetime ended, with no old contents retained.
    alignas(Regions) unsigned char storage[sizeof(Regions)];QuantParams q;mx::InstalledPreparationCall call{{},q,nullptr};
    for(int generation=0;generation<2;++generation) {
      auto* regions=new(storage) Regions(make_regions(generation));
      {Invocation current{call,*regions};check(std::addressof(current.regions)==regions,"placement-reused map not borrowed");
        check(current.regions.at("fp4_act_scale").first==region_size(specs[3],generation),"stale placement-reused value");}
      regions->~Regions();++cases;
    }
    // A valid earlier call does not admit a newly unsupported geometry.
    f.reset(3);f.rows=2;f.run();check_cleanup(false);check(!mock.constructed && mock.provider_calls==1,"prior map admitted changed geometry");++cases;
  }else if(name=="diagnostics") {
    std::string captured,observed;
    for(int mode=0;mode<3;++mode) {
      Fixture f;if(mode==1)lease.capture_enabled=true;if(mode==2)lease.observer=&observer;
      f.run();check_cleanup();
      if(mode==0)check(lease.metadata.empty() && mock.workspace_metadata.empty() && mock.events.empty() && !mock.captures,"lean path produced diagnostics");
      else {
        auto expected=expected_metadata(f);check(lease.metadata.size()>=expected.size() && lease.metadata.compare(lease.metadata.size()-expected.size(),expected.size(),expected)==0,"workspace metadata changed");
        check(mock.captures==1,"diagnostic output capture missing");
        if(mode==1)captured=lease.metadata;
        else {observed=lease.metadata;check(mock.workspace_metadata==expected,"observer workspace bytes differ");
          check(mock.events==std::vector<std::string>{"runner_identity","runner_workspace"},"observer event ordering changed");}
      }
      ++cases;
    }
    check(captured==observed,"captured/observer metadata parity changed");
  }else if(name=="unsupported") {
    for(int mode=0;mode<5;++mode) {
      Fixture f;if(mode==0)lease.active=false;if(mode==1)f.rows=2;if(mode==2)f.runner.c1.good=false;
      if(mode==3)f.swizzled=false;
      if(mode==4)f.pdl=true;
      f.run();check_cleanup(false);check(!mock.constructed && mock.provider_calls==1 && mock.consumers==1 && !mock.maps,"unsupported path constructed/retained map");++cases;
    }
    {Fixture f;mock.qualified=false;f.run();check_cleanup();check(mock.provider_calls==1 && mock.consumers==1 && !mock.captures,"unqualified incumbent behavior changed");++cases;}
  }else throw std::runtime_error("unknown case");
  std::cout<<name<<": "<<cases<<" CPU controls passed\n";return 0;
}catch(std::exception const& error){std::cerr<<error.what()<<'\n';return 1;}}
