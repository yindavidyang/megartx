// Manual installed preparation capture. No expert weights, TMA setup or GEMMs.
#include "m1_installed_bridge.cuh"
#include "../kernels/m1_installed_preparation.cuh"
#define main descriptor_report_main
#include "m1_host_abi_probe.cpp"
#undef main
#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;
using Bytes = std::vector<unsigned char>;
using namespace megartx::experimental;
using namespace megartx::installed_probe;
void check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
Bytes read(fs::path p, std::size_t size) {
  if (fs::is_symlink(p) || !fs::is_regular_file(p) || fs::file_size(p) != size)
    throw std::runtime_error("fixture extent/type mismatch");
  Bytes b(size); std::ifstream f(p, std::ios::binary);
  f.read(reinterpret_cast<char*>(b.data()), size);
  if (!f) throw std::runtime_error("fixture read failed");
  return b;
}
void write(fs::path p, Bytes const& b) {
  if (fs::exists(p)) throw std::runtime_error("output exists");
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<char const*>(b.data()), b.size());
  if (!f) throw std::runtime_error("capture write failed");
}
std::size_t scratch = 0;
struct Buffer {
  unsigned char* storage = nullptr;
  std::size_t size;
  cudaStream_t stream;
  Buffer(std::size_t n, cudaStream_t s) : size(n), stream(s) {
    if (scratch + n + 64 > (8 << 20)) throw std::runtime_error("scratch limit before allocation");
    check(cudaMalloc(&storage, n + 64)); scratch += n + 64;
    check(cudaMemsetAsync(storage, 0xA7, 32, stream));
    check(cudaMemsetAsync(data(), 0xD3, n, stream));
    check(cudaMemsetAsync(data() + n, 0xB6, 32, stream));
  }
  ~Buffer() { if (storage) { cudaStreamSynchronize(stream); cudaFree(storage); } }
  Buffer(Buffer const&) = delete;
  unsigned char* data() const { return storage + 32; }
  void upload(unsigned char const* b) { check(cudaMemcpyAsync(data(), b, size, cudaMemcpyHostToDevice, stream)); }
  Bytes snapshot(bool guards = false) const {
    Bytes b(size + 64);
    check(cudaMemcpyAsync(b.data(), storage, b.size(), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    if (!std::all_of(b.begin(), b.begin()+32, [](auto v) {return v==0xA7;}) ||
        !std::all_of(b.end()-32, b.end(), [](auto v) {return v==0xB6;}))
      throw std::runtime_error("guard corruption");
    return guards ? b : Bytes(b.begin()+32,b.end()-32);
  }
};
struct Inputs {
  Buffer ids, weights, aq, sf, scale;
  explicit Inputs(cudaStream_t s) : ids(32,s),weights(32,s),aq(1408,s),sf(22528,s),scale(4,s) {}
  void upload(Bytes const& b) {
    ids.upload(b.data()); weights.upload(b.data()+32);
    aq.upload(b.data()+64); sf.upload(b.data()+1472);
    float one=1.0f; scale.upload(reinterpret_cast<unsigned char*>(&one));
    check(cudaStreamSynchronize(ids.stream));  // Host uploads stay alive through DMA completion.
  }
  Bytes snapshot() {
    Bytes b; for (auto p : {&ids,&weights,&aq,&sf}) {auto r=p->snapshot(); b.insert(b.end(),r.begin(),r.end());} return b;
  }
};
struct Outputs {
  Buffer rank, sorted, offsets, aq, weights, sf;
  explicit Outputs(cudaStream_t s) : rank(32,s),sorted(32,s),offsets(1032,s),aq(11264,s),weights(32,s),sf(2883584,s) {}
  M1Buffers bind(Inputs const& i) {
    return {reinterpret_cast<int const*>(i.ids.data()),reinterpret_cast<uint32_t const*>(i.weights.data()),
      i.aq.data(),i.sf.data(),reinterpret_cast<int*>(rank.data()),reinterpret_cast<int*>(sorted.data()),
      reinterpret_cast<int64_t*>(offsets.data()),aq.data(),reinterpret_cast<uint32_t*>(weights.data()),sf.data()};
  }
  Bytes snapshot() {
    Bytes b; for(auto p:{&rank,&sorted,&offsets,&aq,&weights}) {auto r=p->snapshot(); b.insert(b.end(),r.begin(),r.end());}
    auto r=sf.snapshot(true); b.insert(b.end(),r.begin(),r.end()); return b;
  }
};
void headroom() {
  std::size_t free,total; check(cudaMemGetInfo(&free,&total));
  if(free<(std::size_t{2}<<30)) throw std::runtime_error("GPU headroom");
  std::ifstream f("/proc/meminfo"); std::string line; bool ok=false;
  while(std::getline(f,line)) if(line.rfind("MemAvailable:",0)==0) {
    ok=std::stoull(line.substr(13))>=(std::size_t{8}<<20); break;
  }
  if(!ok) throw std::runtime_error("host headroom");
}
void abi(fs::path output) {
  if(fs::exists(output)) throw std::runtime_error("ABI output exists");
  fs::create_directory(output);
  std::ofstream d(output/"host-abi.json"); auto previous=std::cout.rdbuf(d.rdbuf());
  descriptor_report_main(1,nullptr); std::cout.rdbuf(previous);
  stock::QuantParams q{};
  std::ofstream f(output/"quantparams.json");
  f<<"{\"scope\":\"installed_typed_quantparams\",\"type\":";
  megartx_probe::type<stock::QuantParams>(f); f<<",\"fields\":{";
  bool first=true;
#define QFIELD(x) if(!first)f<<',';first=false;megartx_probe::name(f,#x);f<<':';megartx_probe::member(f,q,q.x)
  QFIELD(fp4); QFIELD(fp4.fc1); QFIELD(fp4.fc2);
  QFIELD(fp4.fc1.use_per_expert_act_scale); QFIELD(fp4.fc1.act_global_scale);
  QFIELD(fp4.fc1.weight_block_scale); QFIELD(fp4.fc1.global_scale);
  QFIELD(fp4.fc2.use_per_expert_act_scale); QFIELD(fp4.fc2.act_global_scale);
  QFIELD(fp4.fc2.weight_block_scale); QFIELD(fp4.fc2.global_scale);
#undef QFIELD
  f<<"}}\n";
}
void stock_prepare(Bridge& bridge, Inputs& in, Outputs& out, cudaStream_t stream, bool enable_pdl=false) {
  auto b=out.bind(in);
  if(!bridge.maps(b.ids,b.sorted_to_slot,b.slot_to_sorted,b.offsets,1,128,8,0,128,enable_pdl,stream))
    throw std::runtime_error("stock fused map unexpectedly declined");
  check(cudaGetLastError());
  // The stock expansion only checks this weight-scale pointer for presence. It is
  // fixture-owned SF storage, never a resident weight table, and is not dereferenced.
  auto qp=stock::QuantParams::FP4(reinterpret_cast<float const*>(in.scale.data()),in.sf.data(),nullptr,
                                 nullptr,nullptr,nullptr,false,false);
  bridge.expand(reinterpret_cast<__nv_fp4_e2m1 const*>(b.aq),reinterpret_cast<__nv_fp4_e2m1*>(b.expanded_aq),
    reinterpret_cast<float const*>(b.weight_bits),reinterpret_cast<float*>(b.permuted_weight_bits),
    b.sorted_to_slot,nullptr,1,2816,8,128,qp,false,b.offsets,b.expanded_sf,b.sf,true,
    nullptr,nullptr,nullptr,nullptr,enable_pdl,stream);
  check(cudaGetLastError());
}
stock::QuantParams fixture_quant(Inputs const& in) {
  return stock::QuantParams::FP4(reinterpret_cast<float const*>(in.scale.data()),in.sf.data(),nullptr,
                                nullptr,nullptr,nullptr,false,false);
}
int rejection_checks(Inputs& in, Outputs& out, cudaStream_t stream) {
  auto qp=fixture_quant(in);InstalledPreparationCall c{out.bind(in),qp,stream};
  c.correction_runner_active=true;
  auto before=out.snapshot();int checks=0;
  auto rejects=[&](InstalledPreparationCall const& unsupported) {
    if(candidate_eligible(unsupported,true) || out.snapshot()!=before)
      throw std::runtime_error("unsupported candidate request touched output");
    ++checks;
  };
  for(int m:{2,4,8}){auto x=c;x.tokens=m;rejects(x);}
  {auto x=c;x.hidden=2815;rejects(x);}
  {auto x=c;x.experts=127;rejects(x);}
  {auto x=c;x.top_k=7;rejects(x);}
  {auto x=c;x.tp_size=2;rejects(x);}
  {auto x=c;x.ep_size=2;rejects(x);}
  {auto x=c;x.swizzled_input_sf=false;rejects(x);}
  {auto x=c;x.enable_pdl=true;rejects(x);}
  {auto x=c;x.correction_runner_active=false;rejects(x);}
  {auto x=c;x.min_latency=true;rejects(x);}
  {auto x=c;x.lora=true;rejects(x);}
  {auto x=c;x.groupwise=true;rejects(x);}
  {auto x=c;x.all_to_all=true;rejects(x);}
  {auto x=c;x.buffers.aq+=1;rejects(x);}
  {auto x=c;x.buffers.expanded_sf=nullptr;rejects(x);}
  {auto x=c;x.buffers.expanded_aq=const_cast<unsigned char*>(x.buffers.aq);rejects(x);}
  {auto x=c;x.buffers.expanded_sf=in.sf.data();rejects(x);} // Actual allocation is too short.
  auto original=in.ids.snapshot();
  for(int bad:{-1,128,0}) {
    auto raw=original;
    if(bad==0)std::copy(raw.begin()+4,raw.begin()+8,raw.begin());
    else std::memcpy(raw.data(),&bad,4);
    in.ids.upload(raw.data());check(cudaStreamSynchronize(stream));rejects(c);
  }
  in.ids.upload(original.data());check(cudaStreamSynchronize(stream));
  return checks;
}
int run(int argc,char** argv) {
  if(argc==3 && std::string(argv[1])=="abi") {abi(argv[2]);return 0;}
  if(argc!=5 || std::string(argv[1])!="capture") throw std::runtime_error("usage: binary abi NEW_OUTPUT | capture MODULE FIXTURES NEW_OUTPUT");
  fs::path fixtures=argv[3],output=argv[4];
  if(fs::exists(output))throw std::runtime_error("capture output exists");
  fs::create_directory(output); Bridge bridge(argv[2]);
  check(cudaSetDevice(0));headroom();cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  try {
    Inputs in(stream); Outputs incumbent(stream),candidate(stream);headroom();
    for(int i=0;i<3;++i) {
      auto name="case"+std::to_string(i);auto raw=read(fixtures/(name+".input"),24000);
      std::array<int,8> ids;std::memcpy(ids.data(),raw.data(),32);
      for(int j=0;j<8;++j)if(ids[j]<0||ids[j]>=128||std::count(ids.begin(),ids.end(),ids[j])!=1)
        throw std::runtime_error("invalid fixture route");
      in.upload(raw);
      auto before=incumbent.sf.snapshot(true);
      if(before!=candidate.sf.snapshot(true)) throw std::runtime_error("preparation scratch differs before launch");
      write(output/(name+".stock-before"),before);
      auto qp=fixture_quant(in);
      std::array<unsigned char,sizeof(qp)> original_quant;
      std::memcpy(original_quant.data(),&qp,sizeof(qp));
      InstalledPreparationCall stock_call{incumbent.bind(in),qp,stream};
      stock_call.correction_runner_active=i!=1;
      int stock_calls=0;
      auto fallback=[&]{++stock_calls;stock_prepare(bridge,in,incumbent,stream,stock_call.enable_pdl);};
      // Case 0: default disabled. Case 1: opt-in, unsupported correction context.
      // Case 2: PDL is unsupported for the candidate; preserve the stock callback.
      if(i==2)stock_call.enable_pdl=true;
      auto incumbent_backend=prepare(stock_call,fallback,i!=0);
      InstalledPreparationCall candidate_call{candidate.bind(in),qp,stream};
      candidate_call.correction_runner_active=true;
      auto candidate_backend=prepare(candidate_call,[&]{throw std::runtime_error("qualified fixture unexpectedly fell back");},true);
      if(incumbent_backend!=PreparationBackend::Stock || stock_calls!=1 || candidate_backend!=PreparationBackend::Fused)
        throw std::runtime_error("preparation selection/fallback mismatch");
      check(cudaStreamSynchronize(stream));
      if(std::memcmp(original_quant.data(),&qp,sizeof(qp)))throw std::runtime_error("QuantParams mutated");
      auto a=incumbent.snapshot(),b=candidate.snapshot(),after=in.snapshot();
      a.insert(a.end(),after.begin(),after.end());b.insert(b.end(),after.begin(),after.end());
      // Preserve each observation even if comparison later reports a mismatch.
      write(output/(name+".stock"),a);write(output/(name+".fused"),b);
    }
    auto rejection_count=rejection_checks(in,candidate,stream);
    std::ofstream owners(output/"owners.json");
    owners<<"{\"scope\":\"fixture_owned_disjoint_allocations\",\"scratch_bytes\":"<<scratch<<",\"allocations\":[";
    int id=0;for(auto p:{&in.ids,&in.weights,&in.aq,&in.sf,&in.scale,&incumbent.rank,&incumbent.sorted,&incumbent.offsets,
      &incumbent.aq,&incumbent.weights,&incumbent.sf,&candidate.rank,&candidate.sorted,&candidate.offsets,&candidate.aq,&candidate.weights,&candidate.sf}) {
      if(id)owners<<',';owners<<"{\"owner\":\"alloc_"<<id++<<"\",\"bytes\":"<<p->size+64<<",\"origin\":32,\"capacity\":"<<p->size
        <<",\"alignment\":32,\"alignment_verified\":"<<(reinterpret_cast<uintptr_t>(p->storage)%256==0?"true":"false")<<"}";
    }
    owners<<"],\"stream\":\"one_owned_nonblocking_stream\",\"reused_scratch_cases\":3,\"tma_or_gemm_calls\":0,"
      <<"\"native_opt_in_tested\":true,\"stock_fallback_cases\":3,\"candidate_cases\":3,\"rejection_checks\":"<<rejection_count
      <<",\"quantparams_unchanged\":true}\n";
    check(cudaStreamSynchronize(stream));
  } catch(...) {cudaStreamSynchronize(stream);cudaStreamDestroy(stream);throw;}
  check(cudaStreamDestroy(stream));return 0;
}
int main(int argc,char** argv) {
  try{return run(argc,argv);}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
