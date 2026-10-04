// Host compiler only, with read-only installed CUTLASS/CUDA headers.
// No CUDA calls, library loads, device work, model data or native descriptors.
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "../kernels/m1_sf_layout_contract.hpp"
#include <iostream>

using Config=cutlass::detail::Sm1xxBlockScaledConfig<16>;
using Layout=Config::LayoutSF;
using Contract=megartx::experimental::M1SfLayoutContract<Config>;

template <class F>
void rejects(F&& f) {
  try { f(); } catch(std::runtime_error const&) { return; }
  throw std::runtime_error("layout drift accepted");
}

int main() {
  Contract proof;
  std::cout<<"{\"scope\":\"host_only_pinned_cutlass_m1_sf_layout\",\"stages\":[";
  for(int stage=0;stage<2;++stage) {
    int n=stage?2816:1408,k=stage?704:2816;
    Layout actual=Config::tile_atom_to_shape_SFA(cute::make_shape(1,n,k,1));
    proof.validate(actual,stage,false);
    proof.validate(actual,stage,true);
    // LayoutSF has exactly these dynamic semantic leaves. Its other nested
    // fields are type-enforced constants; object padding is deliberately ignored.
    for(int leaf=0;leaf<5;++leaf) {
      auto changed=actual;
      switch(leaf) {
        case 0: ++cute::get<0,1>(changed.shape());break;
        case 1: ++cute::get<1,1>(changed.shape());break;
        case 2: ++cute::get<2,1>(changed.shape());break;
        case 3: ++cute::get<0,1>(changed.stride());break;
        case 4: ++cute::get<2,1>(changed.stride());break;
      }
      rejects([&]{proof.validate(changed,stage,false);});
      rejects([&]{proof.validate(changed,stage,true);});
    }
    rejects([&]{proof.validate(actual,1-stage,false);});
    rejects([&]{proof.validate(actual,-1,false);});
    rejects([&]{proof.validate(actual,2,false);});
    // Zero-row groups are still read back. Their metadata may have a nonzero
    // atom cosize, but no active payload validation/proof is inferred from it.
    auto inactive=Config::tile_atom_to_shape_SFA(cute::make_shape(0,n,k,1));
    if(cute::size<0>(inactive)!=0)
      throw std::runtime_error("zero-row source layout changed");
    if(stage)std::cout<<',';
    std::cout<<"{\"k\":"<<k<<",\"rows\":"<<cute::size<0>(actual)
        <<",\"bytes\":"<<cute::cosize(actual)<<",\"inactive_rows\":"<<cute::size<0>(inactive)
        <<",\"inactive_bytes\":"<<cute::cosize(inactive)<<",\"offsets\":[";
    for(int row=0;row<128;++row)for(int block=0;block<k/16;++block) {
      if(row || block)std::cout<<',';
      std::cout<<actual(cute::make_coord(row,block*16,0));
    }
    std::cout<<"]}";
  }
  std::cout<<"],\"dynamic_leaf_drift_controls\":20,\"stage_drift_controls\":6}\n";
}
