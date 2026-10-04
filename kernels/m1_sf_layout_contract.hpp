// Host-only proof for the two source-bound M1 NVFP4 SF carrier layouts.
// No allocation, owner, route, pointer, native-output or eligibility cache.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include "cute/layout.hpp"

namespace megartx::experimental {
template <class Layout>
inline void require_dense_m1_sf_layout(Layout const& layout, int k) {
  auto bytes=std::size_t(cute::cosize(layout));
  if((k!=2816 && k!=704) || cute::size<0>(layout)!=128 ||
      cute::size<1>(layout)!=k || bytes!=128*std::size_t(k/16))
    throw std::runtime_error("unqualified actual physical TMA SF domain");
  std::vector<unsigned char> seen(bytes,0);
  for(int row=0;row<128;++row)for(int kk=0;kk<k;kk+=16) {
    auto offset=std::size_t(layout(cute::make_coord(row,kk,0)));
    if(offset>=bytes)
      throw std::runtime_error("actual SF layout address escapes its carrier");
    seen[offset]=1;
  }
  if(!std::all_of(seen.begin(),seen.end(),[](auto v){return v==1;}))
    throw std::runtime_error("actual physical SF TMA domain is not the pinned dense carrier");
}

// Config is the exact installed Desc::NVFP4BlockScaledConfig, not a learned
// layout. The proof depends only on pinned source constants. A newly compiled
// or loaded bridge has its own immutable proof; there is no model generation
// or workspace lifetime to invalidate. Actual returned layouts stay fresh.
template <class Config>
class M1SfLayoutContract {
  using Layout=typename Config::LayoutSF;
  std::array<Layout,2> layouts_;
public:
  M1SfLayoutContract():layouts_{
      Config::tile_atom_to_shape_SFA(cute::make_shape(1,1408,2816,1)),
      Config::tile_atom_to_shape_SFA(cute::make_shape(1,2816,704,1))} {
    require_dense_m1_sf_layout(layouts_[0],2816);
    require_dense_m1_sf_layout(layouts_[1],704);
  }
  void require_match(Layout const& actual, int stage) const {
    if(stage<0 || stage>=2)
      throw std::runtime_error("invalid M1 SF layout stage");
    // Typed semantic tuples include every nested shape/stride value. Object
    // padding and pointer equality cannot establish this identity.
    if(!(actual.shape()==layouts_[stage].shape() &&
         actual.stride()==layouts_[stage].stride()))
      throw std::runtime_error("actual SF layout differs from source-derived M1 contract");
  }
  void validate(Layout const& actual, int stage, bool diagnostic) const {
    require_match(actual,stage);
    if(diagnostic)require_dense_m1_sf_layout(actual,stage?704:2816);
  }
};
}  // namespace megartx::experimental
