// Compile only against explicitly captured installed headers and build flags.
// Host-only metadata probe: no kernel, CUDA runtime call, device access or weights.
#include "m1_probe_wire.hpp"
#include "moe_gemm_kernels.h"
#include <iostream>

using G = tensorrt_llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput;
using B = G::NVFP4BlockScaledConfig;

void layouts(char const* stage, int n, int k, bool swap) {
  auto normal = cute::make_shape(1, n, k, 1);
  auto transposed = cute::make_shape(n, 1, k, 1);
  auto act = swap ? B::tile_atom_to_shape_SFB(transposed) : B::tile_atom_to_shape_SFA(normal);
  auto weight = swap ? B::tile_atom_to_shape_SFA(transposed) : B::tile_atom_to_shape_SFB(normal);
  auto& out = std::cout;
  out << "{\"stage\":\"" << stage << "\",\"n\":" << n << ",\"k\":" << k
      << ",\"swap_ab\":" << (swap ? "true" : "false") << ",\"act_offsets\":[";
  for (int b = 0; b < k / 16; ++b) {
    if (b) out << ',';
    out << act(cute::make_coord(0, b * 16, 0));
  }
  out << "],\"weight_tile_offsets\":[";
  for (int row = 0; row < 128; ++row)
    for (int b = 0; b < k / 16; ++b) {
      if (row || b) out << ',';
      out << weight(cute::make_coord(row, b * 16, 0));
    }
  out << "],\"weight_last_offsets\":[";
  for (int b = 0; b < k / 16; ++b) {
    if (b) out << ',';
    out << weight(cute::make_coord(n - 1, b * 16, 0));
  }
  out << "]}";
}

int main(int argc, char**) {
  if (argc != 1) { std::cerr << "This fixed M1 host probe takes no arguments.\n"; return 2; }
  G object{};
  auto& out = std::cout;
  out << "{\"scope\":\"typed_host_m1_layout_samples\",\"little_endian\":"
      << (megartx_probe::little_endian() ? "true" : "false") << ",\"types\":{";
#define TYPE(key, value) out << "\"" key "\":"; megartx_probe::type<value>(out)
  TYPE("descriptor", G); out << ',';
  TYPE("problem", G::ProblemShape::UnderlyingProblemShape); out << ',';
  TYPE("stride_a", G::StrideA); out << ',';
  TYPE("stride_b", G::StrideB); out << ',';
  TYPE("sf_layout", B::LayoutSF); out << ',';
  TYPE("element_sf", G::ElementSF);
#undef TYPE
  out << "},\"fields\":{";
  bool first = true;
#define FIELD(value) \
  if (!first) out << ','; first = false; \
  megartx_probe::name(out, #value); out << ':'; megartx_probe::member(out, object, object.value)
  FIELD(swap_ab); FIELD(shape_info); FIELD(stride_act); FIELD(stride_weight);
  FIELD(ptr_act); FIELD(ptr_weight); FIELD(stride_c); FIELD(ptr_c); FIELD(stride_d); FIELD(ptr_d);
  FIELD(fusion); FIELD(alpha_scale_ptr_array); FIELD(fpX_block_scaling_factors_act);
  FIELD(fpX_block_scaling_factors_weight); FIELD(fpX_block_scaling_factors_stride_act);
  FIELD(fpX_block_scaling_factors_stride_weight); FIELD(fpX_block_scaling_type);
  FIELD(int4_groupwise_params); FIELD(gemm_workspace); FIELD(gemm_workspace_size);
  FIELD(precomputed_scheduler_workspace); FIELD(precomputed_scheduler_workspace_size);
  FIELD(precomputed_scheduler_total_routed_tokens); FIELD(enable_pdl);
  FIELD(fused_finalize_epilogue.ptr_final_output);
  FIELD(fused_finalize_epilogue.stride_final_output_transposed);
  FIELD(fused_finalize_epilogue.stride_final_output);
  FIELD(fused_finalize_epilogue.ptr_bias); FIELD(fused_finalize_epilogue.ptr_router_scales);
  FIELD(fused_finalize_epilogue.ptr_source_token_index);
  FIELD(fused_finalize_epilogue.num_rows_in_final_output);
  FIELD(fused_finalize_epilogue.shape_override); FIELD(fused_finalize_epilogue.use_reduction);
#undef FIELD
  out << "},\"scale_layouts\":[";
  layouts("fc1", 1408, 2816, false); out << ',';
  layouts("fc1", 1408, 2816, true); out << ',';
  layouts("fc2", 2816, 704, false); out << ',';
  layouts("fc2", 2816, 704, true);
  out << "],\"workspace_layout\":null,\"consumer_masks\":null}\n";
}
