// Installed-only typed launcher binding. This does not include/rebuild stock kernels.
#pragma once
#include "moe_kernels.h"
#include <dlfcn.h>
#include <stdexcept>
#include <type_traits>

namespace tensorrt_llm::kernels::cutlass_kernels {
// Declarations transcribed from the hash-pinned installed implementation.
bool fusedBuildExpertMapsSortFirstToken(int const*, int*, int*, int64_t*, int64_t,
                                      int, int, int, int, bool, cudaStream_t);
template <class Input, class Expanded>
void expandInputRowsKernelLauncher(Input const*, Expanded*, float const*, float*,
    int const*, int const*, int64_t, int64_t, int, int, QuantParams const&, bool,
    int64_t*, TmaWarpSpecializedGroupedGemmInput::ElementSF*,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const*, bool, void const*, float*,
    float const*, float const**, bool, cudaStream_t);
}

namespace megartx::installed_probe {
namespace stock = tensorrt_llm::kernels::cutlass_kernels;
using Map = decltype(&stock::fusedBuildExpertMapsSortFirstToken);
using Expand = decltype(&stock::expandInputRowsKernelLauncher<__nv_fp4_e2m1, __nv_fp4_e2m1>);
inline constexpr char map_symbol[] =
    "_ZN12tensorrt_llm7kernels15cutlass_kernels34fusedBuildExpertMapsSortFirstTokenEPKiPiS4_PlliiiibP11CUstream_st";
inline constexpr char expand_symbol[] =
    "_ZN12tensorrt_llm7kernels15cutlass_kernels29expandInputRowsKernelLauncherI13__nv_fp4_e2m1S3_EEvPKT_PT0_PKfPfPKiSD_lliiRKNS1_11QuantParamsEbPlPhPKhbPKvSB_SA_PSA_bP11CUstream_st";
struct Bridge {
  void* module;
  Map maps;
  Expand expand;
  explicit Bridge(char const* path) : module(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
    if (!module) throw std::runtime_error(dlerror());
    maps = reinterpret_cast<Map>(dlsym(module, map_symbol));
    expand = reinterpret_cast<Expand>(dlsym(module, expand_symbol));
    if (!maps || !expand) throw std::runtime_error("installed preparation symbols missing");
    Dl_info a{}, b{};
    if (!dladdr(reinterpret_cast<void*>(maps), &a) ||
        !dladdr(reinterpret_cast<void*>(expand), &b) || a.dli_fbase != b.dli_fbase)
      throw std::runtime_error("launchers do not belong to the same loaded module");
  }
  ~Bridge() { if (module) dlclose(module); }
  Bridge(Bridge const&) = delete;
};
}  // namespace megartx::installed_probe
