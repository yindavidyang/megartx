"""Compile the byte-identical adapter with mocked APIs; test prepare(), without CUDA."""
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


MOCK_CUDA = r'''#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
using cudaStream_t=void*;
enum cudaError_t { cudaSuccess=0, cudaErrorInvalidValue=1, cudaErrorLaunchFailure=719 };
enum cudaMemoryType { cudaMemoryTypeHost=1, cudaMemoryTypeDevice=2 };
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone=0, cudaStreamCaptureStatusActive=1 };
enum cudaMemcpyKind { cudaMemcpyDeviceToHost=2 };
struct cudaPointerAttributes { cudaMemoryType type;int device; };
using CUdeviceptr=std::uintptr_t;using CUcontext=void*;using CUstream=void*;
enum CUresult { CUDA_SUCCESS=0, CUDA_ERROR_INVALID_VALUE=1, CUDA_ERROR_LAUNCH_FAILED=719 };
char const* cudaGetErrorString(cudaError_t);
cudaError_t cudaStreamIsCapturing(cudaStream_t,cudaStreamCaptureStatus*);
cudaError_t cudaGetDevice(int*);
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes*,void const*);
cudaError_t cudaMemcpyAsync(void*,void const*,std::size_t,cudaMemcpyKind,cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
CUresult cuGetErrorString(CUresult,char const**);
CUresult cuCtxGetCurrent(CUcontext*);
CUresult cuStreamGetCtx(CUstream,CUcontext*);
CUresult cuMemGetAddressRange(CUdeviceptr*,std::size_t*,CUdeviceptr);
'''
MOCK_QUANT = r'''#pragma once
namespace tensorrt_llm::kernels::cutlass_kernels {
struct QuantParams {
  struct Inputs {struct Gemm {bool use_per_expert_act_scale=false;unsigned char const* weight_block_scale=nullptr;};Gemm fc1,fc2;} fp4;
};
}
'''


class PreparationControlFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=shutil.which("c++") or shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("CPU C++ compiler unavailable")
        cls.temporary=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temporary.cleanup)
        cls.work=Path(cls.temporary.name);repo=Path(__file__).resolve().parents[1]
        cls.header=(repo/"kernels/m1_installed_preparation.cuh").read_bytes()
        (cls.work/"m1_installed_preparation.cuh").write_bytes(cls.header)
        (cls.work/"mock_cuda.hpp").write_text(MOCK_CUDA)
        (cls.work/"cuda.h").write_text('#pragma once\n#include "mock_cuda.hpp"\n')
        (cls.work/"moe_kernels.h").write_text(MOCK_QUANT)
        # Keep the ordinary buffer declaration identical to the production kernel.
        kernel=(repo/"kernels/m1_maps_expand.cuh").read_text()
        buffers=re.search(r"struct M1Buffers \{.*?\n\};",kernel,re.S)
        if not buffers:raise AssertionError("production buffer declaration missing")
        (cls.work/"m1_maps_expand.cuh").write_text(
            '#pragma once\n#include "mock_cuda.hpp"\nnamespace megartx::experimental {\n'
            'constexpr int kHidden=2816,kExperts=128,kTopK=8,kAQBytes=1408,kSFBlocks=176,kSFBytes=2883584;\n'
            +buffers[0]+'\ninline bool aligned(void const* p,int n){return p && reinterpret_cast<std::uintptr_t>(p)%n==0;}\n'
            'cudaError_t launch(M1Buffers,cudaStream_t,bool);\n}\n')
        cls.source=repo/"probes/m1_preparation_flow_test.cpp";cls.binary=cls.work/"prepare-flow"
        result=subprocess.run([compiler,"-std=c++17","-O2","-Wall","-Wextra","-I",str(cls.work),
                               str(cls.source),"-o",str(cls.binary)],capture_output=True,text=True,timeout=30)
        if result.returncode:raise AssertionError(result.stderr)

    def run_case(self,*args):
        result=subprocess.run([str(self.binary),*args],capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_compiled_adapter_header_is_byte_identical(self):
        self.assertEqual(hashlib.sha256((self.work/"m1_installed_preparation.cuh").read_bytes()).digest(),
                         hashlib.sha256(self.header).digest())

    def test_runtime_query_errors_propagate_before_incumbent_or_output_mutation(self):
        for api in ("cudaStreamIsCapturing","cudaGetDevice","cudaMemcpyAsync","cudaStreamSynchronize"):
            with self.subTest(api=api):self.run_case("fault",api,"1")
        for occurrence in range(1,11):
            with self.subTest(api="cudaPointerGetAttributes",occurrence=occurrence):
                self.run_case("fault","cudaPointerGetAttributes",str(occurrence))
        self.run_case("fault","cudaPointerGetAttributes","1","invalid_value")

    def test_driver_query_errors_propagate_before_incumbent_or_output_mutation(self):
        for api in ("cuCtxGetCurrent","cuStreamGetCtx"):
            with self.subTest(api=api):self.run_case("fault",api,"1")
        for occurrence in range(1,11):
            with self.subTest(api="cuMemGetAddressRange",occurrence=occurrence):
                self.run_case("fault","cuMemGetAddressRange",str(occurrence))
        self.run_case("fault","cuMemGetAddressRange","1","invalid_value")
        self.run_case("fault","cuCtxGetCurrent","1","diagnostic_failure")

    def test_candidate_submission_error_never_attempts_fallback(self):
        self.run_case("fault","launch","1")

    def test_supported_and_unsupported_prepare_controls(self):
        for control in ("supported","disabled","geometry","capture","context","host_pointer",
                        "short_range","alias","duplicate","incumbent_failure",
                        "per_expert_unverified","per_expert_copy"):
            with self.subTest(control=control):self.run_case(control)

    def test_lean_supported_and_unsupported_controls_check_current_state_once(self):
        for control in ("supported","stock_supported","changed_after_previous","disabled","geometry",
                        "capture","context","host_pointer","short_range","alias","duplicate",
                        "incumbent_failure","per_expert_unverified","per_expert_copy"):
            with self.subTest(control=control):self.run_case("lean",control)

    def test_lean_runtime_and_driver_errors_never_fallback_or_retry(self):
        for api in ("cudaStreamIsCapturing","cudaGetDevice","cudaMemcpyAsync","cudaStreamSynchronize",
                    "cuCtxGetCurrent","cuStreamGetCtx","launch"):
            with self.subTest(api=api):self.run_case("lean","fault",api,"1")
        for api in ("cudaPointerGetAttributes","cuMemGetAddressRange"):
            for occurrence in range(1,11):
                with self.subTest(api=api,occurrence=occurrence):
                    self.run_case("lean","fault",api,str(occurrence))
            self.run_case("lean","fault",api,"1","invalid_value")
        self.run_case("lean","fault","cuCtxGetCurrent","1","diagnostic_failure")


if __name__=="__main__":
    unittest.main()
