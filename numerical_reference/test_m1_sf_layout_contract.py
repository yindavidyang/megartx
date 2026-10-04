"""Compile the exact host proof with counted CuTE APIs, without CUDA."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from m1_sf_layout_proof import verify_host_layouts
from m1_preparation_reference import sf_coordinate


MOCK_LAYOUT = r'''#pragma once
#include <array>
#include <cstddef>
#include <tuple>
namespace cute {
inline int calls=0,bad_config=0;
struct Layout {
  std::array<int,3> shape_{128,2816,1},stride_{16,4,512};
  int irrelevant_padding=0;
  auto& shape(){return shape_;}auto const& shape()const{return shape_;}
  auto& stride(){return stride_;}auto const& stride()const{return stride_;}
  int operator()(std::array<int,3> c)const {
    ++calls;int r=c[0],b=c[1]/16;
    int offset=(b/4)*stride_[2]+(r%32)*stride_[0]+(r/32)*stride_[1]+b%4;
    return bad_config==1?0:bad_config==2?128*(shape_[1]/16):offset;
  }
};
inline auto make_coord(int r,int k,int l){return std::array<int,3>{r,k,l};}
inline auto make_shape(int m,int n,int k,int l){return std::array<int,4>{m,n,k,l};}
template<int I>inline int size(Layout const& a){return a.shape_[I];}
inline int cosize(Layout const& a){return 128*(a.shape_[1]/16);}
struct Config {
  using LayoutSF=Layout;
  static LayoutSF tile_atom_to_shape_SFA(std::array<int,4> s){return {{128,s[2],s[3]},{16,4,512}};}
};
}
'''
TEST = r'''
#include "m1_sf_layout_contract.hpp"
#include <cassert>
template<class F>void rejects(F&& f) {
  bool failed=false;try{f();}catch(std::runtime_error const&){failed=true;}assert(failed);
}
int main() {
  using Proof=megartx::experimental::M1SfLayoutContract<cute::Config>;
  Proof proof;assert(cute::calls==28160);cute::calls=0;
  for(int i=0;i<120;++i)for(int stage=0;stage<2;++stage)for(int expert=0;expert<8;++expert) {
    auto actual=cute::Config::tile_atom_to_shape_SFA(cute::make_shape(1,stage?2816:1408,stage?704:2816,1));
    actual.irrelevant_padding=expert;proof.validate(actual,stage,false);
  }
  assert(cute::calls==0); // Production branch never evaluates a coordinate.
  for(int stage=0;stage<2;++stage) {
    auto actual=cute::Config::tile_atom_to_shape_SFA(cute::make_shape(1,stage?2816:1408,stage?704:2816,1));
    int before=cute::calls;proof.validate(actual,stage,true);
    assert(cute::calls-before==128*((stage?704:2816)/16));
    for(int leaf=0;leaf<3;++leaf) {
      auto changed=actual;++changed.shape_[leaf];
      rejects([&]{proof.validate(changed,stage,false);});
      rejects([&]{proof.validate(changed,stage,true);});
      changed=actual;++changed.stride_[leaf];
      rejects([&]{proof.validate(changed,stage,false);});
      rejects([&]{proof.validate(changed,stage,true);});
    }
    // Reusing the same object address after mutation cannot reuse eligibility.
    ++actual.stride_[2];rejects([&]{proof.validate(actual,stage,false);});
    --actual.stride_[2];proof.validate(actual,stage,false);
    rejects([&]{proof.validate(actual,1-stage,false);});
    rejects([&]{proof.validate(actual,-1,false);});
    rejects([&]{proof.validate(actual,2,false);});
  }
  for(int failure=1;failure<3;++failure) {
    cute::bad_config=failure;rejects([&]{Proof bad;});
  }
}
'''


class SfLayoutContractTests(unittest.TestCase):
    def test_exact_bridge_delta_preserves_every_other_native_check_and_call(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/"probes/m1_live_bridge.cu").read_text()
        edits=[('#include "../kernels/m1_sf_layout_contract.hpp"\n',''),
            ('mx::M1SfLayoutContract<Desc::NVFP4BlockScaledConfig> const& sf_layout_contract() {\n'
             '  static const mx::M1SfLayoutContract<Desc::NVFP4BlockScaledConfig> proof;\n'
             '  return proof;\n}\n',''),
            ('    // Prove the fixed source layouts once before any admitted device call.\n'
             '    // Initialization errors are handled by this same C ABI error boundary.\n'
             '    sf_layout_contract();\n',''),
            ('  // copies and fences stay fresh. Only the source-invariant dense-layout proof\n'
             '  // is reused; captured/observer diagnostics also enumerate each actual layout.\n',
             '  // copies, fences and layout enumeration are validation, not optional evidence.\n'),
            ('      // Compare every actual semantic shape/stride tuple to the independently\n'
             '      // proven source layout, before using its dense-carrier conclusion. Never\n'
             '      // cache an expert pointer, route prefix, owner or returned descriptor.\n'
             '      sf_layout_contract().validate(layouts[e],stage,lease.capture_enabled || lease.observer);\n',
             '      std::vector<unsigned char> seen(bytes,0);\n'
             '      for(int row=0;row<128;++row)for(int kk=0;kk<k;kk+=16) {\n'
             '        auto offset=size_t(layouts[e](cute::make_coord(row,kk,0)));\n'
             '        require(offset<bytes,"actual SF layout address escapes its carrier");\n'
             '        seen[offset]=1;\n'
             '      }\n'
             '      require(std::all_of(seen.begin(),seen.end(),[](auto v){return v==1;}),\n'
             '              "actual physical SF TMA domain is not the pinned dense carrier");\n')]
        def normalize(value):
            for current,previous in edits:
                self.assertEqual(value.count(current),1)
                value=value.replace(current,previous)
            return hashlib.sha256(value.encode()).hexdigest()
        # Exact e886b98 source; changing its hash requires a separate reviewed
        # scope. Blind updates of the new current-vector ledger cannot admit
        # an unrelated removed check or altered native arithmetic/dispatch.
        prior="428cef5dd354994141ee73e155b92bafd4b8a61195c918f7908792ed34411e0a"
        self.assertEqual(normalize(source),prior)
        self.assertNotEqual(normalize(source.replace("rows!=1 || hidden!=2816","rows!=2 || hidden!=2816")),prior)

    def test_exact_compiled_header_avoids_production_enumeration_and_rejects_drift(self):
        compiler = shutil.which("c++") or shutil.which("g++")
        if not compiler:
            self.skipTest("CPU C++ compiler unavailable")
        root = Path(__file__).resolve().parents[1]
        header = (root/"kernels/m1_sf_layout_contract.hpp").read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work/"cute").mkdir()
            (work/"cute/layout.hpp").write_text(MOCK_LAYOUT)
            (work/"m1_sf_layout_contract.hpp").write_bytes(header)
            self.assertEqual((work/"m1_sf_layout_contract.hpp").read_bytes(), header)
            (work/"test.cpp").write_text(TEST)
            result = subprocess.run([compiler,"-std=c++17","-O2","-Wall","-Wextra","-I",str(work),
                                     str(work/"test.cpp"),"-o",str(work/"test")],
                                    capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            result = subprocess.run([str(work/"test")],capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_native_boundary_retains_zero_row_route_range_and_readbacks(self):
        root = Path(__file__).resolve().parents[1]
        source = (root/"probes/m1_live_bridge.cu").read_text()
        descriptor = source[source.index("using Shape=Desc::ProblemShape"):]
        self.assertEqual(descriptor.count("cudaMemcpyAsync("),7)
        self.assertEqual(descriptor.count("cudaStreamSynchronize("),1)
        self.assertLess(descriptor.index("if(!token_rows)continue;"),
                        descriptor.index("sf_layout_contract().validate("))
        self.assertIn("sf_layout_contract().validate(layouts[e],stage,lease.capture_enabled || lease.observer);",descriptor)
        # Exercise the byte-identical production contains() function with
        # symbolic addresses, including endpoint, short range and wrap cases.
        contains = re.search(r"bool contains\(.*?\n\}",source,re.S)[0]
        compiler = shutil.which("c++") or shutil.which("g++")
        if not compiler:
            self.skipTest("CPU C++ compiler unavailable")
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work/"contains.cpp").write_text('#include <cstdint>\n#include <cstddef>\n#include <cassert>\n#include <initializer_list>\n'
                'struct View {void* pointer;uint64_t bytes;void* storage;uint64_t storage_bytes;};\n'+contains+
                '\nint main(){for(uint64_t bytes: {22528,5632}) {View v{reinterpret_cast<void*>(4096),bytes,nullptr,0};'
                'assert(contains(v,v.pointer,bytes));assert(!contains(v,v.pointer,bytes+1));'
                'assert(contains(v,reinterpret_cast<void*>(4096+bytes),0));'
                'assert(!contains(v,reinterpret_cast<void*>(4096+bytes),1));'
                'assert(!contains(v,reinterpret_cast<void*>(4095),1));'
                'assert(!contains(v,reinterpret_cast<void*>(UINTPTR_MAX),1));'
                '--v.bytes;assert(!contains(v,v.pointer,bytes));}}\n')
            result = subprocess.run([compiler,"-std=c++17",str(work/"contains.cpp"),"-o",str(work/"contains")],
                                    capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            subprocess.run([str(work/"contains")],check=True,capture_output=True,timeout=5)

    def test_independent_exhaustive_offsets_padding_and_expert_boundaries(self):
        proof = {"scope":"host_only_pinned_cutlass_m1_sf_layout", "dynamic_leaf_drift_controls":20,
                 "stage_drift_controls":6, "stages":[
                     {"k":k,"rows":128,"bytes":128*(k//16),"inactive_rows":0,"inactive_bytes":512,
                      "offsets":[sf_coordinate(row,block,k//16) for row in range(128) for block in range(k//16)]}
                     for k in (2816,704)]}
        report = verify_host_layouts(proof)
        self.assertEqual(report["source_coordinates_avoided_per_120_call_lane"],27033600)
        self.assertEqual(report["initial_proof_coordinates"],28160)
        for stage in proof["stages"]:
            original=stage["offsets"][0];stage["offsets"][0]=stage["offsets"][1]
            with self.assertRaisesRegex(ValueError,"offsets"):
                verify_host_layouts(proof)
            stage["offsets"][0]=original
        host = os.environ.get("MEGARTX_M1_SF_HOST_PROOF")
        if host:
            actual=verify_host_layouts(json.loads(Path(host).read_text()))
            self.assertEqual([s["offsets_sha256"] for s in actual["stages"]],
                             [s["offsets_sha256"] for s in report["stages"]])


if __name__ == "__main__":
    unittest.main()
