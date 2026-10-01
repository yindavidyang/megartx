"""Synthetic corruption tests; these fixtures cannot attest an installed producer."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import m1_installed_compare as checker
import m1_kernel_fixture as fixture
from test_m1_abi_probe import host_fixture


def quant_fixture():
    fields = {"fp4": {"offset":192,"size":64,"align":8},
              "fp4.fc1": {"offset":192,"size":32,"align":8},
              "fp4.fc2": {"offset":224,"size":32,"align":8}}
    for stage, base in (("fc1",192), ("fc2",224)):
        for name, relative, size, align in (("use_per_expert_act_scale",0,1,1),
                ("act_global_scale",8,8,8), ("weight_block_scale",16,8,8), ("global_scale",24,8,8)):
            fields[f"fp4.{stage}.{name}"]={"offset":base+relative,"size":size,"align":align}
    return {"scope":"installed_typed_quantparams","type":{"size":344,"align":8},"fields":fields}


def owner_fixture():
    sizes = (32,32,1408,22528,4)+(32,32,1032,11264,32,2883584)*2
    return {"scope":"fixture_owned_disjoint_allocations","scratch_bytes":sum(n+64 for n in sizes),
            "allocations":[{"owner":f"alloc_{i}","bytes":n+64,"origin":32,"capacity":n,
                            "alignment":32,"alignment_verified":True} for i,n in enumerate(sizes)],
            "stream":"one_owned_nonblocking_stream","reused_scratch_cases":3,"tma_or_gemm_calls":0,
            "native_opt_in_tested":True,"stock_fallback_cases":3,"candidate_cases":3,
            "rejection_checks":22,"quantparams_unchanged":True}


class InstalledCompareTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root=Path(temporary.name)
        self.fixtures,self.captures,self.abi=root/"fixtures",root/"captures",root/"abi"
        fixture.create(self.fixtures);self.captures.mkdir();self.abi.mkdir()
        before=b"\xA7"*32+b"\xD3"*2883584+b"\xB6"*32
        for i in range(3):
            expected=(self.fixtures/f"case{i}.expected").read_bytes()
            raw=(self.fixtures/f"case{i}.input").read_bytes()
            (self.captures/f"case{i}.stock-before").write_bytes(before)
            for backend in ("stock","fused"):
                (self.captures/f"case{i}.{backend}").write_bytes(expected+raw)
            before=expected[12392:]
        (self.abi/"host-abi.json").write_text(json.dumps(host_fixture()))
        (self.abi/"quantparams.json").write_text(json.dumps(quant_fixture()))
        (self.captures/"owners.json").write_text(json.dumps(owner_fixture()))

    def compare(self):
        return checker.compare(self.fixtures,self.captures,self.abi)

    def test_synthetic_agreement_never_unlocks_runtime_or_producer_attestation(self):
        result=self.compare()
        self.assertTrue(result["byte_equal"])
        self.assertTrue(result["all_conservative_mask_bytes_compared"])
        self.assertTrue(result["native_opt_in_tested"])
        self.assertEqual([c["useful_sf_bytes"] for c in result["cases"]],[1408]*3)
        for key in ("producer_identity_verified","runtime_workspace_owners_verified",
                    "physical_gemm_consumer_masks_verified","candidate_selectable"):
            self.assertFalse(result[key])

    def test_each_backend_checks_maps_aq_weight_sf_padding_guards_and_inputs(self):
        for backend in ("stock","fused"):
            path=self.captures/f"case2.{backend}"
            original=path.read_bytes()
            for offset in (0,32,64,1096,12360,12392,12428,fixture.OUTPUT_BYTES-1,fixture.OUTPUT_BYTES+1500):
                with self.subTest(backend=backend,offset=offset):
                    data=bytearray(original);data[offset]^=1;path.write_bytes(data)
                    with self.assertRaisesRegex(ValueError,"differs from independent oracle"):
                        self.compare()
                    path.write_bytes(original)

    def test_stale_scratch_snapshot_and_route_mutation_are_rejected(self):
        p=self.captures/"case2.stock-before";original=p.read_bytes()
        data=bytearray(original);data[32]^=1;p.write_bytes(data)
        with self.assertRaisesRegex(ValueError,"scratch continuity"):
            self.compare()
        p.write_bytes(original)
        p=self.fixtures/"case0.input";data=bytearray(p.read_bytes());data[:4]=data[4:8];p.write_bytes(data)
        with self.assertRaisesRegex(ValueError,"route changed"):
            self.compare()

    def test_opaque_quantparams_fields_and_nested_bounds_are_not_guessed(self):
        q=quant_fixture();checker.check_quantparams(q)
        for change in (lambda x:x["fields"]["fp4.fc1.act_global_scale"].update(offset=100),
                       lambda x:x["fields"]["fp4.fc2.global_scale"].update(size=65536),
                       lambda x:x["fields"].update(guessed_field={}),
                       lambda x:x["type"].update(size=True)):
            value=copy.deepcopy(q);change(value)
            with self.assertRaises(ValueError):checker.check_quantparams(value)

    def test_owner_extent_guards_alignment_and_total_are_exact(self):
        for change in (lambda x:x["allocations"][10].update(origin=0),
                       lambda x:x["allocations"][16].update(capacity=1),
                       lambda x:x["allocations"][2].update(alignment_verified=1),
                       lambda x:x.update(scratch_bytes=8<<20),
                       lambda x:x.update(tma_or_gemm_calls=True),
                       lambda x:x.update(candidate_selectable=True)):
            value=owner_fixture();change(value)
            with self.assertRaises(ValueError):checker.check_owners(value)

    def test_metadata_and_capture_links_truncation_and_duplicate_keys_are_rejected(self):
        p=self.captures/"case0.stock";original=p.read_bytes();p.write_bytes(original[:-1])
        with self.assertRaises(ValueError):self.compare()
        p.unlink();p.symlink_to(self.captures/"case0.fused")
        with self.assertRaises(ValueError):self.compare()
        p.unlink();p.write_bytes(original)
        p=self.abi/"quantparams.json";p.write_text('{"scope":"a","scope":"b"}')
        with self.assertRaisesRegex(ValueError,"duplicate JSON key"):self.compare()
        p.unlink();p.symlink_to(self.abi/"host-abi.json")
        with self.assertRaisesRegex(ValueError,"symlinks"):self.compare()


if __name__=="__main__":
    unittest.main()
