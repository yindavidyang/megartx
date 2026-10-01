"""Synthetic capture corruption and host-wire tests; never target qualification."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

import m1_abi_probe as probe
import m1_preparation_reference as oracle
from test_m1_preparation_reference import context, row


def host_fixture():
    samples = []
    for stage, n, k, swap in (("fc1", 1408, 2816, False), ("fc1", 1408, 2816, True),
                               ("fc2", 2816, 704, False), ("fc2", 2816, 704, True)):
        blocks = k // 16
        samples.append({"stage": stage, "n": n, "k": k, "swap_ab": swap,
            "act_offsets": [oracle.sf_coordinate(0, b, blocks) for b in range(blocks)],
            "weight_tile_offsets": [oracle.sf_coordinate(r, b, blocks) for r in range(128) for b in range(blocks)],
            "weight_last_offsets": [oracle.sf_coordinate(n - 1, b, blocks) for b in range(blocks)]})
    return {"scope": "typed_host_m1_layout_samples", "little_endian": True,
            "types": {name: {"size": 512 if name == "descriptor" else 8, "align": 8}
                      for name in ("descriptor", "problem", "stride_a", "stride_b", "sf_layout")}
                     | {"element_sf": {"size": 1, "align": 1}},
            "fields": {name: {"offset": i * 8, "size": 8, "align": 8}
                       for i, name in enumerate(probe.HOST_FIELDS)},
            "scale_layouts": samples, "workspace_layout": None, "consumer_masks": None}


def build_bundle(root, ids=(127, 0, 82, 42, 126, 7, 89, 12), *, index=0, before=None):
    input_row, a, b = row(ids), context("fc1"), context("fc2", True, "finalize")
    prepared = oracle.prepare(input_row, a, b, reference_abi=oracle.REFERENCE_ABI)
    before = before or b"\xA7" * 32 + b"\xD3" * probe.SF_BYTES + b"\xB6" * 32
    raw = {"ids": struct.pack("<8i", *ids), "weights": input_row.route_weight_bits,
           "aq": input_row.packed_fp4, "sf": input_row.swizzled_sf,
           "fc1_shapes": prepared.fc1.problem_shapes, "fc2_shapes": prepared.fc2.problem_shapes,
           "sf_before": before, "sf_after": oracle.materialize_sf(prepared, before, origin=32)}
    raw["input_after"] = raw["ids"] + raw["weights"] + raw["aq"] + raw["sf"]
    raw.update({name: getattr(prepared, name) for name in (
        "slot_to_sorted", "sorted_to_slot", "expert_offsets", "expanded_aq", "permuted_weight_bits")})
    case = {"fc1": probe.wire(a), "fc2": probe.wire(b),
            "descriptors": {"fc1": probe.wire(prepared.fc1.active_descriptors),
                            "fc2": probe.wire(prepared.fc2.active_descriptors)}, "files": {}}
    (root / f"case{index}").mkdir()
    for name, data in raw.items():
        relative = f"case{index}/{name}.bin"
        (root / relative).write_bytes(data)
        case["files"][name] = {"path": relative, "sha256": probe.digest(data), "bytes": len(data)}
    return case, prepared, raw["sf_after"]


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "producer"
        self.root.mkdir()
        case, self.expected, self.after = build_bundle(self.root)
        self.packet = {"schema": probe.SCHEMA, "geometry": dict(probe.GEOMETRY), "origin": "synthetic",
                       "source_audit": None, "host_probe": None,
                       "bindings": dict.fromkeys(probe.BINDINGS), "owners": None, "cases": [case]}
        self.save()

    def save(self):
        (self.root / "capture.json").write_text(json.dumps(self.packet))

    def refresh(self, key, data):
        record = self.packet["cases"][0]["files"][key]
        (self.root / record["path"]).write_bytes(data)
        record.update(sha256=probe.digest(data), bytes=len(data))
        self.save()

    def test_complete_cpu_fixture_is_not_installed_abi_or_candidate_receipt(self):
        result, payloads = probe.validate_bundle(self.root)
        self.assertTrue(result["cpu_semantic_match"])
        self.assertFalse(result["installed_abi_verified"])
        self.assertFalse(result["candidate_selectable"])
        self.assertIn("typed_host_probe_missing", result["unsupported"])
        self.assertLess(result["captured_payload_bytes"], 6 << 20)
        self.assertEqual(set(payloads), {"capture.json"} | {f"case0/{name}.bin" for name in probe.FILE_SIZES})

    def test_capture_copies_only_hash_bound_artifacts_exclusively(self):
        (self.root / "private-unrelated.txt").write_text("do not publish")
        destination = Path(self.temp.name) / "capture"
        result = probe.capture(self.root, destination)
        self.assertTrue(result["cpu_semantic_match"])
        self.assertFalse((destination / "private-unrelated.txt").exists())
        probe.validate_bundle(destination)
        with self.assertRaises(FileExistsError):
            probe.capture(self.root, destination)
        with self.assertRaises(ValueError):
            probe.capture(self.root, self.root / "nested")

    def test_optional_host_owners_and_digest_assertions_remain_unqualified(self):
        host = json.dumps(host_fixture()).encode()
        (self.root / "host-abi.json").write_bytes(host)
        self.packet["host_probe"] = {"path": "host-abi.json", "sha256": probe.digest(host), "bytes": len(host)}
        self.packet["bindings"] = dict.fromkeys(probe.BINDINGS, "0" * 64)
        self.packet["origin"] = "target_incumbent"
        self.packet["owners"] = {
            role: {"allocation": f"alloc_{index}", "offset": 0, "capacity": size,
                   "allocation_extent": size, "alignment": 16, "lifetime": [0, 6]}
            for index, (role, size) in enumerate(probe.role_views(self.expected).items())}
        self.save()
        result, payloads = probe.validate_bundle(self.root)
        self.assertIn("host-abi.json", payloads)
        self.assertNotIn("typed_host_probe_missing", result["unsupported"])
        self.assertNotIn("workspace_owner_registry_missing", result["unsupported"])
        self.assertIn("installed_probe_build_vs_loaded_module_equivalence_unverified", result["unsupported"])
        self.assertFalse(result["installed_abi_verified"])
        self.assertFalse(result["candidate_selectable"])
        destination = Path(self.temp.name) / "with-host"
        probe.capture(self.root, destination)
        self.assertEqual((destination / "host-abi.json").read_bytes(), host)
        (self.root / "host-abi.json").write_bytes(host + b" ")
        with self.assertRaisesRegex(ValueError, "host probe digest/size"):
            probe.validate_bundle(self.root)

    def test_repeat_and_disjoint_three_case_fixture_and_same_context(self):
        case, _, after = build_bundle(self.root, index=1, before=self.after)
        self.packet["cases"].append(case)
        case, _, _ = build_bundle(self.root, tuple(range(20, 28)), index=2, before=after)
        self.packet["cases"].append(case)
        self.save()
        result, _ = probe.validate_bundle(self.root)
        self.assertTrue(result["route_stress_present"])
        self.assertEqual(result["cases"], 3)
        self.assertLess(result["captured_payload_bytes"], 18 << 20)
        before_record = self.packet["cases"][1]["files"]["sf_before"]
        after_record = self.packet["cases"][1]["files"]["sf_after"]
        before, after = (self.root / before_record["path"]).read_bytes(), (self.root / after_record["path"]).read_bytes()
        for record, original in ((before_record, before), (after_record, after)):
            changed = bytearray(original)
            changed[0] ^= 1  # Independently valid guards, but a different allocation history.
            (self.root / record["path"]).write_bytes(changed)
            record["sha256"] = probe.digest(changed)
        self.save()
        with self.assertRaisesRegex(ValueError, "continue from the previous"):
            probe.validate_bundle(self.root)
        for record, original in ((before_record, before), (after_record, after)):
            (self.root / record["path"]).write_bytes(original)
            record["sha256"] = probe.digest(original)
        self.packet["cases"][1]["fc1"]["swap_ab"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "same configured"):
            probe.validate_bundle(self.root)

    def test_changed_digest_and_truncated_file_rejected(self):
        path = self.root / "case0/aq.bin"
        path.write_bytes(bytes(1408))
        with self.assertRaisesRegex(ValueError, "digest/size"):
            probe.validate_bundle(self.root)
        self.refresh("aq", bytes(1407))
        with self.assertRaisesRegex(ValueError, "physical byte extent"):
            probe.validate_bundle(self.root)

    def test_refreshed_hash_does_not_hide_wrong_maps_packing_sf_or_stale_shape(self):
        for key, index in (("slot_to_sorted", 0), ("expanded_aq", 1),
                           ("sf_after", 32), ("sf_after", 32 + 4),
                           ("sf_after", 0), ("fc1_shapes", 24), ("input_after", 100)):
            path = self.root / f"case0/{key}.bin"
            original = path.read_bytes()
            bad = bytearray(original)
            bad[index] ^= 1
            self.refresh(key, bytes(bad))
            with self.subTest(key=key, index=index), self.assertRaises((ValueError, oracle.PreparationMismatch)):
                probe.validate_bundle(self.root)
            self.refresh(key, original)

    def test_duplicate_bad_ids_and_bool_geometry_rejected(self):
        self.refresh("ids", struct.pack("<8i", *([0] * 8)))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            probe.validate_bundle(self.root)
        self.packet["geometry"]["m"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "only M1"):
            probe.validate_bundle(self.root)

    def test_small_m_and_extra_cases_have_no_inferred_support(self):
        for m in (2, 4, 8):
            self.packet["geometry"]["m"] = m
            self.save()
            with self.subTest(m=m), self.assertRaisesRegex(ValueError, "only M1"):
                probe.validate_bundle(self.root)
        self.packet["geometry"]["m"] = 1
        self.packet["cases"] *= 4
        self.save()
        with self.assertRaisesRegex(ValueError, "one to three"):
            probe.validate_bundle(self.root)

    def test_descriptor_bool_pointer_owner_and_global_substitution_rejected(self):
        original = copy.deepcopy(self.packet)
        desc = self.packet["cases"][0]["descriptors"]["fc1"][0]
        for key, value in (("expert", False), ("activation", {"address": "0x1000"}),
                           ("weight_globals", desc["weight_globals"][:1])):
            self.packet = copy.deepcopy(original)
            self.packet["cases"][0]["descriptors"]["fc1"][0][key] = value
            self.save()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "descriptor mismatch"):
                probe.validate_bundle(self.root)

    def test_unknown_qualification_keys_and_synthetic_relabel_cannot_unlock(self):
        self.packet["origin"] = "target_incumbent"
        self.save()
        result, _ = probe.validate_bundle(self.root)
        self.assertFalse(result["candidate_selectable"])
        self.assertFalse(result["installed_abi_verified"])
        self.packet["gpu_verified"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            probe.validate_bundle(self.root)

    def test_traversal_absolute_paths_and_symlinks_rejected(self):
        record = self.packet["cases"][0]["files"]["aq"]
        for bad in ("../aq.bin", "/tmp/aq.bin", "case0/../aq.bin"):
            record["path"] = bad
            self.save()
            with self.subTest(path=bad), self.assertRaises(ValueError):
                probe.validate_bundle(self.root)
        path = self.root / "case0/aq.bin"
        path.unlink()
        path.symlink_to(self.root / "case0/expanded_aq.bin")
        record["path"] = "case0/aq.bin"
        self.save()
        with self.assertRaisesRegex(ValueError, "symlinks"):
            probe.validate_bundle(self.root)

    def test_json_duplicates_nonfinite_oversize_and_unknown_keys_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'{"x":1.0}',
                    b" " * (probe.MAX_JSON + 1)):
            with self.subTest(size=len(raw)), self.assertRaises(ValueError):
                probe.decode_json(raw)
        self.packet["cases"][0]["files"]["weights_payload"] = {}
        self.save()
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            probe.validate_bundle(self.root)

    def test_nonregular_payload_rejected_without_opening_a_blocking_fifo(self):
        path = self.root / "case0/aq.bin"
        path.unlink()
        os.mkfifo(path)
        with self.assertRaisesRegex(ValueError, "bounded regular file"):
            probe.validate_bundle(self.root)

    def test_cli_cpu_only_and_default_refusal_are_distinct(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(probe.main(["validate", str(self.root), "--cpu-only"]), 0)
        self.assertFalse(json.loads(output.getvalue())["candidate_selectable"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(probe.main(["validate", str(self.root)]), 2)

    def test_machine_readable_request_has_caps_and_no_target_authorization(self):
        request = probe.fixture_request()
        self.assertEqual(request["geometry"]["m"], 1)
        self.assertEqual(request["raw_bytes_per_case"], sum(probe.FILE_SIZES.values()))
        self.assertFalse(request["gpu_run_authorized_by_this_file"])
        self.assertTrue(all(value is None for value in request["bindings"].values()))
        output = Path(self.temp.name) / "request.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(probe.main(["request", "--output", str(output)]), 0)
            self.assertEqual(probe.main(["request", "--output", str(output)]), 1)
        self.assertEqual(json.loads(output.read_bytes()), request)


class HostLayoutAndOwnershipTests(unittest.TestCase):
    def test_all_stage_swap_coordinates_and_typed_extent_checks(self):
        fixture = host_fixture()
        probe.check_host_probe(fixture)
        for mutate in (
            lambda x: x["types"]["descriptor"].update(size=8),
            lambda x: x["types"]["sf_layout"].update(align=3),
            lambda x: x["fields"]["swap_ab"].update(offset=512),
            lambda x: x["scale_layouts"][2]["act_offsets"].__setitem__(0, 1),
            lambda x: x["scale_layouts"][3]["weight_last_offsets"].__setitem__(0, 0),
            lambda x: x["scale_layouts"][0].update(n=1408.0),
            lambda x: x.update(workspace_layout={"guessed": True}),
        ):
            bad = copy.deepcopy(fixture)
            mutate(bad)
            with self.assertRaises(ValueError):
                probe.check_host_probe(bad)

    def test_aliased_sf_requires_disjoint_declared_lifetimes(self):
        roles = {"fc1.activation_sf": 2883584, "fc2.activation_sf": 720896}
        owners = {role: {"allocation": "alloc_0", "offset": 32, "capacity": capacity,
                         "allocation_extent": 2883648, "alignment": 16,
                         "lifetime": [1, 3] if role.startswith("fc1") else [4, 5]}
                  for role, capacity in roles.items()}
        probe.check_owners(owners, roles)
        for changes in ({"lifetime": [3, 5]}, {"allocation_extent": 2883584},
                        {"alignment": 3}, {"offset": -1}, {"allocation": "0xdeadbeef"}):
            bad = copy.deepcopy(owners)
            bad["fc2.activation_sf"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                probe.check_owners(bad, roles)

    def test_source_collection_is_bounded_read_only_and_mismatch_is_unsupported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            roots = {name: root / name for name in ("flashinfer", "cutlass")}
            for package, path, _ in probe.REFERENCE_FILES:
                file = roots[package] / path
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"synthetic source, not installed")
            audit = probe.source_audit(roots)
            self.assertFalse(probe.check_source_audit(audit))
            self.assertFalse(audit["installed_abi_verified"])
            audit["reference_source_match"] = True
            with self.assertRaises(ValueError):
                probe.check_source_audit(audit)
            file = roots["cutlass"] / probe.REFERENCE_FILES[-1][1]
            file.write_bytes(b"x" * (probe.MAX_FILE + 1))
            with self.assertRaisesRegex(ValueError, "bounded regular"):
                probe.source_audit(roots)

    def test_native_wire_helpers_compile_and_measure_cpu_members_without_addresses(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("system C++ compiler unavailable; native target headers remain untested")
        include = Path(__file__).resolve().parents[1] / "probes"
        code = '''#include "m1_probe_wire.hpp"
#include <iostream>
struct S { char small; std::int64_t large; };
int main() { S obj{}; std::cout << "{\\\"type\\\":";
  megartx_probe::type<S>(std::cout); std::cout << ",\\\"member\\\":";
  megartx_probe::member(std::cout, obj, obj.large); std::cout << "}";
  try { megartx_probe::name(std::cout, "bad/name"); return 2; }
  catch (std::invalid_argument const&) {} }
'''
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "wire.cpp"
            source.write_text(code)
            subprocess.run([compiler, "-std=c++17", "-I", str(include), str(source), "-o", str(root / "wire")],
                           check=True, capture_output=True, timeout=30)
            result = subprocess.run([str(root / "wire")], check=True, capture_output=True, timeout=5)
        obj = json.loads(result.stdout)
        self.assertLessEqual(obj["member"]["offset"] + obj["member"]["size"], obj["type"]["size"])
        self.assertEqual(obj["member"]["size"], 8)
        self.assertNotIn(b"0x", result.stdout)


if __name__ == "__main__":
    unittest.main()
