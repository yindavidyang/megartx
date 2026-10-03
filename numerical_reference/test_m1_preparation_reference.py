"""Synthetic adversarial fixtures; no model capture, candidate, or GPU imports."""

from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch

import m1_preparation_reference as ref


def bits(value):
    return struct.pack("<f", value)


def context(stage, swap=False, fusion="none"):
    tables = tuple(b"".join(bits(1 + expert / 256 + projection / 4)
                           for expert in range(ref.E))
                   for projection in range(2 if stage == "fc1" else 1))
    return ref.StageContext(stage, swap, fusion,
                            b"".join(bits(2 + expert / 128) for expert in range(ref.E)),
                            bits(0.5), tables, "synthetic independent original globals")


def row(ids=(127, 0, 82, 42, 126, 7, 89, 12), *, weight_bits=None, aq=None, codes=None):
    # Invalid scale bytes deliberately fill all unused input padding.
    physical = bytearray([0xCD]) * (128 * ref.H // 16)
    codes = codes if codes is not None else bytes(range(127)) + b"\x80" + bytes(range(48))
    for block, code in enumerate(codes):
        # Independent row-zero layout, with four useful bytes per 512-byte tile.
        physical[(block // 4) * 512 + block % 4] = code
    aq = aq if aq is not None else bytes(range(256)) * 5 + bytes(range(128))
    weights = weight_bits if weight_bits is not None else b"".join(bits(slot - 3.5) for slot in range(8))
    return ref.InputRow(tuple(ids), weights, aq, bytes(physical))


class M1PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fc1, cls.fc2 = context("fc1"), context("fc2", True, "finalize")

    def prepare(self, input_row=None, fc1=None, fc2=None):
        return ref.prepare(input_row or row(), fc1 or self.fc1, fc2 or self.fc2,
                           reference_abi=ref.REFERENCE_ABI)

    def check_maps(self, input_row, prepared):
        ranks = struct.unpack("<8i", prepared.slot_to_sorted)
        inverse = struct.unpack("<8i", prepared.sorted_to_slot)
        offsets = struct.unpack("<129q", prepared.expert_offsets)
        self.assertEqual(sorted(ranks), list(range(8)))
        self.assertEqual(sorted(inverse), list(range(8)))
        for slot, expert in enumerate(input_row.selected_ids):
            self.assertEqual(ranks[slot], sum(other < expert for other in input_row.selected_ids))
            self.assertEqual(inverse[ranks[slot]], slot)
            self.assertEqual(offsets[expert + 1] - offsets[expert], 1)
        self.assertEqual(offsets, tuple(sum(expert < boundary for expert in input_row.selected_ids)
                                        for boundary in range(129)))
        self.assertEqual((offsets[0], offsets[-1]), (0, 8))

    def test_literal_unsorted_low_high_maps(self):
        prepared = self.prepare()
        self.assertEqual(struct.unpack("<8i", prepared.slot_to_sorted), (7, 0, 4, 3, 6, 1, 5, 2))
        self.assertEqual(struct.unpack("<8i", prepared.sorted_to_slot), (1, 5, 7, 3, 2, 6, 4, 0))
        self.check_maps(row(), prepared)

    def test_every_expert_in_every_route_slot_including_all_six_corrections(self):
        # The affected IDs 42,82,126,89,7,12 are all included, in all eight slots.
        for expert, slot in itertools.product(range(128), range(8)):
            others = [(expert + delta) % 128 for delta in range(1, 8)]
            others.insert(slot, expert)
            input_row = row(others)
            with self.subTest(expert=expert, slot=slot):
                self.check_maps(input_row, self.prepare(input_row))

    def test_all_fp4_byte_patterns_are_replicated_without_decoding(self):
        input_row, prepared = row(), self.prepare()
        self.assertEqual(set(input_row.packed_fp4), set(range(256)))
        self.assertEqual(prepared.expanded_aq, input_row.packed_fp4 * 8)
        self.assertEqual(len(prepared.expanded_aq), 11264)
        zero = self.prepare(row(aq=bytes(1408)))
        self.assertEqual(zero.expanded_aq, bytes(11264))

    def test_signed_zero_equal_negative_and_subnormal_route_weights_are_bitwise(self):
        words = (0, 0x80000000, 0x3E000000, 0x3E000000,
                 0xBE000000, 1, 0x80000001, 0x7F7FFFFF)
        input_row = row(weight_bits=struct.pack("<8I", *words))
        prepared = self.prepare(input_row)
        self.assertEqual(struct.unpack("<8I", prepared.permuted_weight_bits),
                         tuple(words[slot] for slot in (1, 5, 7, 3, 2, 6, 4, 0)))

    def test_literal_swizzle_and_group_base_boundaries(self):
        for coordinates, expected in (((0, 0, 176), 0), ((0, 4, 176), 512),
                                      ((31, 3, 176), 499), ((32, 0, 176), 4),
                                      ((64, 0, 176), 8), ((127, 175, 176), 22527),
                                      ((128, 0, 176), 22528)):
            self.assertEqual(ref.sf_coordinate(*coordinates), expected)
        self.assertEqual(ref.grouped_sf_base(0, 0, 2816), 0)
        self.assertEqual(ref.grouped_sf_base(1, 0, 2816), 22528)
        self.assertEqual(ref.grouped_sf_base(2, 1, 2816), 45056)
        self.assertEqual(ref.grouped_sf_base(128, 8, 2816), 2883584)
        self.assertEqual(ref.grouped_sf_base(128, 8, 704), 720896)

    def test_all_sf_coordinates_bijective_and_match_committed_format_oracle(self):
        from nvfp4_reference import sf_offset_128x4
        for blocks in (176, 44):
            addresses = set()
            for r, b in itertools.product(range(256), range(blocks)):
                address = ref.sf_coordinate(r, b, blocks)
                self.assertEqual(address, sf_offset_128x4(r, b, blocks))
                addresses.add(address)
            self.assertEqual(addresses, set(range(256 * blocks)))

    def test_valid_sf_bytes_exact_and_all_padding_guards_sources_unchanged(self):
        input_row, prepared = row(), self.prepare()
        source_before = (input_row.packed_fp4, input_row.swizzled_sf, input_row.route_weight_bits)
        capacity = 2883584
        before = b"\xA7" * 32 + b"\xD3" * capacity + b"\xB6" * 32
        after = ref.materialize_sf(prepared, before, origin=32)
        ref.require_sf_storage(prepared, before, after, origin=32)
        changed, touched = set(), set()
        for write in prepared.expanded_sf_writes:
            self.assertEqual(write.ref.owner, "fc1.activation_sf")
            for offset, code in enumerate(write.data, write.ref.byte_offset):
                self.assertNotIn(offset, touched)
                touched.add(offset)
                self.assertEqual(after[32 + offset], code)
                if code != before[32 + offset]:
                    changed.add(32 + offset)
        self.assertEqual(len(touched), 1408)
        self.assertEqual({code for write in prepared.expanded_sf_writes for code in write.data},
                         set(range(127)) | {128})
        self.assertEqual(sum(a != b for a, b in zip(before, after)), len(changed))
        self.assertEqual((input_row.packed_fp4, input_row.swizzled_sf, input_row.route_weight_bits), source_before)
        for index in (0, 31, 32 + capacity, len(after) - 1, 32 + 4):
            bad = bytearray(after)
            bad[index] ^= 1
            with self.subTest(index=index), self.assertRaises(ref.PreparationMismatch):
                ref.require_sf_storage(prepared, before, bytes(bad), origin=32)

    def test_zero_scales_copy_without_touching_input_padding(self):
        prepared = self.prepare(row(codes=bytes(176)))
        self.assertEqual(sum(len(write.data) for write in prepared.expanded_sf_writes), 1408)
        self.assertTrue(all(write.data == bytes(4) for write in prepared.expanded_sf_writes))

    def test_independent_swap_flags_all_problem_shapes_and_symbolic_bounds(self):
        for swap1, swap2 in itertools.product((False, True), repeat=2):
            prepared = self.prepare(fc1=context("fc1", swap1), fc2=context("fc2", swap2))
            for stage, n, k, swapped in ((prepared.fc1, 1408, 2816, swap1),
                                         (prepared.fc2, 2816, 704, swap2)):
                shapes = struct.unpack("<384q", stage.problem_shapes)
                self.assertEqual(len(stage.active_descriptors), 8)
                for expert in range(128):
                    count = int(expert in row().selected_ids)
                    expected = (n, count, k) if swapped else (count, n, k)
                    self.assertEqual(shapes[expert * 3:expert * 3 + 3], expected)
                for descriptor in stage.active_descriptors:
                    self.assertEqual(descriptor.logical_mnk, (1, n, k))
                    rank = sum(e < descriptor.expert for e in row().selected_ids)
                    self.assertEqual(descriptor.activation.byte_offset, rank * k // 2)
                    self.assertEqual(descriptor.weight.byte_offset, descriptor.expert * n * k // 2)
                    self.assertEqual(descriptor.weight_sf.byte_offset, descriptor.expert * n * k // 16)
                    self.assertEqual(descriptor.activation_mk_strides, (k, 1))
                    self.assertEqual(descriptor.weight_kn_strides, (1, k))
                    self.assertEqual(descriptor.output.byte_offset, rank * n * 2)
                    for view in (descriptor.activation, descriptor.weight, descriptor.activation_sf,
                                 descriptor.weight_sf, descriptor.output, descriptor.alpha.ref):
                        self.assertLessEqual(view.byte_offset + view.extent, view.capacity)

    def test_finalize_map_weights_and_separate_global_bindings(self):
        prepared = self.prepare()
        for descriptor in prepared.fc2.active_descriptors:
            rank = sum(e < descriptor.expert for e in row().selected_ids)
            self.assertIsNone(descriptor.output)
            self.assertEqual(descriptor.finalize_map, ref.BufferRef("sorted_to_slot", rank * 4, 4, 32))
            self.assertEqual(descriptor.finalize_weights, ref.BufferRef("permuted_weight_bits", rank * 4, 4, 32))
        for descriptor in prepared.fc1.active_descriptors:
            gate, up = descriptor.weight_globals
            self.assertNotEqual(gate.bits, up.bits)
            self.assertEqual((gate.ref.owner, up.ref.owner), ("fc1.gate_global", "fc1.up_global"))
            self.assertEqual(descriptor.alpha.ref.byte_offset, descriptor.expert * 4)
            self.assertEqual(descriptor.activation_global.bits, bits(0.5))

    def test_repeated_and_rapid_disjoint_routes_clear_every_problem(self):
        route_sequence = [(127, 126, 125, 124, 123, 122, 121, 120), tuple(range(8))] * 12
        previous = None
        storage = b"\xA7" * 32 + b"\xD3" * 2883584 + b"\xB6" * 32
        for ids in route_sequence + [tuple(range(8))] * 4:
            prepared = self.prepare(row(ids))
            self.check_maps(row(ids), prepared)
            updated = ref.materialize_sf(prepared, storage, origin=32)
            ref.require_sf_storage(prepared, storage, updated, origin=32)
            storage = updated
            for stage in (prepared.fc1, prepared.fc2):
                self.assertEqual({desc.expert for desc in stage.active_descriptors}, set(ids))
            if previous and set(ids).isdisjoint(previous):
                values = struct.unpack("<384q", prepared.fc1.problem_shapes)
                self.assertTrue(all(values[expert * 3] == 0 for expert in previous))
                values = struct.unpack("<384q", prepared.fc2.problem_shapes)
                self.assertTrue(all(values[expert * 3 + 1] == 0 for expert in previous))
            previous = ids

    def test_seeded_route_permutations_preserve_id_weight_correspondence(self):
        rng = random.Random(2816128)
        for _ in range(64):
            ids = tuple(rng.sample(range(128), 8))
            input_row = row(ids)
            prepared = self.prepare(input_row)
            self.check_maps(input_row, prepared)
            for rank, slot in enumerate(sorted(range(8), key=ids.__getitem__)):
                self.assertEqual(prepared.permuted_weight_bits[rank * 4:rank * 4 + 4],
                                 input_row.route_weight_bits[slot * 4:slot * 4 + 4])

    def test_byte_and_descriptor_negative_controls_each_fail(self):
        prepared = self.prepare()
        weights = bytearray(prepared.permuted_weight_bits)
        weights[:4], weights[4:8] = weights[4:8], weights[:4]
        packed = bytearray(prepared.expanded_aq)
        packed[1] = (packed[1] >> 4) | ((packed[1] & 15) << 4)
        sf = list(prepared.expanded_sf_writes)
        sf[0] = replace(sf[0], data=bytes([sf[0].data[0] ^ 1]) + sf[0].data[1:])
        shapes = list(struct.unpack("<384q", prepared.fc1.problem_shapes))
        shapes[1 * 3] = 1  # Expert 1 is inactive, but this stale descriptor claims M=1.
        bad_stage = replace(prepared.fc1, problem_shapes=struct.pack("<384q", *shapes))
        descriptors = list(prepared.fc1.active_descriptors)
        descriptor = descriptors[-1]
        compact = 7 * 128 * 176  # Wrong: uses sorted rank in place of expert/prefix base.
        descriptors[-1] = replace(descriptor, activation_sf=replace(descriptor.activation_sf, byte_offset=compact))
        bad_base_stage = replace(prepared.fc1, active_descriptors=tuple(descriptors))
        controls = {
            "route weight without ID": replace(prepared, permuted_weight_bits=bytes(weights)),
            "reversed nibble pair": replace(prepared, expanded_aq=bytes(packed)),
            "one SF byte": replace(prepared, expanded_sf_writes=tuple(sf)),
            "stale nonzero M": replace(prepared, fc1=bad_stage),
            "compact SF base": replace(prepared, fc1=bad_base_stage),
            "gate substituted for up": self.prepare(fc1=replace(self.fc1,
                weight_global_bits=(self.fc1.weight_global_bits[0],) * 2)),
        }
        ref.require_equal(prepared, prepared)
        for name, bad in controls.items():
            with self.subTest(control=name), self.assertRaises(ref.PreparationMismatch):
                ref.require_equal(prepared, bad)

    def test_invalid_ids_bytes_scales_and_nonfinite_weights_rejected(self):
        input_row = row()
        changes = [("selected_ids", (0,) * 8), ("selected_ids", tuple(range(7)) + (128,)),
                   ("selected_ids", tuple(range(7)) + (-1,)), ("selected_ids", (False,) + tuple(range(1, 8))),
                   ("packed_fp4", bytes(1407)), ("packed_fp4", bytearray(1408)),
                   ("swizzled_sf", bytes(176)), ("route_weight_bits", bytes(31))]
        for code in (127, 129, 255):
            changes.append(("swizzled_sf", bytes([code]) + input_row.swizzled_sf[1:]))
        for word in (0x7F800000, 0xFF800000, 0x7FC00001):
            changes.append(("route_weight_bits", struct.pack("<I", word) + input_row.route_weight_bits[4:]))
        for field, value in changes:
            with self.subTest(field=field, value_type=type(value)), self.assertRaises(ValueError):
                replace(input_row, **{field: value})

    def test_unknown_abi_fusion_swap_and_incomplete_global_contracts_rejected(self):
        for abi in ("installed", "0.6.18.post1", None, ""):
            with self.subTest(abi=abi), self.assertRaises(ref.UnsupportedReference):
                ref.prepare(row(), self.fc1, self.fc2, reference_abi=abi)
        for change in ({"swap_ab": None}, {"swap_ab": 1}, {"fusion": "gated_activation"},
                       {"weight_global_bits": (self.fc1.weight_global_bits[0],)},
                       {"alpha_bits": bytes(512)}, {"scalar_provenance": ""}):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                replace(self.fc1, **change)
        with self.assertRaises(ValueError):
            ref.prepare(row(), self.fc2, self.fc1, reference_abi=ref.REFERENCE_ABI)

    def test_symbolic_bounds_and_scale_geometry_are_enforced(self):
        for args in (("owner", 9, 2, 10), ("", 0, 1, 1), ("owner", -1, 1, 10),
                     ("owner", 0, True, 10)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                ref.BufferRef(*args)
        for args in ((-1, 0, 176), (0, 176, 176), (0, 0, 175)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                ref.sf_coordinate(*args)
        with self.assertRaises(ref.UnsupportedReference):
            ref.grouped_sf_base(1, 0, 2800)
        with self.assertRaises(ValueError):
            ref.materialize_sf(self.prepare(), bytes(2883583))
        with self.assertRaises(ref.UnsupportedReference):
            ref.materialize_sf(replace(self.prepare(), reference_abi="installed"), bytes(2883584))

    def test_oracle_import_is_stdlib_only(self):
        root = str(Path(__file__).resolve().parent)
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import m1_preparation_reference; "
                "assert not any(name in sys.modules for name in ('numpy','torch','vllm','flashinfer'))")
        subprocess.run([sys.executable, "-S", "-c", code, root], check=True, capture_output=True)

    def test_committed_source_pins_match_exact_base_evidence(self):
        root = Path(__file__).resolve().parents[1]
        historical = root / "docs/evidence/m1-preparation-source-pins.json"
        pins = json.loads(historical.read_text())
        self.assertFalse(pins["installed_binary_abi_verified"])
        self.assertFalse(pins["candidate_available"])
        self.assertEqual(pins["reference_semantic_abi"], ref.REFERENCE_ABI)
        overlay_path = root / "docs/evidence/m1-live-source-pins.json"
        overlay = json.loads(overlay_path.read_text()) if overlay_path.exists() else None
        replacements = {}
        if overlay is not None:
            self.assertEqual(overlay["historical_ledger_sha256"], hashlib.sha256(historical.read_bytes()).hexdigest())
            self.assertEqual(overlay["base_commit"], "955832939062ac6b9e7bb2698b4181c1472225c3")
            replacements = {r["path"]: r for r in overlay["superseded_inputs"]}
            self.assertEqual(set(replacements), {"src/megartx/vllm_scale_plugin.py"})
        capture_path = root / "docs/evidence/m1-capture-free-source-pins.json"
        capture_overlay = json.loads(capture_path.read_text())
        self.assertEqual(capture_overlay["base_commit"], "cc82e59c8053a72af37d9e06a76405d1ef31181e")
        self.assertEqual(capture_overlay["previous_ledger_sha256"], hashlib.sha256(overlay_path.read_bytes()).hexdigest())
        for field in ("gpu_execution_verified", "graphs_qualified", "performance_qualified"):
            self.assertIs(capture_overlay[field], False)
        self.assertEqual(set(capture_overlay["controller_source_hashes"]), {
            "m1_live.py", "vllm_scale_plugin.py", "m1_execution.py", "controlled_capture.py",
            "controlled_kv_capture.py"})

        normal_path = root / "docs/evidence/m1-normal-source-pins.json"
        normal_overlay = json.loads(normal_path.read_text())
        self.assertEqual(normal_overlay["base_commit"], "cc82e59c8053a72af37d9e06a76405d1ef31181e")
        self.assertEqual(normal_overlay["parent_ledger_sha256"], hashlib.sha256(overlay_path.read_bytes()).hexdigest())

        combined_path = root / "docs/evidence/m1-reconciliation-source-pins.json"
        combined = json.loads(combined_path.read_text())
        self.assertEqual(combined["base_commit"], "cc82e59c8053a72af37d9e06a76405d1ef31181e")
        self.assertEqual(combined["pr_heads"], {
            "pr11": "6e7674377a483fb404be93811a1b58eae0edf475",
            "pr12": "2a3b35c34f6bfdc5d52e0387f1bdd0da891d4d5a"})
        history = combined["historical_ledgers"]
        self.assertEqual(history["capture_free_sha256"], hashlib.sha256(capture_path.read_bytes()).hexdigest())
        self.assertEqual(history["normal_sha256"], hashlib.sha256(normal_path.read_bytes()).hexdigest())
        lifecycle_path = root / "docs/evidence/m1-process-lifecycle-source-pins.json"
        lifecycle = json.loads(lifecycle_path.read_text())
        self.assertEqual(lifecycle["parent_head"], "3ed6ee85bd2bf08e4a2d9d20f10352c45f208b32")
        self.assertEqual(lifecycle["parent_branch"], "integration/m1-capture-free-validation-bd09109")
        self.assertEqual(lifecycle["parent_tree"], "74d5e44771cba35d837e6245302c12e16a26e7e1")
        self.assertEqual(lifecycle["parent_ledger_sha256"], hashlib.sha256(combined_path.read_bytes()).hexdigest())
        self.assertEqual(lifecycle["public_validation_head"], "3ed6ee85bd2bf08e4a2d9d20f10352c45f208b32")
        self.assertEqual(lifecycle["public_validation_tree"], "74d5e44771cba35d837e6245302c12e16a26e7e1")
        self.assertEqual(lifecycle["base_alignment_tree"], lifecycle["public_validation_tree"])
        self.assertEqual(lifecycle["verified_pr_heads"], combined["pr_heads"])
        self.assertEqual(lifecycle["source_reconciliation"]["verified_combined_merge_commit"],
                         "d002a11c81a32987e4fe00b055f5658ab2f9b914")
        self.assertTrue(lifecycle["source_reconciliation"]["pr_heads_match_verified_merge_parents"])
        self.assertFalse(lifecycle["source_reconciliation"]["reconstructed_unverified_pr11_source_accepted"])
        self.assertEqual(lifecycle["failed_predecessor_snapshot"]["commit"],
                         "11ec001a9ebc4be7ab62af589b7b44e767a77d98")
        self.assertFalse(lifecycle["failed_predecessor_snapshot"]["accepted_as_base"])
        self.assertIs(lifecycle["native_build_verified"], False)
        self.assertIs(lifecycle["gpu_execution_verified"], False)
        lifecycle_changes = {r["path"]: r for r in lifecycle["superseded_parent_sources"]}
        for section in ("runtime_source_hashes", "observer_source_hashes"):
            for path, expected in combined[section].items():
                digest = hashlib.sha256((root / path).read_bytes()).hexdigest()
                if path in lifecycle_changes:
                    self.assertEqual(lifecycle_changes[path]["parent_sha256"], expected, path)
                    self.assertEqual(digest, lifecycle_changes[path]["current_sha256"], path)
                else:
                    self.assertEqual(digest, expected, path)
        for path, expected in lifecycle["added_source_hashes"].items():
            self.assertEqual(hashlib.sha256((root / path).read_bytes()).hexdigest(), expected, path)

        capture_plugin = next(r for r in capture_overlay["superseded_inputs"]
                              if r["path"] == "src/megartx/vllm_scale_plugin.py")
        normal_plugin = next(r for r in normal_overlay["superseded_inputs"]
                             if r["path"] == "src/megartx/vllm_scale_plugin.py")
        replacement = next(r for r in combined["superseded_inputs"]
                           if r["path"] == "src/megartx/vllm_scale_plugin.py")
        self.assertEqual(replacement["historical_sha256"], replacements[replacement["path"]]["current_sha256"])
        self.assertEqual(replacement["pr11_sha256"], capture_plugin["current_sha256"])
        self.assertEqual(replacement["pr12_sha256"], normal_plugin["current_sha256"])
        replacements[replacement["path"]] = dict(replacements[replacement["path"]],
                                                   current_sha256=replacement["combined_sha256"])

        for record in pins["committed_inputs"]:
            digest = hashlib.sha256((root / record["path"]).read_bytes()).hexdigest()
            if record["path"] in replacements:
                replacement = replacements[record["path"]]
                self.assertEqual(replacement["historical_sha256"], record["sha256"])
                self.assertEqual(digest, replacement["current_sha256"], record["path"])
            else:
                self.assertEqual(digest, record["sha256"], record["path"])

    def test_stale_combined_source_or_ledger_hash_is_rejected(self):
        root = Path(__file__).resolve().parents[1]
        native = root / "probes/m1_live_bridge.cu"
        ledger = root / "docs/evidence/m1-reconciliation-source-pins.json"
        read_bytes, read_text = Path.read_bytes, Path.read_text
        for mutation in ("native_source", "ledger_hash"):
            def changed_bytes(path):
                data = read_bytes(path)
                return data + b"\n// simulated bridge edit\n" if path == native and mutation == "native_source" else data
            def changed_text(path, *args, **kwargs):
                data = read_text(path, *args, **kwargs)
                if path == ledger and mutation == "ledger_hash":
                    record = json.loads(data)
                    record["runtime_source_hashes"]["probes/m1_live_bridge.cu"] = "0" * 64
                    return json.dumps(record)
                return data
            with self.subTest(mutation=mutation), patch.object(Path, "read_bytes", changed_bytes), \
                 patch.object(Path, "read_text", changed_text), \
                 self.assertRaisesRegex(AssertionError, "probes/m1_live_bridge.cu"):
                self.test_committed_source_pins_match_exact_base_evidence()


class OriginalGlobalNegativeTests(unittest.TestCase):
    def test_substituted_gate_global_rejected_by_existing_original_projection_oracle(self):
        import numpy as np
        import controlled_reference as controlled
        import nvfp4_reference as formats
        # Sixteen exact unit terms with an original up global of 0.5 yield eight.
        q = np.full((1, 8), 0x22, dtype=np.uint8)
        sf = formats.swizzle_128x4(np.asarray([[0x38]], dtype=np.uint8))
        original = (q.copy(), np.asarray([[0x38]], dtype=np.uint8), 0.5)
        f32word = lambda value: np.asarray([value], dtype=np.float32).view(np.uint32)
        positive = controlled.replay_projection(q, sf, original,
            input_global_bits=f32word(1), alpha_bits=f32word(0.5),
            candidate_bits=controlled.bf16_bits([[8]]))
        self.assertTrue(positive["comparison"]["raw_bf16_bits_equal"])
        with self.assertRaisesRegex(ValueError, "original projection/global product"):
            controlled.replay_projection(q, sf, original,
                input_global_bits=f32word(1), alpha_bits=f32word(0.25),
                candidate_bits=controlled.bf16_bits([[4]]))


if __name__ == "__main__":
    unittest.main()
