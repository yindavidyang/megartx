"""Synthetic, CPU-only negative controls; no native result is asserted."""

import copy
from functools import lru_cache
import hashlib
from pathlib import Path
import struct
import unittest

import prefill_native_control as control
import prefill_cache_reference as retained


def rows(layer, position):
    return row_payload(layer, position % 128)


@lru_cache(maxsize=3840)
def row_payload(layer, position):
    # Nonuniform across layer, position, K/V and coordinate. Always finite BF16.
    count = control.row_bytes(layer) // 2
    key = struct.pack("<" + "H" * count, *(
        0x3A00 + (position + layer + coordinate) % 128 for coordinate in range(count)))
    value = struct.pack("<" + "H" * count, *(
        0x3D00 + (3 * position + 2 * layer + coordinate) % 128 for coordinate in range(count)))
    return key, value


def maps(frame):
    tables = {layer: {p: 5 * p + 13 + layer for p in control.required_positions(
        layer, frame.start, frame.end)} for layer in control.LAYERS}
    writers = {layer: {p: tables[layer][p] for p in range(frame.start, frame.end)}
               for layer in control.LAYERS}
    return tables, writers


@lru_cache(maxsize=2)
def prefix_payload(end):
    """Explicit synthetic setup, not a claim that previous forwards ran."""
    result = control.ProcessedKVControl()
    result.end = end
    for layer in control.LAYERS:
        for p in range(end):
            result.expected[layer][p] = control.row_digest(layer, p, *rows(layer, p))
            result.slots[layer][p] = 5 * p + 13 + layer
    return result


def seed_prefix(end):
    return copy.deepcopy(prefix_payload(end))


def precheck(ledger, frame, tables):
    for layer in control.LAYERS:
        for position in control.required_positions(layer, frame.start, frame.end):
            if position < frame.start:
                ledger.retained(layer, position, tables[layer][position],
                                *rows(layer, position), phase="pre")


def write(ledger, frame, writers):
    for layer in control.LAYERS:
        for position in range(frame.start, frame.end):
            ledger.processed(layer, position, writers[layer][position], *rows(layer, position))


def postcheck(ledger, frame, tables):
    for layer in control.LAYERS:
        for position in control.required_positions(layer, frame.start, frame.end):
            ledger.retained(layer, position, tables[layer][position],
                            *rows(layer, position), phase="post")


class UnionAndBudgetTests(unittest.TestCase):
    def test_independent_union_agrees_with_retained_reference_for_every_frame(self):
        for start in list(range(0, 2048, 256)) + list(range(2048, 2303)):
            end = start + (256 if start < 2048 else 1)
            for layer in control.LAYERS:
                self.assertEqual(list(control.required_positions(layer, start, end)), list(
                    retained.chunk_required_positions(start, end, layer not in control.GLOBAL)))

    def test_local_union_precedes_final_window_and_global_retains_zero(self):
        self.assertEqual(control.required_positions(0, 1024, 1280), range(1, 1280))
        self.assertEqual(control.required_positions(5, 1792, 2048), range(2048))
        self.assertNotEqual(control.required_positions(0, 1024, 1280), range(256, 1280))

    def test_budget_includes_both_contexts_all_layers_both_roles_heads_and_metadata(self):
        page_sizes = tuple(32 if i in control.GLOBAL else 16 for i in control.LAYERS)
        budget = control.evidence_budget(page_sizes)
        self.assertEqual(budget["sample_positions"], (15, 16, 31, 32, 1023, 1024, 2047, 2048))
        self.assertEqual(budget["kv_bytes"], 3604480)
        self.assertEqual(budget["head_bytes"], 2097152)
        self.assertEqual(budget["total_bytes"], 7798784)
        self.assertLess(budget["total_bytes"], 8 << 20)

    def test_unknown_geometry_or_extra_context_cannot_expand_budget(self):
        for sizes in ((16,) * 29, (True,) * 30, (255,) * 30, (16, 32, 64) * 10):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                control.evidence_budget(sizes)
        with self.assertRaises(ValueError):
            control.evidence_budget((16,) * 30, 3)

    def test_actual_16_token_pages_and_finite_capture_work(self):
        self.assertEqual(control.evidence_budget((16,) * 30)["total_bytes"], 6897664)
        work = control.capture_work_budget()
        self.assertEqual(work["capture_frames"], 9)
        self.assertEqual(work["transferred_bytes_per_context"], 4024934400)
        self.assertEqual(work["single_row_equivalent_reads"], 547650)
        self.assertLess(work["transferred_bytes_per_context"], work["transfer_limit_bytes_per_context"])

    def test_exact_fixed_schedule_and_uncached_final_output(self):
        for start in range(0, 2048, 256):
            control.Frame(start, start + 256)
        for start in range(2048, 2303):
            control.Frame(start, start + 1)
        for start, end in ((0, 2048), (0, 255), (True, 257), (2048, 2050), (2303, 2304)):
            with self.subTest(frame=(start, end)), self.assertRaises(ValueError):
                control.Frame(start, end)


class MatchingTests(unittest.TestCase):
    def identity(self):
        return {**{field: "a" * 64 for field in control.IDENTITY_FIELDS},
                "prompt_tokens": 2048, "chunk_tokens": 256, "output_tokens": 256,
                "capacity_tokens": 2304, "compute_dtype": "bfloat16", "kv_dtype": "bfloat16"}

    def test_exact_same_path_identity(self):
        self.assertTrue(control.match_native_identity(self.identity(), self.identity()))

    def test_every_independent_identity_dimension_fails_when_changed(self):
        for key in control.IDENTITY_FIELDS:
            candidate = self.identity()
            candidate[key] = "b" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                control.match_native_identity(self.identity(), candidate)

    def test_cross_path_or_missing_identity_cannot_be_called_matched(self):
        for key, value in (("chunk_tokens", 2048), ("output_tokens", 2),
                           ("kv_dtype", "fp8"), ("prompt_tokens", True)):
            candidate = self.identity()
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                control.match_native_identity(self.identity(), candidate)
        candidate = self.identity()
        candidate.pop("sampler_sha256")
        with self.assertRaises(ValueError):
            control.match_native_identity(self.identity(), candidate)


class ProcessedControlTests(unittest.TestCase):
    def test_capture_does_not_expand_to_all_255_decode_prefixes(self):
        ledger = control.ProcessedKVControl()
        ledger.end = 2049
        frame = control.Frame(2049, 2050)
        with self.assertRaisesRegex(ValueError, "first consumed anchor"):
            ledger.begin(frame, *maps(frame))

    def test_first_chunk_all_layers_actual_source_rows_and_post_write_bits(self):
        frame = control.Frame(0, 256)
        tables, writers = maps(frame)
        ledger = control.ProcessedKVControl()
        ledger.begin(frame, tables, writers)
        write(ledger, frame, writers)
        postcheck(ledger, frame, tables)
        self.assertEqual(ledger.finish(queries_complete=True), 256)

    def test_anchor_all_layers_full_required_union_and_global_prefix(self):
        frame = control.Frame(2048, 2049)
        tables, writers = maps(frame)
        ledger = seed_prefix(frame.start)
        ledger.begin(frame, tables, writers)
        precheck(ledger, frame, tables)
        write(ledger, frame, writers)
        postcheck(ledger, frame, tables)
        self.assertEqual(ledger.finish(queries_complete=True), 2049)

    def test_remapped_required_prefix_is_rejected_before_writer(self):
        frame = control.Frame(1024, 1280)
        tables, writers = maps(frame)
        ledger = seed_prefix(frame.start)
        tables[0][1] += 1
        with self.assertRaisesRegex(ValueError, "remapped"):
            ledger.begin(frame, tables, writers)
        self.assertTrue(ledger.poisoned)

    def test_last_window_only_is_insufficient_before_queries_finish(self):
        frame = control.Frame(1024, 1280)
        tables, writers = maps(frame)
        tables[0] = {p: slot for p, slot in tables[0].items() if p >= 256}
        with self.assertRaisesRegex(ValueError, "union"):
            seed_prefix(frame.start).begin(frame, tables, writers)

    def test_global_prefix_cannot_be_recycled(self):
        frame = control.Frame(2048, 2049)
        tables, writers = maps(frame)
        del tables[29][0]
        with self.assertRaisesRegex(ValueError, "prefix"):
            seed_prefix(frame.start).begin(frame, tables, writers)

    def test_independent_table_catches_wrong_writer_slot(self):
        frame = control.Frame(0, 256)
        tables, writers = maps(frame)
        writers[29][31] += 1
        ledger = control.ProcessedKVControl()
        with self.assertRaisesRegex(ValueError, "independent table"):
            ledger.begin(frame, tables, writers)

    def test_post_read_requires_a_preceding_processed_callback(self):
        frame = control.Frame(0, 256)
        tables, writers = maps(frame)
        ledger = control.ProcessedKVControl()
        ledger.begin(frame, tables, writers)
        with self.assertRaisesRegex(ValueError, "differs"):
            ledger.retained(0, 0, tables[0][0], *rows(0, 0), phase="post")

    def test_malformed_retained_phase_poisons_instead_of_raising_unhandled_typeerror(self):
        frame = control.Frame(0, 256)
        tables, writers = maps(frame)
        for phase in ([], {}, None, 1, True, "unknown"):
            ledger = control.ProcessedKVControl()
            ledger.begin(frame, tables, writers)
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                ledger.retained(0, 0, tables[0][0], *rows(0, 0), phase=phase)
            self.assertTrue(ledger.poisoned)

    def test_k_v_swap_and_byte_corruption_fail_exactly(self):
        frame = control.Frame(0, 256)
        tables, writers = maps(frame)
        for swap in (True, False):
            ledger = control.ProcessedKVControl()
            ledger.begin(frame, tables, writers)
            write(ledger, frame, writers)
            k, v = rows(29, 255)
            bad = (v, k) if swap else (bytes([k[0] ^ 1]) + k[1:], v)
            with self.subTest(swap=swap), self.assertRaisesRegex(ValueError, "differs"):
                ledger.retained(29, 255, tables[29][255], *bad, phase="post")
            self.assertTrue(ledger.poisoned)

    def test_missing_prefix_check_blocks_first_native_writer(self):
        frame = control.Frame(2048, 2049)
        tables, writers = maps(frame)
        ledger = seed_prefix(frame.start)
        ledger.begin(frame, tables, writers)
        with self.assertRaisesRegex(ValueError, "before complete"):
            ledger.processed(0, 2048, writers[0][2048], *rows(0, 2048))

    def test_partial_layer_or_unsynchronized_handoff_never_commits(self):
        frame = control.Frame(0, 256)
        tables, writers = maps(frame)
        for complete in (False, True, 1):
            ledger = control.ProcessedKVControl()
            ledger.begin(frame, tables, writers)
            ledger.processed(0, 0, writers[0][0], *rows(0, 0))
            with self.subTest(complete=complete), self.assertRaises(ValueError):
                ledger.finish(queries_complete=complete)
            self.assertEqual(ledger.end, 0)
            self.assertTrue(ledger.poisoned)

    def test_actual_bf16_type_extent_finiteness_and_signed_zero(self):
        raw = struct.pack("<2H", 0, 0x8000)
        self.assertEqual(control.finite_bf16(raw, 2), (0, 0x8000))
        for bad in (bytearray(raw), raw + b"x", struct.pack("<2H", 0x7F80, 0),
                    struct.pack("<2H", 0xFF80, 0), struct.pack("<2H", 0x7FC1, 0)):
            with self.subTest(raw=bad), self.assertRaises(ValueError):
                control.finite_bf16(bad, 2)

    def test_caller_mapping_mutation_does_not_change_admitted_mapping(self):
        frame = control.Frame(0, 256)
        tables, writers = maps(frame)
        ledger = control.ProcessedKVControl()
        ledger.begin(frame, tables, writers)
        tables[0][0] = 999999
        ledger.processed(0, 0, writers[0][0], *rows(0, 0))
        self.assertEqual(ledger.slots[0][0], writers[0][0])


class FrontierTests(unittest.TestCase):
    def fixture(self):
        heads = [(p, p + 1, p != 2047) for p in range(255, 2048, 256)]
        heads += [(p, p + 1, False) for p in range(2048, 2303)]
        emissions = [(7 + i, 1234 + i) for i in range(256)]
        inputs = [(2048 + i, emissions[i][1]) for i in range(255)]
        return tuple(range(2048)), heads, emissions, inputs, 2303

    def test_final_prompt_uncached_anchor_and_next_input_chain(self):
        receipt = control.verify_frontier(*self.fixture())
        self.assertEqual(receipt["checked_heads"], 263)
        self.assertEqual(receipt["uncached_output_position"], 2303)
        self.assertNotIn("tokens", receipt)
        # Independently spell the existing receipt encoding; do not import its
        # observer/helper implementation into this oracle.
        expected = hashlib.sha256(("[" + ",".join(str(1234 + i) for i in range(256)) + "]").encode()).hexdigest()
        self.assertEqual(receipt["token_ids_sha256"], expected)
        self.assertEqual(receipt["token_ids_encoding"], "canonical-compact-json-integer-array-utf8")
        self.assertNotEqual(receipt["token_ids_sha256"], hashlib.sha256(
            struct.pack("<256q", *range(1234, 1490))).hexdigest())

    def test_intermediate_head_cannot_supply_anchor(self):
        args = list(self.fixture())
        args[2][0] = (6, args[2][0][1])
        with self.assertRaises(ValueError):
            control.verify_frontier(*args)

    def test_actual_next_input_must_equal_preceding_emission(self):
        for index in (0, 254):
            args = list(self.fixture())
            position, token = args[3][index]
            args[3][index] = (position, token + 1)
            with self.subTest(index=index), self.assertRaises(ValueError):
                control.verify_frontier(*args)

    def test_wrong_frontier_duplicate_or_omitted_outputs_fail(self):
        for mutation in range(5):
            args = list(self.fixture())
            if mutation == 0:
                args[4] = 2304
            elif mutation == 1:
                args[1][7] = (2047, 2048, True)
            elif mutation == 2:
                args[2].pop()
            elif mutation == 3:
                args[1][0] = (255, 256, 1)
            else:
                args[3][0] = (2047, args[3][0][1])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                control.verify_frontier(*args)


class NumericalPolicyTests(unittest.TestCase):
    def test_exact_repeatability_does_not_claim_independent_arithmetic(self):
        for changed in (True, False):
            report = control.compare_bf16_samples(b"\x80\x3f", b"\x80\x3f",
                                                  changed_arithmetic_path=changed)
            self.assertTrue(report["exact_sample_agreement"])
            self.assertEqual(report["same_path_repeatability_passed"], not changed)
            self.assertFalse(report["independent_arithmetic_qualified"])
            self.assertIsNone(report["numerical_tolerance"])

    def test_small_difference_is_not_accepted_by_fabricated_tolerance(self):
        report = control.compare_bf16_samples(b"\x80\x3f", b"\x81\x3f",
                                              changed_arithmetic_path=True)
        self.assertEqual(report["max_abs"], 0.0078125)
        self.assertEqual(report["different_words"], 1)
        self.assertFalse(report["exact_sample_agreement"])
        self.assertFalse(report["whole_model_quality_qualified"])

    def test_signed_zero_difference_remains_a_bit_difference(self):
        report = control.compare_bf16_samples(b"\0\0", b"\0\x80", changed_arithmetic_path=False)
        self.assertEqual(report["max_abs"], 0)
        self.assertEqual(report["different_words"], 1)
        self.assertFalse(report["same_path_repeatability_passed"])

    def test_runtime_observer_and_mutation_modules_are_not_dependencies(self):
        source = Path(control.__file__).read_text()
        import ast
        tree = ast.parse(source)
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertEqual(imports, {"dataclasses", "hashlib", "json", "math", "struct"})


if __name__ == "__main__":
    unittest.main()
