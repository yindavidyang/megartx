"""Independent semantic/format controls for the conditional layer-0 contract."""
from decimal import Decimal, localcontext
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import layer0_norm_attention_reference as ref
from router_reference import bf16_decode, bf16_rne_bits


def bits(values):
    return bf16_rne_bits(np.asarray(values, dtype=np.float32))


class NormAttentionContractTests(unittest.TestCase):
    def test_constant_rms_weight_is_direct_scale_not_delta_gamma(self):
        x, weight = bits([[1] * 8]), bits([2] * 8)
        interval = ref.rms_interval(x, weight)
        self.assertTrue(ref.check_interval(bits([[2] * 8]), interval)["conditional_interval_pass"])
        self.assertFalse(ref.check_interval(bits([[3] * 8]), interval)["conditional_interval_pass"])

    def test_rms_nonuniform_weight_matches_decimal_reference(self):
        x, w = bits([[1, -2, 3, -4]]), bits([0.5, 2, -1, 0])
        with localcontext() as ctx:
            ctx.prec = 80
            eps = Decimal.from_float(float(np.float32(1e-6)))
            denom = (Decimal(30) / 4 + eps).sqrt()
            expected = bits([[float(Decimal(a) * Decimal(b) / denom) for a, b in zip((1, -2, 3, -4), ("0.5", "2", "-1", "0"))]])
        self.assertTrue(ref.check_interval(expected, ref.rms_interval(x, w))["conditional_interval_pass"])
        self.assertFalse(ref.check_interval(expected[:, ::-1].copy(), ref.rms_interval(x, w))["conditional_interval_pass"])

    def test_zero_rms_is_supported_without_singular_reduction(self):
        x = bits([[0] * 256])
        self.assertTrue(ref.check_interval(x, ref.rms_interval(x))["conditional_interval_pass"])

    def test_rms_rejects_underflow_nonfinite_and_wrong_weight_extent(self):
        for x, w in ((bits([[2.0 ** -80] * 4]), None), (np.array([[0x7f80]], dtype=np.uint16), None),
                     (bits([[1, 2]]), bits([1]))):
            with self.assertRaises(ValueError):
                ref.rms_interval(x, w)

    def test_rms_rejects_other_epsilon_and_extended_lane(self):
        with self.assertRaises(ValueError):
            ref.rms_interval(bits([[1]]), epsilon=1e-5)
        with self.assertRaises(ValueError):
            ref.rms_interval(bits([[1] * 2817]))

    def test_neox_rope_quarter_turn_has_exact_sign_and_pairing(self):
        x = np.zeros((1, 256), dtype=np.float32)
        x[0, 0], x[0, 128], x[0, 1], x[0, 129] = 2, 3, -1, 4
        cache = np.concatenate((np.zeros(128), np.ones(128)))[None, :]
        expected = np.concatenate((-x[:, 128:], x[:, :128]), axis=1)
        expected[expected == 0] = 0  # +0 - +0 in the actual source expression
        np.testing.assert_array_equal(ref.rope_reference(bits(x), bits(cache)), bits(expected))
        self.assertFalse(np.array_equal(ref.rope_reference(bits(x), bits(cache)), bits(x)))

    def test_rope_rejects_fp32_cache_and_wrong_head_geometry(self):
        with self.assertRaises(ValueError):
            ref.rope_reference(bits([[1] * 256]), np.zeros((1, 256), dtype=np.uint32))
        with self.assertRaises(ValueError):
            ref.rope_reference(bits([[1] * 128]), bits([[0] * 256]))

    def test_constant_attention_is_exact_for_both_source_profiles(self):
        q = bits([[0] * 4] * 4)
        k, v = bits(np.zeros((3, 2, 4))), bits(np.full((3, 2, 4), 2))
        for provider in ("fa2", "xqa"):
            interval = ref.attention_interval(q, k, v, provider=provider)
            np.testing.assert_array_equal(interval["ideal"], np.full((4, 4), 2))
            self.assertTrue(ref.check_interval(bits([[2] * 4] * 4), interval)["conditional_interval_pass"])
            self.assertFalse(ref.check_interval(bits([[4] * 4] * 4), interval)["conditional_interval_pass"])

    def test_grouped_query_heads_use_contiguous_groups(self):
        q = bits(np.zeros((4, 4)))
        k = bits(np.zeros((2, 2, 4)))
        v = bits([[[1] * 4, [3] * 4], [[1] * 4, [3] * 4]])
        interval = ref.attention_interval(q, k, v, provider="fa2")
        np.testing.assert_array_equal(interval["ideal"], np.array([[1] * 4, [1] * 4, [3] * 4, [3] * 4]))
        self.assertFalse(ref.check_interval(bits([[1] * 4, [3] * 4, [1] * 4, [3] * 4]), interval)["conditional_interval_pass"])

    def test_scale_and_value_permutation_controls_are_discriminating(self):
        q, k, v = bits([[1, 0]]), bits([[[0, 0]], [[4, 0]]]), bits([[[0, 0]], [[8, 8]]])
        interval = ref.attention_interval(q, k, v, provider="fa2")
        # Closed form 8*exp(4)/(1+exp(4)) from the two-key mathematical softmax.
        with localcontext() as ctx:
            ctx.prec = 80
            e = Decimal(4).exp()
            expected = bits([[float(8 * e / (1 + e))] * 2])
        self.assertTrue(ref.check_interval(expected, interval)["conditional_interval_pass"])
        self.assertFalse(ref.check_interval(bits([[4, 4]]), interval)["conditional_interval_pass"])
        swapped = ref.attention_interval(q, k, v[::-1].copy(), provider="fa2")
        self.assertFalse(ref.check_interval(bits(swapped["ideal"]), interval)["conditional_interval_pass"])

    def test_joint_kv_permutation_cannot_certify_slot_ownership(self):
        q, k, v = bits([[1, 0]]), bits([[[0, 0]], [[4, 0]]]), bits([[[0, 0]], [[8, 8]]])
        a = ref.attention_interval(q, k, v, provider="fa2")
        b = ref.attention_interval(q, k[::-1].copy(), v[::-1].copy(), provider="fa2")
        np.testing.assert_array_equal(bits(a["ideal"]), bits(b["ideal"]))

    def test_attention_fails_closed_for_unknown_profile_and_unbounded_context(self):
        q, k, v = bits([[1, 1]]), bits(np.ones((34, 1, 2))), bits(np.ones((34, 1, 2)))
        with self.assertRaises(ValueError):
            ref.attention_interval(q, k, v, provider="fa2")
        with self.assertRaises(ValueError):
            ref.attention_interval(q, k[:2], v[:2], provider="native")

    def test_attention_rejects_exponent_ftz_lane(self):
        with self.assertRaises(ValueError):
            ref.attention_interval(bits([[100, 100]]), bits([[[100, 100]], [[-100, -100]]]),
                                   bits([[[1, 1]], [[1, 1]]]), provider="xqa")

    def test_xqa_declares_partial_cast_without_native_acceptance(self):
        q, k, v = bits([[1, 1]]), bits([[[1, 1]]]), bits([[[1, 1]]])
        interval = ref.attention_interval(q, k, v, provider="xqa")
        self.assertEqual(interval["partial_output_bf16_casts"], 1)
        self.assertFalse(ref.check_interval(bits([[1, 1]]), interval)["native_arithmetic_qualified"])

    def test_cuda_launch_correlation_includes_annotation_parent_and_async_kernel(self):
        span = {"ts": 10, "dur": 5, "pid": 3, "tid": 4}
        events = [{"ph": "X", "cat": "cuda_runtime", "ts": 11, "pid": 3, "tid": 4, "args": {"correlation": 7}},
                  {"ph": "X", "cat": "kernel", "ts": 100, "name": "kernel_mha", "args": {"correlation": 7}},
                  {"ph": "X", "cat": "cuda_runtime", "ts": 11, "pid": 3, "tid": 5, "args": {"correlation": 8}},
                  {"ph": "X", "cat": "kernel", "ts": 101, "name": "other", "args": {"correlation": 8}}]
        self.assertEqual([e["name"] for e in ref.span_launches(events, span)], ["kernel_mha"])

    def test_identical_kernel_name_cannot_hide_one_block_vs_multiblock(self):
        span = {"ts": 10, "dur": 5, "pid": 3, "tid": 4}
        profile = {"kernel_name_sha256": hashlib.sha256(b"kernel_mha").hexdigest(),
                   "grid": [21, 8, 1], "block": [128, 1, 2]}
        events = [{"ph": "X", "cat": "cuda_runtime", "ts": 11, "pid": 3, "tid": 4, "args": {"correlation": 7}},
                  {"ph": "X", "cat": "kernel", "ts": 100, "name": "kernel_mha", "args": {"correlation": 7, "grid": [1, 8, 1], "block": [128, 1, 2]}}]
        with self.assertRaises(ValueError):
            ref.require_launch_profile(events, span, profile)
        events[1]["args"]["grid"] = [21, 8, 1]
        ref.require_launch_profile(events, span, profile)


class ReplayPacketTests(unittest.TestCase):
    """Exercise the complete reader with real hashes, NPZ fields and launches."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "replay"
        self.directory.mkdir()
        self.directory.with_suffix(".exit").write_text("0\n")
        widths = {"input_norm": 2816, "q_norm": 4096, "k_norm": 2048,
                  "v_norm": 2048, "post_attention_norm": 2816, "pre_ff2_norm": 2816}
        row = {name + "_out": np.zeros(width, dtype=np.uint16) for name, width in widths.items()}
        row.update(rope_q_out=bits(np.zeros(4096)), rope_k_out=bits(np.zeros(2048)),
                   attention_q=bits(np.zeros(4096)), attention_out=bits(np.ones(4096)))
        self.case = {"rows": {31: row, 32: row}, "prefixes": [{
            "stored_prefix_k": bits(np.zeros((33, 2048))),
            "stored_prefix_v": bits(np.ones((33, 2048)))}]}

        self.arrays = {}
        for label, count in (("full_m33", 2), ("cached_m32", 1), ("cached_m1", 1)):
            for name, width in widths.items():
                self.arrays[label + "_" + name] = np.zeros((count, width), dtype=np.uint16)
            self.arrays[label + "_rope"] = np.zeros((count, 6144), dtype=np.uint16)
        attention_labels = ("full_fa2", "cached_fa2_m32", "cached_xqa", "cached_inputs_fa2",
                            "full_inputs_xqa", "constant_fa2", "constant_xqa")
        for label in attention_labels:
            self.arrays[label] = bits(np.ones((2 if label == "full_fa2" else 1, 16, 256)))
        self.arrays = {label + "_r" + str(repeat): value.copy()
                       for label, value in self.arrays.items() for repeat in range(3)}

        def profile(provider, grid, block):
            return {"provider": provider, "grid": grid, "block": block,
                    "kernel_name_sha256": hashlib.sha256(("fixture_" + provider).encode()).hexdigest()}

        full_fa2 = profile("fa2", [2, 1, 8], [32, 4, 1])
        cached_fa2 = profile("fa2", [1, 1, 8], [32, 4, 1])
        xqa = profile("xqa", [21, 8, 1], [128, 1, 2])
        self.dispatch = {"full": {"launches": [full_fa2]}, "cached": {"launches": [cached_fa2, xqa]}}
        events = []
        for label in attention_labels:
            launch = (cached_fa2 if label == "cached_fa2_m32" else
                      xqa if label in {"cached_xqa", "full_inputs_xqa", "constant_xqa"} else full_fa2)
            for repeat in range(3):
                correlation = len(events) + 1
                timestamp = correlation * 10
                events.extend([
                    {"ph": "X", "cat": "user_annotation", "pid": 1, "tid": 2,
                     "ts": timestamp, "dur": 5,
                     "name": "megartx.norm_attention." + label + ".r" + str(repeat)},
                    {"ph": "X", "cat": "cuda_runtime", "pid": 1, "tid": 2,
                     "ts": timestamp + 1, "args": {"correlation": correlation}},
                    {"ph": "X", "cat": "kernel", "ts": timestamp + 1000,
                     "name": "fixture_" + launch["provider"],
                     "args": {"correlation": correlation, "grid": launch["grid"], "block": launch["block"]}}])
        trace = self.directory / "trace.json.gz"
        with gzip.open(trace, "wt") as stream:
            json.dump({"traceEvents": events}, stream)
        self.report = {"sources": ref.REPLAY_SOURCE_SHA256, "repeats": 3,
                       "torch": "2.13.0+cu130", "cuda": "13.0", "xqa_table_capacity_tokens": 8448,
                       "retained_dispatch": self.dispatch, "peak_allocated_bytes": 0,
                       "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()}
        self.write_packet()

    def write_packet(self):
        result = self.directory / "results.npz"
        np.savez_compressed(result, **self.arrays)
        self.report["results_sha256"] = hashlib.sha256(result.read_bytes()).hexdigest()
        (self.directory / "report.json").write_text(json.dumps(self.report))

    def read_packet(self):
        return ref.analyze_replay(self.directory, self.case, self.case, self.dispatch)

    def assert_dtype_rejected(self, field, dtype):
        before = self.arrays[field]
        self.arrays[field] = before.astype(dtype)
        self.assertNotEqual(before.dtype, self.arrays[field].dtype)
        np.testing.assert_array_equal(before, self.arrays[field])
        self.write_packet()  # Bind the changed representation; rejection must not be a digest failure.
        with self.assertRaisesRegex(ValueError, "raw BF16 dtype"):
            self.read_packet()

    def test_valid_raw_bf16_packet_binds_every_attention_launch(self):
        report = self.read_packet()
        self.assertTrue(report["raw_boundary_reproduction"])
        self.assertTrue(report["constant_controls_pass"])
        self.assertEqual(report["attention_launches_bound"], 21)
        self.assertTrue(all(check["conditional_interval_pass"] for check in report["crossed_conditional_checks"].values()))

    def test_first_repeat_float32_codes_rejected_with_matching_digest(self):
        self.assert_dtype_rejected("full_fa2_r0", np.float32)

    def test_second_repeat_float32_codes_rejected_with_matching_digest(self):
        self.assert_dtype_rejected("full_fa2_r1", np.float32)

    def test_third_repeat_uint32_codes_rejected_with_matching_digest(self):
        self.assert_dtype_rejected("cached_m1_q_norm_r2", np.uint32)


if __name__ == "__main__":
    unittest.main()
