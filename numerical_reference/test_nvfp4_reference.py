"""CPU golden cases for the independently implemented NVFP4 oracle."""

import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

import nvfp4_reference as ref


class FormatTests(unittest.TestCase):
    def test_all_byte_combinations_preserve_low_then_high_nibble(self):
        packed = np.arange(256, dtype=np.uint8).reshape(16, 16)
        expected = np.asarray(
            [[n for b in row for n in (int(b) % 16, int(b) // 16)] for row in packed],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(ref.unpack_nibbles(packed), expected)
        np.testing.assert_array_equal(ref.pack_nibbles(expected), packed)

    def test_fp4_golden_values_and_signed_zero(self):
        expected = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]
        self.assertEqual([ref.decode_e2m1(i) for i in range(16)], expected)
        self.assertEqual(math.copysign(1, ref.decode_e2m1(8)), -1)
        self.assertEqual(ref.encode_e2m1_rne(-0.0), 8)

    def test_fp4_boundaries_round_to_even_and_saturate(self):
        ties = [(0.25, 0), (.75, 2), (1.25, 2), (1.75, 4), (2.5, 4), (3.5, 6), (5., 6)]
        for i, (value, expected) in enumerate(ties):
            self.assertEqual(ref.encode_e2m1_rne(value), expected)
            self.assertEqual(ref.encode_e2m1_rne(-value), expected | 8)
            self.assertEqual(ref.encode_e2m1_rne(math.nextafter(value, -math.inf)), i)
            self.assertEqual(ref.encode_e2m1_rne(math.nextafter(value, math.inf)), i + 1)
        self.assertEqual(ref.encode_e2m1_rne(math.inf), 7)
        self.assertEqual(ref.encode_e2m1_rne(-math.inf), 15)
        with self.assertRaises(ValueError):
            ref.encode_e2m1_rne(math.nan)

    def test_e4m3fn_golden_values(self):
        expected = {0x00: 0., 0x01: 2**-9, 0x07: 7*2**-9,
                    0x08: 2**-6, 0x38: 1., 0x77: 240., 0x78: 256., 0x7E: 448.}
        for code, value in expected.items():
            self.assertEqual(ref.decode_e4m3fn(code), value)
            self.assertEqual(ref.decode_e4m3fn(code | 128), -value)
        self.assertTrue(math.isnan(ref.decode_e4m3fn(0x7F)))
        self.assertTrue(math.isnan(ref.decode_e4m3fn(0xFF)))

    def test_all_finite_e4m3fn_codes_roundtrip(self):
        for code in range(256):
            value = ref.decode_e4m3fn(code)
            if math.isfinite(value):
                self.assertEqual(ref.encode_e4m3fn_rne_satfinite(value), code)

    def test_e4m3fn_every_positive_midpoint(self):
        for lower in range(126):
            a, b = ref.decode_e4m3fn(lower), ref.decode_e4m3fn(lower + 1)
            midpoint = (a + b) / 2
            expected = lower if lower % 2 == 0 else lower + 1
            self.assertEqual(ref.encode_e4m3fn_rne_satfinite(midpoint), expected)
            self.assertEqual(ref.encode_e4m3fn_rne_satfinite(math.nextafter(midpoint, -math.inf)), lower)
            self.assertEqual(ref.encode_e4m3fn_rne_satfinite(math.nextafter(midpoint, math.inf)), lower + 1)
        self.assertEqual(ref.encode_e4m3fn_rne_satfinite(math.inf), 126)

    def test_q_times_scale_exhaustive_exact_bf16_proof(self):
        summary = ref.exhaustive_qsf_bf16_summary()
        self.assertEqual(summary["finite_products"], 4064)
        self.assertEqual(summary["products_not_exact_in_bf16"], 0)
        self.assertEqual(summary["largest_absolute_product"], 2688)
        self.assertEqual(summary["smallest_nonzero_absolute_product"], 2**-10)

    def test_bf16_round_even_known_ties_and_subnormals(self):
        x = np.asarray([1 + 2**-8, 1 + 3*2**-8, -(1 + 2**-8), 2**-133, 2**-134])
        expected = [1., 1 + 2**-6, -1., 2**-133, 0.]
        np.testing.assert_array_equal(ref.round_bf16(x), np.asarray(expected, dtype=np.float32))
        self.assertTrue(np.isnan(ref.round_bf16(np.asarray([math.nan]))[0]))

    def test_invalid_shapes_and_scales_fail(self):
        packed = np.ones((1, 8), dtype=np.uint8)
        for scale in (0x7F, 0xB8):
            with self.assertRaises(ValueError):
                ref.decode_qsf(packed, np.asarray([[scale]], dtype=np.uint8))
        with self.assertRaises(ValueError):
            ref.decode_qsf(packed, np.ones((1, 2), dtype=np.uint8))
        with self.assertRaises(ValueError):
            ref.unpack_nibbles(np.ones((1, 8), dtype=np.int32))


class LayoutTests(unittest.TestCase):
    def test_known_128x4_offsets(self):
        expected = {(0, 0): 0, (1, 0): 16, (31, 0): 496, (32, 0): 4,
                    (63, 0): 500, (127, 0): 508, (0, 4): 512, (128, 0): 1024}
        for (row, block), offset in expected.items():
            self.assertEqual(ref.sf_offset_128x4(row, block, 8), offset)

    def test_scale_layout_independent_coordinate_roundtrip_and_padding(self):
        linear = np.arange(129 * 5, dtype=np.int32).reshape(129, 5) + 1
        physical = ref.swizzle_128x4(linear)
        self.assertEqual(physical.shape, (256, 8))
        np.testing.assert_array_equal(ref.unswizzle_128x4(physical, 129, 5), linear)
        self.assertEqual(np.count_nonzero(physical), linear.size)

    def test_both_real_projection_dimensions(self):
        for rows, blocks in ((704, 176), (2816, 44), (1408, 176)):
            linear = np.arange(rows * blocks, dtype=np.int32).reshape(rows, blocks)
            np.testing.assert_array_equal(ref.unswizzle_128x4(ref.swizzle_128x4(linear), rows, blocks), linear)


class QuantizerTests(unittest.TestCase):
    def test_known_group_and_inverse_global_semantics(self):
        x = np.asarray([[0, .25, .5, .75, 1, 1.25, 1.5, 1.75,
                         2, 2.5, 3, 3.5, 4, 5, 6, -6]], dtype=np.float32)
        expected = np.asarray([[0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 15]], dtype=np.uint8)
        for profile in ("vllm_python_rn", "flashinfer_strict_rn"):
            q = ref.quantize_activation_rn(x, 256., profile=profile)
            np.testing.assert_array_equal(ref.unpack_nibbles(q.packed), expected)
            self.assertEqual(int(q.linear_sf[0, 0]), 0x78)
            np.testing.assert_array_equal(ref.decode_qsf(q.packed, q.linear_sf) / 256., ref.E2M1_TABLE[expected])

    def test_zero_blocks_no_nans_and_zero_scale(self):
        for profile in ("vllm_python_rn", "flashinfer_strict_rn"):
            q = ref.quantize_activation_rn(np.zeros((2, 32), dtype=np.float32), 537., profile=profile)
            self.assertTrue(np.all(q.linear_sf == 0))
            self.assertTrue(np.all(ref.decode_qsf(q.packed, q.linear_sf) == 0))

    def test_named_scale_operation_orders_at_e4m3_boundary(self):
        # FP32 fixtures constructed at adjacent E4M3 midpoints. Expected bytes
        # are fixed from the explicit operations, not a production quantizer.
        x = np.zeros((2, 16), dtype=np.float32)
        x[:, 0] = np.asarray([0x3C4280B8, 0x3C5962AD], dtype=np.uint32).view(np.float32)
        strict = ref.quantize_activation_rn(x, 537., profile="flashinfer_strict_rn")
        vllm = ref.quantize_activation_rn(x, 537., profile="vllm_python_rn")
        np.testing.assert_array_equal(strict.linear_sf[:, 0], [0x39, 0x39])
        np.testing.assert_array_equal(vllm.linear_sf[:, 0], [0x39, 0x3A])
        # amax/6 then global would incorrectly select 0x38 for the first row.

    def test_strict_nonzero_block_with_underflowed_sf_keeps_fp4_codes(self):
        x = np.zeros((1, 16), dtype=np.float32)
        x[0, :2] = [2.**-20, -2.**-20]
        strict = ref.quantize_activation_rn(x, 1., profile="flashinfer_strict_rn")
        vllm = ref.quantize_activation_rn(x, 1., profile="vllm_python_rn")
        self.assertEqual(int(strict.linear_sf[0, 0]), 0)
        np.testing.assert_array_equal(ref.unpack_nibbles(strict.packed)[0, :2], [7, 15])
        np.testing.assert_array_equal(ref.unpack_nibbles(vllm.packed)[0, :2], [0, 8])
        self.assertTrue(np.all(ref.decode_qsf(strict.packed, strict.linear_sf) == 0))

    def test_different_activation_globals_can_change_the_quantized_model(self):
        x = np.zeros((1, 16), dtype=np.float32)
        x[0, :2] = [5., 2.04]
        a = ref.quantize_activation_rn(x, 1.)
        b = ref.quantize_activation_rn(x, float(np.float32(1. / 1.31)))
        self.assertNotEqual(int(ref.unpack_nibbles(a.packed)[0, 1]), int(ref.unpack_nibbles(b.packed)[0, 1]))

    def test_reject_invalid_input_or_profile(self):
        for x in (np.ones((1, 17)), np.full((1, 16), math.nan)):
            with self.assertRaises(ValueError):
                ref.quantize_activation_rn(x, 1.)
        with self.assertRaises(ValueError):
            ref.quantize_activation_rn(np.ones((1, 16)), 0.)


class GemmTests(unittest.TestCase):
    @staticmethod
    def ones(rows=1, k=16):
        packed = np.full((rows, k // 2), 0x22, dtype=np.uint8)
        scales = np.full((rows, k // 16), 0x38, dtype=np.uint8)
        return packed, scales

    def test_exact_projection_alpha_and_single_term_golden(self):
        a_codes = np.zeros((1, 16), dtype=np.uint8)
        a_codes[0, 0] = 2
        a = ref.pack_nibbles(a_codes)
        sf = np.asarray([[0x38]], dtype=np.uint8)
        w, wsf = self.ones()
        oracle = ref.gemm_reference(a, sf, w, wsf, alpha_fp32=.5)
        self.assertEqual(float(oracle.expected_bf16[0, 0]), .5)
        self.assertEqual(ref.projection_alpha(.25, 2), .5)
        self.assertTrue(ref.compare_gemm(oracle.expected_bf16, oracle, output_dtype="bf16")["conditional_envelope_pass"])

    def test_chunked_reference_matches_exact_small_integer_dot(self):
        rng = np.random.default_rng(17)
        a = ref.pack_nibbles(rng.integers(0, 16, size=(3, 32), dtype=np.uint8))
        w = ref.pack_nibbles(rng.integers(0, 16, size=(7, 32), dtype=np.uint8))
        asf, wsf = np.full((3, 2), 0x38, dtype=np.uint8), np.full((7, 2), 0x38, dtype=np.uint8)
        oracle = ref.gemm_reference(a, asf, w, wsf, alpha_fp32=.125, output_chunk_rows=2)
        expected = ref.E2M1_TABLE[ref.unpack_nibbles(a)] @ ref.E2M1_TABLE[ref.unpack_nibbles(w)].T * .125
        np.testing.assert_array_equal(oracle.ideal_scaled_fp64, expected)
        self.assertTrue(ref.compare_gemm(oracle.expected_fp32, oracle, output_dtype="fp32")["conditional_envelope_pass"])

    def test_the_six_checkpoint_global_scale_counterexamples(self):
        # Original F32 globals from the immutable checkpoint scale audit.
        pairs = [(4.1053408494917676e-05, 4.123506005271338e-05, .0018601190531626344),
                 (4.541306407190859e-05, 3.633044980233535e-05, .0018601190531626344),
                 (3.742036278708838e-05, 3.578549512894824e-05, .0011567615438252687),
                 (3.742036278708838e-05, 3.9781843952368945e-05, .001209077425301075),
                 (3.687540811370127e-05, 3.960019239457324e-05, .0025576637126505375),
                 (4.323323810240254e-05, 6.321498221950606e-05, .002197265625)]
        a, asf = self.ones(rows=8)
        # Vary exact block scale mantissas to avoid relying on one BF16 midpoint.
        asf[:, 0] = np.arange(0x38, 0x40, dtype=np.uint8)
        w, wsf = self.ones()
        for gate_global, up_global, input_global in pairs:
            oracle = ref.gemm_reference(a, asf, w, wsf, alpha_fp32=ref.projection_alpha(up_global, input_global))
            old = ref.gemm_reference(a, asf, w, wsf, alpha_fp32=ref.projection_alpha(gate_global, input_global))
            self.assertTrue(ref.compare_gemm(oracle.expected_bf16, oracle, output_dtype="bf16")["conditional_envelope_pass"])
            self.assertFalse(ref.compare_gemm(old.expected_bf16, oracle, output_dtype="bf16")["conditional_envelope_pass"])

    def test_cancellation_does_not_hide_a_wrong_scale_behind_relative_error(self):
        a, asf = self.ones()
        w_codes = np.asarray([[2, 10] * 8], dtype=np.uint8)
        w, wsf = ref.pack_nibbles(w_codes), np.asarray([[0x38]], dtype=np.uint8)
        oracle = ref.gemm_reference(a, asf, w, wsf, alpha_fp32=1.)
        self.assertEqual(float(oracle.ideal_scaled_fp64[0, 0]), 0.)
        self.assertFalse(ref.compare_gemm(np.asarray([[.01]], dtype=np.float32), oracle, output_dtype="fp32")["conditional_envelope_pass"])

    def test_conditional_diagnostic_cannot_qualify_a_native_kernel(self):
        a, asf = self.ones()
        w, wsf = self.ones()
        oracle = ref.gemm_reference(a, asf, w, wsf, alpha_fp32=1.)
        report = ref.compare_gemm(oracle.expected_bf16, oracle, output_dtype="bf16")
        self.assertTrue(report["conditional_envelope_pass"])
        self.assertFalse(report["native_kernel_qualified"])
        self.assertIn("unspecified", report["qualification_blocker"])

    def test_gelu_semantic_and_explicit_experimental_cast_profile(self):
        np.testing.assert_array_equal(ref.gelu_tanh_semantic_fp64(np.asarray([0.])), [0.])
        a, asf = self.ones(k=16)
        w, wsf = self.ones(rows=16, k=16)
        aq = ref.QuantizedActivation(a, asf, 1., "captured-test")
        profile = ref.ExpertCastProfile("bf16", "bf16", "bf16", "bf16", "CPU semantic test, unqualified CUDA")
        stages = ref.expert_semantic_reference(aq, (w, wsf, .125), (w, wsf, .25), (w, wsf, .125),
                    input_dequant_global=1., down_input_dequant_global=1.,
                    cast_profile=profile, quantizer_profile="vllm_python_rn")
        self.assertTrue(np.isfinite(stages["down"]).all())
        self.assertIn("unqualified", stages["qualification"])


class CheckpointTests(unittest.TestCase):
    def test_bounded_reader_reads_original_bytes_and_f32_global(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = "model.language_model.layers.0.experts.42.up_proj"
            tensors = {prefix + ".weight": ("U8", [1, 8], bytes([0x22] * 8)),
                       prefix + ".weight_scale": ("F8_E4M3", [1, 1], bytes([0x38])),
                       prefix + ".weight_scale_2": ("F32", [], struct.pack("<f", .25))}
            header, payload = {}, bytearray()
            for name, (dtype, shape, raw) in tensors.items():
                header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(raw)]}
                payload.extend(raw)
            raw_header = json.dumps(header).encode()
            path = Path(directory) / "model-00001-of-00001.safetensors"
            path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
            reader = ref.CheckpointReader(Path(directory))
            projection, hashes = reader.projection(prefix)
            np.testing.assert_array_equal(ref.decode_weight(*projection), np.full((1, 16), .25))
            self.assertTrue(all(len(value) == 64 for value in hashes.values()))
            with self.assertRaises(ValueError):
                reader.tensor(prefix + ".weight", max_bytes=4)


if __name__ == "__main__":
    unittest.main()
