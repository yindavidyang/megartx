"""CPU argument/immutability tests, without a GPU arithmetic claim."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from megartx import nvfp4_runtime as runtime


class ModeTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.activations = []
        def projection(value):
            return runtime.Projection(object(), object(), object(), np.array([value], dtype=np.float32))
        self.expert = runtime.Expert(42, projection(.25), projection(.5), projection(.75),
                                    np.array([2], dtype=np.float32), np.array([3], dtype=np.float32))
        self.original = (self.expert.up, self.expert.up.global_scale.copy())
        def mm(lane):
            def compute(q, sf, p, activation_global, *, alpha):
                self.calls.append((lane, p, alpha.copy()))
                return alpha.copy()
            return compute
        def quantize(x, global_scale, *, return_multiplier):
            self.assertTrue(return_multiplier)
            return x, object(), np.float32(1) / global_scale
        def activation(gate, up):
            self.activations.append("native")
            return gate * up
        self.patches = [patch.object(runtime, "mm_native", mm("native")),
                        patch.object(runtime, "mm_reference", mm("reference")),
                        patch.object(runtime, "quantize", quantize),
                        patch.object(runtime, "activation_semantic", lambda g, u: self.activations.append("mathematical") or g * u),
                        patch.dict(sys.modules, {"megartx.nvfp4_activation": SimpleNamespace(cutlass_gelu_product=activation)})]
        for p in self.patches:
            p.start()
    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
    def test_native_original_alphas_and_stages(self):
        _, stages = runtime.run_expert(np.ones(1), self.expert, return_stages=True)
        self.assertEqual([c[0] for c in self.calls], ["native"] * 3)
        np.testing.assert_array_equal(stages["up_alpha"], np.array([1], dtype=np.float32))
        np.testing.assert_array_equal(stages["down_alpha"], np.array([2.25], dtype=np.float32))
        self.assertEqual(self.activations, ["native"])
    def test_paired_reference_reuses_native_activation(self):
        runtime.run_expert(np.ones(1), self.expert, mode="paired_reference")
        self.assertEqual([c[0] for c in self.calls], ["reference"] * 3)
        self.assertEqual(self.activations, ["native"])
    def test_negative_changes_only_up_descriptor_alpha(self):
        _, stages = runtime.run_expert(np.ones(1), self.expert, mode="gate_only_negative_control", return_stages=True)
        self.assertIs(self.calls[0][1], self.expert.gate)
        self.assertIs(self.calls[2][1], self.expert.down)
        overridden = self.calls[1][1]
        self.assertIs(overridden.packed, self.expert.up.packed)
        self.assertIs(overridden.scales, self.expert.up.scales)
        self.assertIs(overridden.swizzled, self.expert.up.swizzled)
        self.assertIs(self.expert.up, self.original[0])
        np.testing.assert_array_equal(self.expert.up.global_scale, self.original[1])
        np.testing.assert_array_equal(stages["up_alpha"], stages["gate_alpha"])
        self.assertEqual([c[0] for c in self.calls], ["native"] * 3)
    def test_legacy_reference_retains_separate_mathematical_activation(self):
        runtime.run_expert(np.ones(1), self.expert, mode="reference")
        self.assertEqual(self.activations, ["mathematical"])
    def test_unknown_arithmetic_mode_rejected(self):
        with self.assertRaises(KeyError):
            runtime.run_expert(np.ones(1), self.expert, mode="unlisted")


if __name__ == "__main__":
    unittest.main()
