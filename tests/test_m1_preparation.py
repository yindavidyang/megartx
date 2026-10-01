"""Pre-mutation integration guards: no GPU/backend dependency is imported."""

from dataclasses import replace
import itertools
import subprocess
import sys
import unittest

from megartx.adapter import BackendUnavailable
from megartx.m1_preparation import (
    PENDING, PreparationRequest, run_candidate, select_preparation,
)


def supported_request():
    return PreparationRequest(1, 2816, 128, 8, 704, 1, 1,
        "prequantized_nvfp4_common_row", True, True, False, False, False, False,
        False, True, "eager", False, True)


class M1PreparationGuardTests(unittest.TestCase):
    def test_supported_cpu_contract_never_selects_unimplemented_kernel(self):
        for opt_in, swap1, swap2 in itertools.product((False, True), repeat=3):
            request = replace(supported_request(), fc1_swap_ab=swap1, fc2_swap_ab=swap2)
            decision = select_preparation(request, opt_in=opt_in, selected_ids=tuple(range(8)))
            self.assertEqual(decision.backend, "incumbent")
            self.assertTrue(decision.cpu_contract_eligible)
            self.assertTrue(set(PENDING).issubset(decision.reasons))

    def test_each_unsupported_shape_and_mode_falls_back_before_mutation(self):
        changes = {
            "tokens": 2, "hidden": 2800, "experts": 64, "top_k": 4,
            "intermediate": 768, "tp_size": 2, "ep_size": 2,
            "quantization": "nvfp4_from_bf16", "input_sf_present": False,
            "swizzled_input_sf": False, "use_per_expert_act_scale": True,
            "min_latency_mode": True, "lora": True, "groupwise": True,
            "all_to_all": True, "correction_runner_active": False,
            "execution_mode": "graph", "fc1_swap_ab": None, "fc2_swap_ab": None,
        }
        for key, value in changes.items():
            request = replace(supported_request(), **{key: value})
            before = repr(request)
            with self.subTest(field=key):
                decision = select_preparation(request, opt_in=True)
                self.assertEqual(decision.backend, "incumbent")
                self.assertFalse(decision.cpu_contract_eligible)
                self.assertEqual(repr(request), before)

    def test_invalid_declared_m1_ids_are_errors_not_silent_fallback_or_coercion(self):
        invalid = ((0,) * 8, tuple(range(7)), tuple(range(7)) + (128,),
                   tuple(range(7)) + (-1,), (False,) + tuple(range(1, 8)), list(range(8)))
        for ids in invalid:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                select_preparation(supported_request(), opt_in=True, selected_ids=ids)
        # Unsupported prefill does not reinterpret its token/expert matrix as M1.
        decision = select_preparation(replace(supported_request(), tokens=2), selected_ids=invalid[0])
        self.assertEqual(decision.backend, "incumbent")

    def test_truthy_flags_and_unknown_runtime_contracts_cannot_count_as_eligible(self):
        for key, value in (("tokens", True), ("input_sf_present", 1), ("lora", 0),
                           ("fc1_swap_ab", 1), ("correction_runner_active", 1),
                           ("execution_mode", "unknown")):
            with self.subTest(field=key):
                decision = select_preparation(replace(supported_request(), **{key: value}), opt_in=True)
                self.assertFalse(decision.cpu_contract_eligible)
                self.assertEqual(decision.backend, "incumbent")
        with self.assertRaises(ValueError):
            select_preparation(supported_request(), opt_in="true")

    def test_caller_receipts_and_positive_flags_cannot_unlock_execution(self):
        state = bytearray(b"unchanged decode state")
        with self.assertRaises(BackendUnavailable):
            run_candidate(state, evidence={"abi_verified": True, "compiled": True,
                          "gpu_verified": True, "graph_qualified": True})
        self.assertEqual(state, bytearray(b"unchanged decode state"))

    def test_import_does_not_load_gpu_packages_or_numerical_oracle(self):
        code = ("import sys; import megartx.m1_preparation; "
                "assert not any(name in sys.modules for name in "
                "('numpy','torch','vllm','flashinfer','m1_preparation_reference'))")
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
