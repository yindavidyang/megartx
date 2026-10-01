"""CPU regression checks for the actual-runner binding guard.

Load the source file directly, avoiding package initialization and GPU imports.
The default path works after copying this test into the repository. For a
task-owned review copy, set MEGARTX_INTEGRATION_SOURCE to the helper's path.
Only the four vLLM APIs imported by binding() are stubbed.
"""

import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


SOURCE = Path(os.environ.get(
    "MEGARTX_INTEGRATION_SOURCE",
    str(Path(__file__).resolve().parents[1] / "src/megartx/nvfp4_integration.py"),
))


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "_megartx_integration_guard_subject", SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load integration helper: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load_helper()


class FusedMoEKernelModularImpl:
    pass


class FlashInferExperts:
    pass


class FusedMoEModularMethod:
    """The pinned wrapper's old-owner/kernel identity contract only."""

    def __init__(self, old_quant_method, moe_kernel):
        self.old_quant_method = old_quant_method
        self.moe_kernel = moe_kernel
        self.is_monolithic = False


def adapter(layer, *args, **kwargs):
    raise AssertionError("A binding check must not execute the adapter")


def bypass(layer, *args, **kwargs):
    raise AssertionError("A binding check must not execute a bypass")


class RoutedLayer:
    forward_modular = adapter


class BindingGuardTests(unittest.TestCase):
    def setUp(self):
        self.kernel = types.SimpleNamespace(
            impl=FusedMoEKernelModularImpl(), fused_experts=FlashInferExperts()
        )
        self.owner = types.SimpleNamespace(
            moe_kernel=self.kernel, is_monolithic=False
        )
        self.layer = RoutedLayer()
        self.layer.layer_name = "model.language_model.layers.0.moe"
        self.layer.quant_method = self.owner
        self.runner = types.SimpleNamespace(
            routed_experts=self.layer, _quant_method=self.owner
        )
        self.registry = {self.layer.layer_name: self.runner}
        self.layer._megartx = {
            "registry": self.registry, "owner": self.owner, "kernel": self.kernel
        }
        self.context = types.SimpleNamespace(no_compile_layers=self.registry)

        leaves = {
            "vllm.forward_context": {
                "get_forward_context": lambda: self.context
            },
            "vllm.model_executor.layers.fused_moe.fused_moe_modular_method": {
                "FusedMoEModularMethod": FusedMoEModularMethod
            },
            "vllm.model_executor.layers.fused_moe.modular_kernel": {
                "FusedMoEKernelModularImpl": FusedMoEKernelModularImpl
            },
            "vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe": {
                "FlashInferExperts": FlashInferExperts
            },
        }
        modules = {}
        for name, exports in leaves.items():
            parts = name.split(".")
            for end in range(1, len(parts)):
                package = ".".join(parts[:end])
                if package not in modules:
                    modules[package] = types.ModuleType(package)
                    modules[package].__path__ = []
            modules[name] = types.ModuleType(name)
            modules[name].__dict__.update(exports)
        patcher = mock.patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def assert_rejected_after_mutation(self, mutation, message):
        # A previous successful check must not cache acceptance of later
        # registry, kernel, method or callable changes.
        self.assertIs(helper.binding(self.layer, adapter), self.runner)
        mutation()
        with self.assertRaisesRegex(RuntimeError, message):
            helper.binding(self.layer, adapter)

    def replace_method(self, replacement):
        self.layer.quant_method = replacement
        self.runner._quant_method = replacement

    def test_original_owner_returns_exact_registered_runner(self):
        self.assertIs(helper.binding(self.layer, adapter), self.runner)

    def test_known_modular_wrapper_retains_captured_owner_and_kernel(self):
        wrapper = FusedMoEModularMethod(self.owner, self.kernel)
        self.replace_method(wrapper)
        self.assertIs(helper.binding(self.layer, adapter), self.runner)

    def test_equivalent_context_registry_copy_is_rejected(self):
        copied_registry = dict(self.registry)
        self.assertEqual(copied_registry, self.registry)
        self.assertIsNot(copied_registry, self.registry)
        self.assert_rejected_after_mutation(
            lambda: setattr(self.context, "no_compile_layers", copied_registry),
            "captured runner registry",
        )

    def test_missing_registered_runner_is_rejected(self):
        self.assert_rejected_after_mutation(
            lambda: self.registry.pop(self.layer.layer_name),
            "runner/routed-layer/method identity",
        )

    def test_runner_bound_to_another_routed_layer_is_rejected(self):
        self.assert_rejected_after_mutation(
            lambda: setattr(self.runner, "routed_experts", RoutedLayer()),
            "runner/routed-layer/method identity",
        )

    def test_stale_runner_method_is_rejected_even_if_value_equal(self):
        equivalent_owner = types.SimpleNamespace(**vars(self.owner))
        self.assertEqual(equivalent_owner, self.owner)
        self.assertIsNot(equivalent_owner, self.owner)
        self.assert_rejected_after_mutation(
            lambda: setattr(self.runner, "_quant_method", equivalent_owner),
            "runner/routed-layer/method identity",
        )

    def test_replacement_owner_is_rejected_even_if_runner_agrees(self):
        equivalent_owner = types.SimpleNamespace(**vars(self.owner))
        self.assertEqual(equivalent_owner, self.owner)
        self.assert_rejected_after_mutation(
            lambda: self.replace_method(equivalent_owner),
            "replacement of captured quantization method",
        )

    def test_duck_typed_modular_wrapper_is_rejected(self):
        lookalike = types.SimpleNamespace(
            old_quant_method=self.owner, moe_kernel=self.kernel,
            is_monolithic=False,
        )
        self.assert_rejected_after_mutation(
            lambda: self.replace_method(lookalike),
            "replacement of captured quantization method",
        )

    def test_known_wrapper_with_another_old_owner_is_rejected(self):
        equivalent_owner = types.SimpleNamespace(**vars(self.owner))
        wrapper = FusedMoEModularMethod(equivalent_owner, self.kernel)
        self.assert_rejected_after_mutation(
            lambda: self.replace_method(wrapper),
            "replacement of captured quantization method",
        )

    def test_replaced_kernel_is_rejected_even_if_value_equal(self):
        equivalent_kernel = types.SimpleNamespace(**vars(self.kernel))
        self.assertEqual(equivalent_kernel, self.kernel)
        self.assertIsNot(equivalent_kernel, self.kernel)
        self.assert_rejected_after_mutation(
            lambda: setattr(self.owner, "moe_kernel", equivalent_kernel),
            "kernel changed or is monolithic",
        )

    def test_monolithic_method_is_rejected(self):
        self.assert_rejected_after_mutation(
            lambda: setattr(self.owner, "is_monolithic", True),
            "kernel changed or is monolithic",
        )

    def test_same_named_kernel_implementation_is_rejected(self):
        impostor = type("FusedMoEKernelModularImpl", (), {})()
        self.assertEqual(type(impostor).__name__, FusedMoEKernelModularImpl.__name__)
        self.assert_rejected_after_mutation(
            lambda: setattr(self.kernel, "impl", impostor),
            "Expected modular FlashInfer CUTLASS experts",
        )

    def test_kernel_implementation_subclass_is_rejected(self):
        class ChangedKernelImpl(FusedMoEKernelModularImpl):
            pass

        self.assert_rejected_after_mutation(
            lambda: setattr(self.kernel, "impl", ChangedKernelImpl()),
            "Expected modular FlashInfer CUTLASS experts",
        )

    def test_same_named_experts_implementation_is_rejected(self):
        impostor = type("FlashInferExperts", (), {})()
        self.assertEqual(type(impostor).__name__, FlashInferExperts.__name__)
        self.assert_rejected_after_mutation(
            lambda: setattr(self.kernel, "fused_experts", impostor),
            "Expected modular FlashInfer CUTLASS experts",
        )

    def test_experts_implementation_subclass_is_rejected(self):
        class ChangedExperts(FlashInferExperts):
            pass

        self.assert_rejected_after_mutation(
            lambda: setattr(self.kernel, "fused_experts", ChangedExperts()),
            "Expected modular FlashInfer CUTLASS experts",
        )

    def test_bound_callable_bypass_is_rejected(self):
        self.assert_rejected_after_mutation(
            lambda: setattr(self.layer, "forward_modular", types.MethodType(
                bypass, self.layer
            )),
            "callable bypasses the adapter",
        )

    def test_wrong_expected_callable_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "callable bypasses the adapter"):
            helper.binding(self.layer, bypass)


if __name__ == "__main__":
    unittest.main()
