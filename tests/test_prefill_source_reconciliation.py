"""CPU source compatibility only; no runtime, device or numerical admission."""
import ast
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from megartx import prefill_plan as prefill


ROOT = Path(__file__).resolve().parents[1]
PAIR = ("src/megartx/m1_live.py", "src/megartx/vllm_scale_plugin.py")
sha = lambda data: hashlib.sha256(data).hexdigest()


def ast_signature(node):
    # Python 3.12 adds an empty type_params field absent in Python 3.10.
    node = copy.deepcopy(node)
    for item in ast.walk(node):
        if hasattr(item, "type_params") and item.type_params == []:
            del item.type_params
    return sha(ast.dump(node).encode())


def functions(path):
    result = {}
    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                result[name] = child
                visit(child, name + ".")
            else:
                visit(child, prefix)
    visit(ast.parse(path.read_text()))
    return result


class ReviewedOverlayTests(unittest.TestCase):
    def fixture(self, root):
        plan = prefill.read_json(ROOT / "docs/prefill/profile-plan.json")
        manifest = prefill.read_json(ROOT / "docs/prefill/source-binding.json")
        baseline = {PAIR[0]: b"baseline controller", PAIR[1]: b"baseline plugin",
                    "src/megartx/nvfp4_runtime.py": b"fixed quantizer"}
        candidate = {PAIR[0]: b"reviewed controller", PAIR[1]: b"reviewed plugin"}
        manifest["repo_files"] = {p: sha(v) for p, v in baseline.items()}
        manifest["reviewed_source_overlays"][0]["repo_files"] = {p: sha(v) for p, v in candidate.items()}
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        plan["binding"]["source_manifest_sha256"] = sha(path.read_bytes())
        for p, data in baseline.items():
            target = root / p
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return plan, path, baseline, candidate

    def test_original_and_complete_reviewed_pair_remain_cpu_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, path, baseline, candidate = self.fixture(root)
            for values in (baseline, {**baseline, **candidate}):
                for name, data in values.items():
                    (root / name).write_bytes(data)
                self.assertEqual(prefill.verify_source_binding(plan, path, root), 3)
                self.assertFalse(prefill.intake(plan)["gpu_execution_available"])
                self.assertFalse(prefill.intake(plan)["gpu_qualified"])

    def test_partial_pair_unknown_controller_and_quantizer_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, path, baseline, candidate = self.fixture(root)
            variants = [{**baseline, PAIR[0]: candidate[PAIR[0]]},
                        {**baseline, PAIR[1]: candidate[PAIR[1]]},
                        {**baseline, **candidate, PAIR[0]: b"unknown controller"},
                        {**baseline, **candidate, "src/megartx/nvfp4_runtime.py": b"changed quantizer"}]
            for values in variants:
                for name, data in values.items():
                    (root / name).write_bytes(data)
                with self.assertRaisesRegex(ValueError, "Source drift"):
                    prefill.verify_source_binding(plan, path, root)

    def test_reviewed_math_and_capture_boundaries_equal_original_ast(self):
        path = ROOT / "docs/prefill/pr15-source-reconciliation.json"
        review = prefill.read_json(path)
        manifest = prefill.read_json(ROOT / "docs/prefill/source-binding.json")
        self.assertEqual(sha(path.read_bytes()), manifest["source_review_sha256"])
        self.assertFalse(review["gpu_qualified"])
        self.assertEqual(review["final_pr15_head"], manifest["reviewed_source_overlays"][0]["source_head"])
        sources = {name: functions(ROOT / name) for name in PAIR}
        for location, expected in review["baseline_equal_ast_boundaries"].items():
            name, function = location.split("::")
            self.assertEqual(ast_signature(sources[name][function]), expected, location)
        boundary = review["routed_math_boundary"]
        routed = copy.deepcopy(sources[PAIR[1]][boundary["name"]])
        approved = boundary["approved_added_counter_ast_sha256"]
        class RemoveExactCounter(ast.NodeTransformer):
            def visit_If(self, node):
                if ast_signature(node) == approved:
                    return None
                return self.generic_visit(node)
        routed = RemoveExactCounter().visit(routed)
        self.assertEqual(ast_signature(routed), boundary["baseline_ast_sha256"])

    def test_multirow_stock_fallback_and_candidate_benchmark_shape_guard(self):
        # Existing CPU lease fixtures exercise Python dispatch only, no CUDA work.
        import test_m1_live as lease_tests
        fixture = lease_tests.TestLiveLease()
        review = prefill.read_json(ROOT / "docs/prefill/pr15-source-reconciliation.json")
        candidate = sha((ROOT / PAIR[0]).read_bytes()) == review["files"][PAIR[0]]["candidate_sha256"]
        for rows in (255, 256, 512):
            with tempfile.TemporaryDirectory() as directory:
                native = lease_tests.Native()
                controller = fixture.capture_free(fixture.controller(directory, native))
                kwargs = fixture.kwargs()
                kwargs["input"].shape = (rows, 1408)
                calls = []
                with patch.dict("sys.modules", fixture.modules()):
                    self.assertEqual(controller.invoke(lambda **kw: calls.append(kw) or 9, (), kwargs), 9)
                self.assertEqual(calls, [kwargs])
                self.assertEqual(native.begins, [])
                if candidate:
                    controller.benchmark = SimpleNamespace(fallback=Mock(), backend=Mock())
                    with patch.dict("sys.modules", fixture.modules()):
                        if rows == 256:
                            self.assertEqual(controller.invoke(lambda **kw: 9, (), kwargs), 9)
                            controller.benchmark.fallback.assert_called_once()
                        else:
                            with self.assertRaisesRegex(RuntimeError, "geometry/capture"):
                                controller.invoke(lambda **kw: self.fail("unsupported benchmark submitted"), (), kwargs)
                            controller.benchmark.fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
