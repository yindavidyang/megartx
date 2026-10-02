"""CPU stubs verify retained controlled checks and disabled tensor diagnostics."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import controlled_reference as ref
from megartx.controlled_capture import ControlledCapture, ORIGIN, save_npz
from megartx.controlled_kv_capture import ControlledKV, SlotLedger


class Tensor:
    def __init__(self, value, dtype="f32"):
        self.value, self.dtype, self.device = np.asarray(value), dtype, "cpu-stub"
    @property
    def ndim(self): return self.value.ndim
    @property
    def shape(self): return self.value.shape
    def numel(self): return self.value.size
    def long(self): return self
    def int(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def clone(self): return Tensor(self.value.copy(), self.dtype)
    def all(self): return Tensor(self.value.all())
    def sum(self): return Tensor(self.value.sum())
    def item(self): return self.value.item()
    def __bool__(self): return bool(self.value)
    def __ne__(self, other): return Tensor(self.value != other)
    def __gt__(self, other): return Tensor(self.value > other)
    def __getitem__(self, key): return Tensor(self.value[key], self.dtype)


class CaptureFreeChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.root = Path(self.tmp.name)
        self.plan = ref.make_schedule(np.arange(33, dtype=np.int64))
        source = self.root / "plan";source.mkdir()
        (source/"manifest.json").write_text(json.dumps({k:v for k,v in self.plan.items() if k not in {"tokens", "ids", "weight_bits"}}))
        save_npz(source/"route-table.npz", {k:self.plan[k] for k in ("tokens", "ids", "weight_bits")})
        self.layers = [SimpleNamespace(layer_name=f"model.layers.{i}.mlp", _megartx={"ordinal": i}) for i in range(30)]
        self.kv = SimpleNamespace(begin_case=Mock(), capture=Mock(), finish_case=Mock())
        def make_kv(*args, **kwargs):
            self.assertIs(kwargs.get("diagnostics"), False)
            return self.kv
        torch = SimpleNamespace(bfloat16="bf16", float32="f32",
            as_tensor=lambda array, device=None, dtype="f32": Tensor(array, dtype),
            isfinite=lambda value: Tensor(np.isfinite(value.value)),
            profiler=SimpleNamespace(profile=lambda **kw: self.fail("disabled request profiler")),
            cuda=SimpleNamespace(synchronize=lambda: self.fail("diagnostic request fence")))
        self.env = patch.dict("os.environ", {"MEGARTX_CONTROLLED_DIR": str(self.root/"controlled"),
                                            "MEGARTX_CONTROLLED_PLAN": str(source)}, clear=True)
        self.modules = patch.dict("sys.modules", {"torch": torch,
                                 "megartx.controlled_kv_capture": SimpleNamespace(ControlledKV=make_kv)})
        self.env.start();self.modules.start()
        self.cap = ControlledCapture(object(), self.layers, "native", execution_mode="capture-free")
        (self.root/"capture-request.json").write_text(json.dumps({**ORIGIN,
            "controlled_path": "cached", "schedule_sha256": self.plan["schedule_sha256"]}))

    def tearDown(self):
        self.modules.stop();self.env.stop();self.tmp.cleanup()

    def begin(self, positions):
        self.cap.begin(Tensor(positions, "i64"), Tensor(positions, "i64"))

    def test_routes_are_the_identical_plan_tensors_without_npz_or_trace(self):
        self.begin(list(range(32)))
        with patch("megartx.controlled_capture.save_npz", side_effect=AssertionError("NPZ")), \
             patch("megartx.controlled_capture.bits", side_effect=AssertionError("tensor bits")):
            for i, layer in enumerate(self.layers):
                ids, weights = self.cap.routes(layer, Tensor(np.zeros((32, 2816)), "bf16"),
                    Tensor(np.zeros((32, 8), dtype=np.int32), "i32"), Tensor(np.zeros((32, 8))))
                np.testing.assert_array_equal(ids.value, self.plan["ids"][i, :32])
                np.testing.assert_array_equal(weights.value.view(np.uint32), self.plan["weight_bits"][i, :32])
            self.cap.end()
        self.kv.capture.assert_called_once()
        self.assertIsNone(self.cap.profiler)
        self.assertFalse((self.root/"controlled/cached").exists())

    def test_missing_duplicate_hooks_and_wrong_shapes_still_fail(self):
        self.begin([32]);layer = self.layers[0]
        args = (layer, Tensor(np.zeros((1, 2816)), "bf16"), Tensor(np.zeros((1, 8)), "i32"), Tensor(np.zeros((1, 8))))
        self.cap.routes(*args)
        with self.assertRaisesRegex(RuntimeError, "duplicated"): self.cap.routes(*args)
        with self.assertRaisesRegex(RuntimeError, "bypassed"): self.cap.end()
        self.begin([32])
        with self.assertRaisesRegex(RuntimeError, "shape or dtype"):
            self.cap.routes(layer, Tensor(np.zeros((1, 2816)), "f32"), args[2], args[3])

    def expert(self, expert=82, down=None, weights=None):
        return self.cap.expert(self.layers[0], SimpleNamespace(index=expert), Tensor([0]), Tensor([0]),
            Tensor([1.] if weights is None else weights), {"down": Tensor([1.] if down is None else down)}, None, None)

    def test_expert_live_validation_remains_and_all_stage_serialization_is_absent(self):
        self.begin([32])
        with patch("megartx.controlled_capture.bits", side_effect=AssertionError("stage copy")), \
             patch("megartx.controlled_capture.save_npz", side_effect=AssertionError("stage NPZ")):
            self.expert()
        self.assertEqual(self.cap.stages, [{"layer": 0, "expert": 82}])
        with self.assertRaisesRegex(RuntimeError, "twice"): self.expert()

    def test_nonfinite_zero_negative_weight_and_wrong_expert_row_are_rejected(self):
        self.begin([32])
        for args in ({"down": [np.nan]}, {"down": [0.]}, {"weights": [0.]}, {"weights": [-1.]}, {"expert": 42}):
            with self.subTest(args=args), self.assertRaises(RuntimeError): self.expert(**args)
        self.assertEqual(self.cap.stages, [])

    def test_end_retains_kv_validation_and_complete_correction_set(self):
        self.begin([32]);self.cap.context["seen"] = set(range(30))
        self.cap.routes_seen = {i:list(range(33)) for i in range(30)}
        self.cap.stages = [{"layer":i,"expert":e} for i,e in ref.TARGETS]
        self.cap.end()
        self.kv.capture.assert_called_once();self.kv.finish_case.assert_called_once()
        self.assertTrue(self.cap.complete);self.assertIsNone(self.cap.context)
        self.assertFalse((self.root/"controlled/cached").exists())

    def test_failed_kv_validation_and_missing_correction_propagate(self):
        self.begin([32]);self.cap.context["seen"] = set(range(30))
        self.kv.capture.side_effect = RuntimeError("K/V owner changed")
        with self.assertRaisesRegex(RuntimeError, "K/V owner changed"): self.cap.end()
        self.assertIsNone(self.cap.context);self.assertFalse(self.cap.complete)
        self.kv.capture.side_effect = None
        self.begin([32]);self.cap.context["seen"] = set(range(30))
        self.cap.routes_seen = {i:list(range(33)) for i in range(30)}
        with self.assertRaisesRegex(RuntimeError, "positive correction"): self.cap.end()

    def test_logits_keep_shape_finite_token_and_both_position_checks_without_capture(self):
        self.begin(list(range(32)))
        with patch("megartx.controlled_capture.save_npz", side_effect=AssertionError("logit NPZ")), \
             patch.object(Path, "open", side_effect=AssertionError("logit record")):
            self.cap.logits(Tensor(np.zeros((1, 262144))), [31], [31], "verified_storage_view", 32)
            self.cap.complete = True
            self.cap.logits(Tensor(np.zeros((1, 262144))), [32], [32], "verified_storage_view", 1)
        self.assertEqual(self.cap.logit_positions, {31,32})
        for tensor, tokens in ((Tensor(np.zeros((1, 10))), [32]),
                               (Tensor(np.full((1, 262144), np.nan)), [32]),
                               (Tensor(np.zeros((1, 262144))), [99])):
            with self.assertRaises(RuntimeError): self.cap.logits(tensor, [32], tokens, "verified_storage_view", 1)
        self.cap.logit_positions = set()
        with self.assertRaisesRegex(RuntimeError, "Both actual input positions"):
            self.cap.logits(Tensor(np.zeros((1,262144))), [32], [32], "verified_storage_view", 1)

    def test_abort_performs_no_evidence_write(self):
        self.begin([32])
        with patch.object(Path, "write_text", side_effect=AssertionError("invalidation evidence")):
            self.cap.abort(RuntimeError("model failed"))
        self.assertIsNone(self.cap.context)

    def test_capture_free_cannot_expand_to_other_controlled_paths(self):
        for path in ("full", "chunked"):
            (self.root/"capture-request.json").write_text(json.dumps({**ORIGIN,
                "controlled_path": path, "schedule_sha256": self.plan["schedule_sha256"]}))
            with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, "bounded cached"):
                self.begin([32])


class KVSerializationPolicy(unittest.TestCase):
    def kv(self, directory, complete=True):
        obj = ControlledKV.__new__(ControlledKV)
        obj.diagnostics, obj.directory = False, Path(directory)/"cached"
        obj.ledger = SlotLedger(tuple(range(33)))
        if complete:
            obj.ledger.record(tuple(range(33)), tuple(range(33)), {i:list(range(33)) for i in range(30)},
                              {i:33 for i in range(30)}, {i:(i,"retained") for i in range(30)})
        obj.snapshots = [(np.zeros(1),np.zeros(1)) for _ in range(30)] if complete else None
        return obj

    def test_validated_kv_snapshots_are_not_serialized(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = self.kv(directory)
            with patch.object(Path, "open", side_effect=AssertionError("KV evidence")), \
                 patch("numpy.savez_compressed", side_effect=AssertionError("KV NPZ")):
                self.assertEqual(obj.finish_case(), [])
            self.assertIsNone(obj.directory)
            self.assertFalse(list(Path(directory).iterdir()))

    def test_disabled_kv_serialization_cannot_skip_complete_slot_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "committed logical positions"):
                self.kv(directory, complete=False).finish_case()


if __name__ == "__main__": unittest.main()
