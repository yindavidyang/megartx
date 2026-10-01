"""CPU-only rejection checks; fake tensors do not assert live GPU execution."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import controlled_reference as ref
from megartx.controlled_capture import ControlledCapture, ORIGIN, save_npz


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.ndim = self.value.ndim
    def numel(self):
        return self.value.size
    def long(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.value


class CaptureGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.plan = ref.make_schedule(np.arange(33, dtype=np.int64))
        self.source = self.root / "plan"
        self.source.mkdir()
        self.manifest = {k: v for k, v in self.plan.items() if k not in {"tokens", "ids", "weight_bits"}}
        (self.source / "manifest.json").write_text(json.dumps(self.manifest))
        save_npz(self.source / "route-table.npz", {k: self.plan[k] for k in ("tokens", "ids", "weight_bits")})
        self.layers = [SimpleNamespace(layer_name=f"model.layers.{i}.mlp", _megartx={"ordinal": i}) for i in range(30)]
        self.env = patch.dict(os.environ, {"MEGARTX_CONTROLLED_DIR": str(self.root / "controlled"), "MEGARTX_CONTROLLED_PLAN": str(self.source)})
        self.env.start()
        fake_kv = SimpleNamespace(ControlledKV=lambda *a: SimpleNamespace())
        self.modules = patch.dict(sys.modules, {"megartx.controlled_kv_capture": fake_kv, "torch": SimpleNamespace()})
        self.modules.start()
    def tearDown(self):
        self.modules.stop()
        self.env.stop()
        self.tmp.cleanup()
    def capture(self, layers=None, mode="native"):
        return ControlledCapture(object(), self.layers if layers is None else layers, mode)
    def request(self, **changes):
        record = {**ORIGIN, "controlled_path": "full", "schedule_sha256": self.plan["schedule_sha256"], **changes}
        (self.root / "capture-request.json").write_text(json.dumps(record))
    def test_exact_origin_and_schedule_retained(self):
        cap = self.capture()
        self.assertEqual(cap.plan["scope"], ref.CONTROLLED_ORIGIN["scope"])
        self.assertEqual(cap.plan["schedule_sha256"], self.plan["schedule_sha256"])
    def test_unmarked_startup_inactive(self):
        cap = self.capture()
        cap.begin(None, None)
        self.assertFalse(cap.active)
    def test_manifest_natural_relabel_rejected(self):
        self.manifest["route_origin"] = "natural"
        (self.source / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):
            self.capture()
    def test_duplicate_layer_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            self.capture(self.layers + [self.layers[0]])
    def test_checkpoint_runtime_offset_rejected(self):
        self.layers[0]._megartx["ordinal"] = 1
        with self.assertRaisesRegex(RuntimeError, "ordinals"):
            self.capture()
    def test_missing_layer_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "thirty"):
            self.capture(self.layers[:-1])
    def test_unmatched_reference_lane_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "arithmetic"):
            self.capture(mode="reference")
    def test_changed_route_hash_rejected_before_forward(self):
        cap = self.capture()
        self.request(schedule_sha256="0" * 64)
        with self.assertRaisesRegex(RuntimeError, "plan"):
            cap.begin(Tensor(np.arange(33)), Tensor(np.arange(33)))
    def test_typed_origin_rejected_before_forward(self):
        cap = self.capture()
        self.request(routing_intervention=1)
        with self.assertRaisesRegex(RuntimeError, "origin"):
            cap.begin(Tensor(np.arange(33)), Tensor(np.arange(33)))
    def test_wrong_actual_token_rejected_before_forward(self):
        cap = self.capture()
        self.request()
        with self.assertRaisesRegex(RuntimeError, "tokens"):
            cap.begin(Tensor(np.ones(33, dtype=np.int64)), Tensor(np.arange(33)))
    def test_repeated_logical_position_rejected(self):
        cap = self.capture()
        self.request()
        with self.assertRaisesRegex(RuntimeError, "tokens"):
            cap.begin(Tensor([0, 0]), Tensor([0, 0]))
    def test_artifact_overwrite_rejected(self):
        path = self.root / "kept.npz"
        save_npz(path, {"input": np.arange(3)})
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_npz(path, {"input": np.zeros(3)})
        self.assertEqual(original, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
