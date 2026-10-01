"""CPU-only checks of the actual one-request observer's fail-closed guards.

Small tensor API stubs establish request bounds/identity and completeness;
they establish neither live hook invocation nor CUDA numerical correctness.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "src/megartx/router_score_capture.py"
spec = importlib.util.spec_from_file_location("_router_capture_guard_subject", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Rows:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.int64)
        self.ndim = self.data.ndim

    def numel(self):
        return self.data.size

    def detach(self):
        return self

    long = cpu = detach

    def numpy(self):
        return self.data


class CaptureGuards(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.observer = module.RouterScoreCapture.__new__(module.RouterScoreCapture)
        self.observer.marker = Path(temporary.name) / "capture-request.json"
        self.observer.active = None
        self.observer.pending = {}
        self.observer.request_identity = None
        self.observer.forward_id = 0
        self.tokens = [123] * 1025
        self.marker = {"id": "one-prefix", "prompt_token_ids": self.tokens,
                       "prompt_sha256": hashlib.sha256(json.dumps(self.tokens).encode()).hexdigest()}
        self.write_marker()

    def write_marker(self):
        self.observer.marker.write_text(json.dumps(self.marker))

    def begin(self, tokens=None, positions=None):
        return self.observer.begin(Rows([123] if tokens is None else tokens), Rows([0] if positions is None else positions))

    def test_exact_prefix_activates_one_context(self):
        self.begin([123] * 256, range(256))
        self.assertEqual(self.observer.active["forward_id"], 0)
        self.assertEqual(self.observer.request_identity, ("one-prefix", self.marker["prompt_sha256"]))

    def test_startup_without_marker_is_excluded(self):
        self.observer.marker.unlink()
        self.begin()
        self.assertIsNone(self.observer.active)
        self.assertEqual(self.observer.forward_id, 0)

    def test_prefix_length_and_hash_are_required(self):
        for changed in ({"prompt_token_ids": self.tokens[:-1]}, {"prompt_sha256": "0" * 64}):
            with self.subTest(changed=changed):
                saved = dict(self.marker)
                self.marker.update(changed)
                self.write_marker()
                with self.assertRaisesRegex(RuntimeError, "exact 1025-token"):
                    self.begin()
                self.marker = saved

    def test_prefix_token_identity_is_required(self):
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.begin([124], [0])

    def test_position_bounds_are_required(self):
        for position in (-1, 1033):
            with self.subTest(position=position), self.assertRaisesRegex(RuntimeError, "bounded decode"):
                self.begin([123], [position])

    def test_missing_token_ids_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "verified unpadded"):
            self.observer.begin(None, Rows([0]))

    def test_more_than_one_chunk_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "verified unpadded"):
            self.begin([123] * 257, range(257))

    def test_token_and_position_row_counts_must_match(self):
        with self.assertRaisesRegex(RuntimeError, "verified unpadded"):
            self.begin([123, 123], [0])

    def test_a_different_request_identity_is_rejected(self):
        self.observer.request_identity = ("different-request", self.marker["prompt_sha256"])
        with self.assertRaisesRegex(RuntimeError, "one request"):
            self.begin()

    def test_forward_count_is_bounded(self):
        self.observer.forward_id = 16
        with self.assertRaisesRegex(RuntimeError, "one request"):
            self.begin()

    def test_unconsumed_context_is_rejected(self):
        self.begin()
        with self.assertRaisesRegex(RuntimeError, "not consumed"):
            self.begin()

    def test_pending_router_is_rejected(self):
        self.observer.pending[0] = {}
        with self.assertRaisesRegex(RuntimeError, "not consumed"):
            self.begin()

    def test_missing_natural_layer_fails_end(self):
        self.begin()
        self.observer.active["seen"] = {0, 1, 2, 3}
        with self.assertRaisesRegex(RuntimeError, "bypassed"):
            self.observer.end()

    def test_unconsumed_stage_fails_end(self):
        self.begin()
        self.observer.active["seen"] = set(module.TARGETS)
        self.observer.pending[0] = {}
        with self.assertRaisesRegex(RuntimeError, "bypassed"):
            self.observer.end()

    def test_complete_natural_capture_clears_context(self):
        self.begin()
        self.observer.active["seen"] = set(module.TARGETS)
        self.observer.end()
        self.assertIsNone(self.observer.active)


if __name__ == "__main__":
    unittest.main()
