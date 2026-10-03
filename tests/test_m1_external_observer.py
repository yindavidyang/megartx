"""CPU protocol tests for the opt-in capture-free M1 sidecar."""
import ctypes
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from megartx.m1_external_observer import ExternalObserver, SCHEMA


CUPTI_PROVIDER = {
    "distribution": "nvidia-cuda-cupti", "version": "13.0.85",
    "library_name": "libcupti.so.13",
    "library_sha256": "e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e",
}


class Function:
    def __init__(self):
        self.callback = None

    def __call__(self, callback):
        self.callback = callback
        return 0


class Native:
    def __init__(self):
        self.megartx_m1_set_external_observer_v1 = Function()


class ExternalObserverTests(unittest.TestCase):
    def test_capture_free_begin_end_and_bounded_payload_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            observer = ExternalObserver(Native(), Path(directory) / "observer",
                                        {"abi_version": 2, "cupti_stream_id_provider": CUPTI_PROVIDER},
                                        stream_id_query=lambda handle: 901)
            frame = {"request_id": "controlled-cached", "positions": [32]}
            observer.begin_call(0, "fused", 77, frame)
            callback = observer._callback_ref
            self.assertEqual(callback(b"lease_begin", b"fused", None, 0, 77, 0), 0)
            payload = ctypes.create_string_buffer(b"i" * 32)
            self.assertEqual(callback(b"payload", b"ids.bin", ctypes.addressof(payload), 32, 77, 0), 0)
            status = ctypes.create_string_buffer(b'{"backend":"fused"}')
            self.assertEqual(callback(b"candidate_status", b"preparation.json",
                ctypes.addressof(status),
                len(b'{"backend":"fused"}'), 77, 1), 0)
            self.assertEqual(callback(b"lease_end", b"fused", None, 0, 77, 1), 0)
            receipt = observer.finish_call(1, True)
            self.assertFalse(receipt["internal_capture_enabled"])
            self.assertEqual(receipt["event_counts"]["payload"], 1)
            self.assertEqual(receipt["stream"], 77)
            self.assertEqual(receipt["profiler_stream_id"], 901)
            self.assertEqual(receipt["stream_id_api"], "cuptiGetStreamIdEx")
            self.assertEqual(receipt["cupti_stream_id_provider"], CUPTI_PROVIDER)
            observer_contract = json.loads((observer.root / "observer-contract.json").read_text())
            self.assertEqual(observer_contract["cupti_stream_id_provider"], CUPTI_PROVIDER)
            self.assertEqual((observer.root / "calls/call-0000/payloads/ids.bin").read_bytes(), b"i" * 32)

    def test_thread_stream_or_event_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            observer = ExternalObserver(Native(), Path(directory) / "observer",
                                        {"cupti_stream_id_provider": CUPTI_PROVIDER},
                                        stream_id_query=lambda handle: 901)
            observer.begin_call(0, "stock", 91, {})
            callback = observer._callback_ref
            self.assertEqual(callback(b"lease_begin", b"stock", None, 0, 92, 0), -1)
            self.assertIsNotNone(observer.callback_error)
            observer.finish_call(-1, False, RuntimeError("native begin failed"))

    def test_model_serialization_rejects_object_arrays_and_marks_perturbation(self):
        with tempfile.TemporaryDirectory() as directory:
            observer = ExternalObserver(Native(), Path(directory) / "observer",
                                        {"cupti_stream_id_provider": CUPTI_PROVIDER},
                                        stream_id_query=lambda handle: 901)
            observer.begin_case("cached", {"token_sha256": "a" * 64})
            with self.assertRaisesRegex(RuntimeError, "object arrays"):
                observer.write_model_npz("routes", "bad.npz", {"unsafe": np.asarray([object()])}, {})
            observer.write_model_npz("routes", "routes-00-00.npz",
                                     {"ids": np.asarray([[1]], dtype=np.int32)}, {"layer": 0})
            with gzip.open(observer.root / "controlled-trace.json.gz", "wb") as stream:
                stream.write(json.dumps({"traceEvents": []}).encode())
            observer.next_call = 30  # Call count is covered by native bridge protocol tests.
            manifest = observer.finish_case(observer.root / "controlled-trace.json.gz")
            self.assertEqual(manifest["schema"], SCHEMA)
            self.assertTrue(manifest["observer_perturbs_execution"])
            self.assertFalse(manifest["timing_qualified"])
            self.assertTrue((observer.root / "observer-manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
