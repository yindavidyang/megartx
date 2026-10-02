"""Synchronous compiled C ABI controls; no model data or CUDA dispatch."""
import argparse
import ctypes
import json
from pathlib import Path


def check(library):
    class View(ctypes.Structure):
        _fields_ = [("pointer", ctypes.c_void_p), ("bytes", ctypes.c_uint64),
                    ("storage", ctypes.c_void_p), ("storage_bytes", ctypes.c_uint64)]
    native = ctypes.CDLL(str(library.resolve()))
    native.megartx_m1_contract_v2.restype = ctypes.c_char_p
    contract = json.loads(native.megartx_m1_contract_v2().decode())
    assert (contract["abi_version"], contract["view_count"], contract["view_bytes"]) == (2, 15, ctypes.sizeof(View))
    assert not hasattr(native, "megartx_m1_begin")
    assert contract["execution_modes"] == ["captured", "capture-free"]
    assert contract["capture_free_begin"] == "megartx_m1_begin_capture_free_v2"
    native.megartx_m1_begin_v2.argtypes = [ctypes.c_uint32, ctypes.POINTER(View), ctypes.c_uint32,
                                         ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p]
    native.megartx_m1_begin_capture_free_v2.argtypes = native.megartx_m1_begin_v2.argtypes[:-1]
    for name in ("megartx_m1_begin_v2", "megartx_m1_begin_capture_free_v2", "megartx_m1_end", "megartx_m1_active"):
        getattr(native, name).restype = ctypes.c_int
    native.megartx_m1_error.restype = native.megartx_m1_metadata.restype = ctypes.c_char_p
    # Deliberately invalid CPU address is safe only if framing is checked first.
    poison = ctypes.cast(ctypes.c_void_p(1), ctypes.POINTER(View))
    for version, count, size in ((1, 15, 32), (2, 7, 32), (2, 16, 32), (2, 15, 16)):
        assert native.megartx_m1_begin_v2(version, poison, count, size, 7, 1, b"unused") == -1
        assert native.megartx_m1_begin_capture_free_v2(version, poison, count, size, 7, 1) == -1
        assert not native.megartx_m1_active()
    views = (View * 15)(*(View(0x10000000 + i * 0x1000000,
                              3185408 if i == 5 else 32000,
                              0x10000000 + i * 0x1000000,
                              3185408 if i == 5 else 32000) for i in range(15)))
    begins = (lambda *args: native.megartx_m1_begin_v2(*args, b"unused-control-directory"),
              native.megartx_m1_begin_capture_free_v2)
    for begin in begins:
        for cycle in range(3):
            assert begin(2, views, 15, ctypes.sizeof(View), 7, 1) == 0
            assert native.megartx_m1_active() == 1
            for nested in begins:
                assert nested(2, views, 15, ctypes.sizeof(View), 7, 1) == -1
                assert native.megartx_m1_active() == 1  # The outer lease must survive rejection.
            assert native.megartx_m1_end() == -1   # No runner was executed.
            assert native.megartx_m1_active() == 0
    views[0].bytes = 32001
    assert native.megartx_m1_begin_v2(2, views, 15, ctypes.sizeof(View), 7, 1, b"unused-control-directory") == -1
    assert native.megartx_m1_begin_capture_free_v2(2, views, 15, ctypes.sizeof(View), 7, 1) == -1
    assert native.megartx_m1_active() == 0
    views[0].bytes = 32000
    assert native.megartx_m1_begin_v2(2, views, 15, ctypes.sizeof(View), 7, 1, None) == -1
    assert native.megartx_m1_active() == 0
    for stream, opt_in in ((0, 1), (7, 2)):
        assert native.megartx_m1_begin_v2(2, views, 15, ctypes.sizeof(View), stream, opt_in, b"unused-control-directory") == -1
        assert native.megartx_m1_begin_capture_free_v2(2, views, 15, ctypes.sizeof(View), stream, opt_in) == -1
        assert native.megartx_m1_active() == 0
    return {"scope": "compiled C ABI controls with synthetic CPU addresses; no CUDA dispatch",
            "live_contract": contract, "invalid_framing_before_dereference": True,
            "historical_begin_symbol_absent": True, "repeat_cycles": 3, "nested_rejection_preserves_outer": True,
            "extent_rejection_restores_inactive_state": True, "null_stream_rejected": True,
            "invalid_optin_rejected": True, "captured_null_directory_rejected": True,
            "execution_modes_tested": ["captured", "capture-free"],
            "cross_mode_nested_rejection_preserves_outer": True,
            "active_after": native.megartx_m1_active()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = check(args.library)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)
