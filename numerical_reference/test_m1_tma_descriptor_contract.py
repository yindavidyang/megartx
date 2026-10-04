"""CPU control-flow tests of the exact live TMA boundary, with no CUDA import.

The real setup method and contract helpers are extracted byte-for-byte. Fake
provider/device APIs exercise branches, not CUDA execution, ABI, or correctness.
"""
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def parent_bridge_source(source):
    """Undo only this descriptor delta; retain the older exact-delta guard."""
    for name in ("INPUT", "OUTPUT"):
        pattern = (r"  // BEGIN M1 SOURCE-BOUND TMA " + name +
                   r" CONTRACT\n.*?  // END M1 SOURCE-BOUND TMA " + name + r" CONTRACT\n")
        source, count = re.subn(pattern, "", source, flags=re.S)
        if count != 1:
            raise AssertionError("exactly one " + name + " contract block required")
    source = source.replace("  // Diagnostic descriptor checks remain mandatory. These D2H\n",
                            "  // Descriptor checks remain mandatory in both execution modes. These D2H\n")
    return source


def _definition(source, start):
    """Balance C++ braces after masking comments and ordinary string literals."""
    token = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/'
    masked = re.sub(token, lambda match: " " * len(match.group()), source, flags=re.S)
    opening = masked.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return source[start:end]


def extracted_source(source):
    contains = _definition(source, source.index("bool contains(View const&"))
    start = source.index('template<> __attribute__((visibility("default"))) std::pair<Desc,Desc> Runner::setupTmaWarpSpecializedInputs(')
    method = _definition(source, start)
    # Only remove specialization syntax: the test Runner is a concrete class.
    method = method.removeprefix("template<> ")
    helpers = ""
    marker = "// BEGIN M1 SOURCE-BOUND TMA HELPER CONTRACT\n"
    if marker in source:
        helpers = source.split(marker, 1)[1].split("// END M1 SOURCE-BOUND TMA HELPER CONTRACT", 1)[0]
    return contains + "\n" + helpers + "\n" + method + "\n"


class TmaDescriptorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++") or shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("CPU C++ compiler unavailable")
        from test_m1_sf_layout_contract import MOCK_LAYOUT
        cls.temp = tempfile.TemporaryDirectory()
        cls.work = Path(cls.temp.name)
        (cls.work / "cute").mkdir()
        (cls.work / "cute/layout.hpp").write_text(MOCK_LAYOUT)
        (cls.work / "m1_sf_layout_contract.hpp").write_bytes((ROOT / "kernels/m1_sf_layout_contract.hpp").read_bytes())
        source = (ROOT / "probes/m1_live_bridge.cu").read_text()
        (cls.work / "extracted_tma.inc").write_text(extracted_source(source))
        cls.exe = cls.work / "tma-flow"
        result = subprocess.run([compiler, "-std=c++17", "-O1", "-Wall", "-Wextra",
                                 "-I", str(cls.work), str(ROOT / "probes/m1_tma_descriptor_flow_test.cpp"),
                                 "-o", str(cls.exe)], capture_output=True, text=True, timeout=60)
        if result.returncode:
            cls.temp.cleanup()
            raise AssertionError(result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def control(self, name):
        result = subprocess.run([str(self.exe), name, str(self.work / name)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, name + ": " + result.stdout + result.stderr)

    def test_production_has_no_descriptor_copies_or_fences(self):
        self.control("production")
        self.control("fresh")

    def test_both_diagnostic_lanes_keep_fourteen_copies_two_fences(self):
        self.control("observer")
        self.control("captured")

    def test_host_input_owner_and_member_drift_stop_before_provider(self):
        self.control("preflight")

    def test_returned_host_descriptor_drift_stops_before_consumer(self):
        self.control("returned")

    def test_pending_errors_are_not_cleared_or_fallen_back(self):
        self.control("errors")

    def test_unbound_and_unqualified_calls_delegate_unchanged(self):
        self.control("unsupported")

    def test_diagnostics_reject_actual_device_metadata_drift(self):
        self.control("device")

    def test_descriptor_delta_normalizes_to_exact_parent(self):
        source = (ROOT / "probes/m1_live_bridge.cu").read_text()
        expected = "a576e25aaafa67941eccb9a7cd5ce42adca6138c1d10519e50844f0ea9493cbd"
        self.assertEqual(hashlib.sha256(parent_bridge_source(source).encode()).hexdigest(), expected)
        changed = source.replace('rows!=1 || hidden!=2816', 'rows!=2 || hidden!=2816')
        self.assertNotEqual(hashlib.sha256(parent_bridge_source(changed).encode()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
