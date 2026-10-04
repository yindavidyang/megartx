"""CPU lifetime tests of the exact native Invocation and synchronous runMoe.

Only the template-specialization prefix is removed for the concrete host Runner.
The map is a real std::map, not a behavioral rewrite or cached fixture. Small
host stubs replace provider/device work; this is not CUDA, ABI, or GPU evidence.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_m1_tma_descriptor_contract import _definition


ROOT = Path(__file__).resolve().parents[1]
RUNNER_MARKER = 'template<> __attribute__((visibility("default"))) void Runner::runMoe('


def definition(source, marker, semicolon=False):
    if source.count(marker) != 1:
        raise AssertionError("exactly one production definition required: " + marker)
    body = _definition(source, source.index(marker))
    if semicolon:
        if source[source.index(marker) + len(body)] != ";":
            raise AssertionError("production declaration must end in a semicolon")
        body += ";"
    return body


def extracted_sources(source):
    declarations = "\n".join([
        definition(source, "struct View {", True),
        definition(source, "struct Lease {", True),
        "thread_local Lease lease;",
        definition(source, "struct Invocation {", True),
        "thread_local Invocation* invocation=nullptr;",
    ]) + "\n"
    # Match the original TLS declarations exactly; neither is a test replacement.
    for line in ("thread_local Lease lease;", "thread_local Invocation* invocation=nullptr;"):
        if source.count(line) != 1:
            raise AssertionError("exact production thread-local declaration required")
    functions = "\n".join([
        definition(source, "bool contains(View const&"),
        definition(source, "void require(bool yes,char const* error)"),
        definition(source, "void observe(char const* event,"),
        definition(source, "int begin_lease("),
        definition(source, RUNNER_MARKER).removeprefix("template<> "),
    ]) + "\n"
    buffers = (ROOT / "kernels/m1_maps_expand.cuh").read_text()
    preparation = (ROOT / "kernels/m1_installed_preparation.cuh").read_text()
    calls = "namespace mx {\n" + "\n".join([
        definition(buffers, "struct M1Buffers {", True),
        definition(preparation, "enum class Fc1InputLane {", True),
        definition(preparation, "struct InstalledPreparationCall {", True),
    ]) + "\n}\n"
    return declarations, functions, calls


class InvocationRegionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("c++") or shutil.which("g++")
        if not cls.compiler:
            raise unittest.SkipTest("CPU C++ compiler unavailable")
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.work = Path(cls.temp.name)
        cls.source = (ROOT / "probes/m1_live_bridge.cu").read_text()
        cls.parts = extracted_sources(cls.source)
        cls.exe, result = cls.compile_harness(cls.parts, "baseline")
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def compile_harness(cls, parts, name):
        directory = cls.work / name
        directory.mkdir()
        for part, data in zip(("declarations", "functions", "calls"), parts):
            (directory / ("extracted_invocation_" + part + ".inc")).write_text(data)
        executable = directory / "invocation-regions-flow"
        command = [cls.compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-pedantic",
                   "-I", str(directory), str(ROOT / "probes/m1_invocation_regions_flow_test.cpp"),
                   "-o", str(executable)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        return executable, result

    def control(self, name):
        result = subprocess.run([str(self.exe), name], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, name + ": " + result.stdout + result.stderr)

    def test_extraction_keeps_real_declarations_and_complete_scope(self):
        declarations, functions, _ = self.parts
        self.assertIn(definition(self.source, "struct Invocation {", True), declarations)
        runner = definition(self.source, RUNNER_MARKER).removeprefix("template<> ")
        self.assertIn(runner, functions)
        # Never trim this to setupTmaWarpSpecializedInputs or a handwritten clone.
        self.assertLess(runner.index("auto const regions=getWorkspaceDeviceBufferSizes("),
                        runner.index("Invocation current{call,regions};"))
        self.assertLess(runner.index("Invocation current{call,regions};"),
                        runner.index("struct ClearInvocation"))
        self.assertLess(runner.index("struct ClearInvocation"), runner.rindex("stock();"))
        self.assertTrue(runner.endswith('"installed preparation call sites bypassed bridge");\n}'))
        self.assertIn("const& regions;", declarations)
        self.assertNotRegex(runner, r"(?:static|thread_local)\s+[^;]*regions")

    def test_reference_identity_zero_allocations_and_owning_control(self):
        self.control("identity")

    def test_owning_member_regression_fails_the_compiled_type_control(self):
        declarations, functions, calls = self.parts
        self.assertEqual(declarations.count("const& regions;"), 1)
        changed = declarations.replace("const& regions;", "regions;")
        _, result = self.compile_harness((changed, functions, calls), "owning-regression")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invocation must borrow the exact map", result.stderr)

    def test_missing_cleanup_regression_fails_before_map_storage_release(self):
        declarations, functions, calls = self.parts
        guard = "~ClearInvocation(){invocation=nullptr;}"
        self.assertEqual(functions.count(guard), 1)
        changed = functions.replace(guard, "~ClearInvocation(){}")
        executable, result = self.compile_harness((declarations, changed, calls), "cleanup-regression")
        self.assertEqual(result.returncode, 0, result.stderr)
        run = subprocess.run([str(executable), "normal"], capture_output=True, text=True, timeout=15)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("map storage released before restoring expected invocation", run.stderr)

    def test_normal_scope_clears_before_map_releases_storage(self):
        self.control("normal")

    def test_every_host_exception_unwinds_without_retry_or_dangling_borrow(self):
        self.control("exceptions")

    def test_rejected_reentrant_lease_and_runner_preserve_outer_invocation(self):
        self.control("reentrant")

    def test_fresh_values_with_different_and_reused_map_and_owner_addresses(self):
        self.control("fresh")

    def test_captured_observer_workspace_metadata_parity(self):
        self.control("diagnostics")

    def test_unsupported_calls_delegate_before_map_construction(self):
        self.control("unsupported")


if __name__ == "__main__":
    unittest.main()
