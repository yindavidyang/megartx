"""CPU fail-closed private artifact/loader admission; no installed changes."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import m1_private_aot as aot


class AotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.fi = self.root / "installed/flashinfer"; self.cache = self.root / "runtime/cache"
        self.fi.mkdir(parents=True); (self.cache / "fused_moe_120").mkdir(parents=True)
        (self.fi / "core.py").write_text("loader")
        (self.cache / "fused_moe_120/fused_moe_120.so").write_text("module")
        self.loaders = {"core.py": aot.sha(self.fi / "core.py")}
        self.modules = {"fused_moe_120": aot.sha(self.cache / "fused_moe_120/fused_moe_120.so")}
        self.output = self.root / "private-aot"
        self.patches = [patch.object(aot, "LOADER_PINS", self.loaders), patch.object(aot, "MODULE_PINS", self.modules)]
        for p in self.patches: p.start(); self.addCleanup(p.stop)
        aot.prepare(self.cache, self.fi, self.output, "a" * 40)

    def test_private_link_keeps_exact_incumbent_inode_and_source(self):
        manifest = aot.validate_cache(self.output, "a" * 40)
        link = self.output / "aot/fused_moe_120/fused_moe_120.so"
        self.assertEqual(link.resolve(), self.cache / "fused_moe_120/fused_moe_120.so")
        self.assertEqual(manifest["helper_sha256"], aot.sha(aot.__file__))
        self.assertFalse(os.environ.get("FLASHINFER_DISABLE_JIT"))

    def test_changed_module_loader_shim_and_source_fail_closed(self):
        paths = [self.fi / "core.py", self.cache / "fused_moe_120/fused_moe_120.so",
                 self.output / "python/sitecustomize.py"]
        for path in paths:
            old = path.read_bytes(); path.write_bytes(b"changed")
            with self.subTest(path=path), self.assertRaises(RuntimeError): aot.validate_cache(self.output)
            path.write_bytes(old)
        with self.assertRaisesRegex(RuntimeError, "source"):
            aot.validate_cache(self.output, "b" * 40)

    def test_external_module_symlink_cannot_promote_installed_cache(self):
        module = self.cache / "fused_moe_120/fused_moe_120.so"
        module.unlink(); module.symlink_to(self.fi / "core.py")
        with self.assertRaisesRegex(RuntimeError, "module identity"):
            aot.validate_cache(self.output)

    def test_corrupt_bootstrap_exits_78_before_any_model(self):
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text()); manifest["helper_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        env = os.environ.copy(); env["MEGARTX_M1_PRIVATE_AOT"] = str(self.output)
        env["PYTHONPATH"] = str(self.output / "python"); env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run([sys.executable, "-c", "raise AssertionError('model dispatch')"],
                                env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 78)
        self.assertIn("Private AOT startup rejected", result.stderr)
        self.assertNotIn("model dispatch", result.stderr)

    def test_incumbent_build_guard_preserves_other_spec_build_behavior(self):
        class Spec:
            def __init__(self, name): self.name = name
            def build(self, *args, **kwargs): return "ordinary build"
        core = types.ModuleType("flashinfer.jit.core"); core.JitSpecNvcc = Spec
        core.__file__ = str(self.fi / "jit/core.py")
        core.jit_env = types.SimpleNamespace(FLASHINFER_AOT_DIR=self.output / "aot", FLASHINFER_JIT_DIR=self.cache)
        fi = types.ModuleType("flashinfer"); jit = types.ModuleType("flashinfer.jit")
        fi.jit = jit; jit.core = core
        with patch.dict(sys.modules, {"flashinfer":fi,"flashinfer.jit":jit,"flashinfer.jit.core":core}), \
             patch.dict(os.environ, {"MEGARTX_M1_PRIVATE_AOT":str(self.output)}), \
             patch.object(aot, "validate_cache", return_value={"flashinfer_root":str(self.fi),"cache":str(self.cache)}):
            aot.install_guard()
            with self.assertRaisesRegex(RuntimeError, "rebuild forbidden"): Spec("fused_moe_120").build()
            self.assertEqual(Spec("other").build(), "ordinary build")


if __name__ == "__main__":
    unittest.main()
