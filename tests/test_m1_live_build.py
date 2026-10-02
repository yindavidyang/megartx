"""CPU regression for actual compiled-control subprocess argv."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


class TestBuildControlCommands(unittest.TestCase):
    def test_binding_and_lease_commands_preserve_paths_as_single_arguments(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            from build_m1_live_bridge import control_commands
        work, module = Path("/tmp/owned build"), Path("/tmp/installed module.so")
        lease, binding = control_commands(Path("/tmp/runtime/python"), work, module)
        self.assertEqual(lease, ["/tmp/runtime/python", "-B", "/tmp/owned build/scripts/check_m1_live_bridge.py",
                                 "/tmp/owned build/m1_live_bridge.so", "/tmp/owned build/lease-controls.json"])
        self.assertEqual(binding, ["/tmp/runtime/python", "-B", "/tmp/owned build/scripts/check_m1_live_bindings.py",
                                   "/tmp/owned build/m1_live_bridge.so", "/tmp/installed module.so",
                                   "/tmp/owned build/binding-controls.json"])


if __name__ == "__main__": unittest.main()
