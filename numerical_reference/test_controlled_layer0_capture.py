"""CPU guards for the bounded observer; these are not live CUDA evidence."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from megartx.controlled_layer0_capture import ControlledLayer0, block_table_slots, extend_prefix_slots, selected_rows


class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.tokens = tuple(range(100, 133))

    def test_full_and_cached_keep_absolute_rows(self):
        self.assertEqual(selected_rows(list(range(33)), self.tokens, self.tokens), [31, 32])
        self.assertEqual(selected_rows(list(range(32)), self.tokens[:32], self.tokens), [31])
        self.assertEqual(selected_rows([32], self.tokens[32:], self.tokens), [0])

    def test_reordered_context_does_not_assume_row_number_equals_position(self):
        self.assertEqual(selected_rows([32, 31], [132, 131], self.tokens), [0, 1])

    def test_wrong_token_and_duplicate_positions_fail(self):
        for positions, tokens in (([32], [131]), ([32, 32], [132, 132]), ([True], [101]), ([-1], [132])):
            with self.subTest(positions=positions), self.assertRaises(RuntimeError):
                selected_rows(positions, tokens, self.tokens)

    def test_missing_boundary_and_unbounded_context_fail(self):
        for positions, tokens in (([0], [100]), ([], []), (list(range(34)), list(range(100, 134)))):
            with self.subTest(rows=len(positions)), self.assertRaises(RuntimeError):
                selected_rows(positions, tokens, self.tokens)

    def test_block_address_calculation_crosses_real_token_axis_boundary(self):
        self.assertEqual(block_table_slots([31, 32], [4, 17, 9], 16), [287, 144])
        self.assertEqual(block_table_slots([32, 31], [4, 17, 9], 16), [144, 287])

    def test_block_addresses_reject_invalid_geometry_and_uncovered_positions(self):
        for positions, table, size in (([32], [1, 2], 16), ([31], [1, -2], 16), ([31], [1, 2], True), ([31], [1, 2], 0), ([True], [1], 16)):
            with self.subTest(positions=positions, size=size), self.assertRaises(RuntimeError):
                block_table_slots(positions, table, size)

    def test_prefix_copy_binds_prefill_and_decode_without_aliasing(self):
        slots = extend_prefix_slots({}, list(range(32)), list(range(100, 132)), 160)
        result = extend_prefix_slots(slots, [32], [7], 160)
        self.assertEqual(result[32], 7)
        self.assertEqual(len(slots), 32)
        self.assertEqual(result[31], 131)

    def test_prefix_copy_rejects_gaps_aliases_reuse_and_invalid_addresses(self):
        for positions, slots in (([1], [0]), ([0, 1], [4, 4]), ([0, 0], [4, 5]), ([0], [-1]), ([0], [160]), ([True], [1])):
            with self.subTest(positions=positions), self.assertRaises(RuntimeError):
                extend_prefix_slots({}, positions, slots, 160)
        with self.assertRaises(RuntimeError):
            extend_prefix_slots({0: 4}, [0], [4], 160)


class ScopeTests(unittest.TestCase):
    def test_deferred_downstream_retains_original_objects_until_decoder_post(self):
        observer = ControlledLayer0.__new__(ControlledLayer0)
        observer.owner = SimpleNamespace(active=True, case="cached")
        observer.in_layer0, observer.capture_policy, observer.flushing = True, "deferred_downstream", False
        observer.frame = {"arrays": {}, "source_rows": 1}
        observer.indices, observer.pending_tensors = [0], {}
        observer.torch = SimpleNamespace(bfloat16="bf16")
        tensor = SimpleNamespace(dtype="bf16", device=SimpleNamespace(type="cuda"), ndim=2, shape=(1, 4096))
        observer._record("attention_out", tensor)
        self.assertIs(observer.pending_tensors["attention_out"], tensor)
        self.assertEqual(observer.frame["arrays"], {})
        with self.assertRaisesRegex(RuntimeError, "repeated"):
            observer._record("attention_out", tensor)
        calls = []
        observer._record = lambda name, value: calls.append((name, value))
        observer.prefix_pending = ("cache", "key", "value", "slots")
        observer._observe_prefix = lambda *args: calls.append(("prefix", args))
        observer._decoder_post(None, (), None)
        self.assertIs(calls[0][1], tensor)
        self.assertEqual(calls[1], ("prefix", ("cache", "key", "value", "slots")))
        self.assertFalse(observer.in_layer0)
        self.assertFalse(observer.flushing)
        self.assertEqual(observer.pending_tensors, {})
        self.assertIsNone(observer.prefix_pending)

    def test_shared_rope_calls_outside_actual_layer0_are_excluded(self):
        observer = ControlledLayer0.__new__(ControlledLayer0)
        observer.owner = SimpleNamespace(active=True, case="full")
        observer.capture_policy = "synchronous"
        observer.in_layer0, observer.frame = False, {}
        self.assertFalse(observer._current())
        observer._decoder_pre(None, ())
        self.assertTrue(observer._current())
        with self.assertRaisesRegex(RuntimeError, "nested or repeated"):
            observer._decoder_pre(None, ())
        observer._decoder_post(None, (), None)
        self.assertFalse(observer._current())
        # A shared RoPE invocation from a subsequent layer must not even
        # inspect its tensors, much less mutate or capture them.
        observer._rope_pre(None, ())
        observer._rope_post(None, (), None)

    def test_lifecycle_rejects_layer0_flag_outside_native_full_cached(self):
        script = Path(__file__).resolve().parents[1] / "scripts/run_scale_validation.py"
        for args in (("--mode", "native"),
                     ("--mode", "paired_reference", "--client", "controlled", "--controlled-path", "full"),
                     ("--mode", "native", "--client", "controlled", "--controlled-path", "chunked"),
                     ("--mode", "native", "--client", "benchmark")):
            # requests is a live-client dependency, deliberately absent from
            # CPU CI. These parser-only refusals must not depend on that app.
            code = "import runpy,sys,types; sys.modules['requests']=types.ModuleType('requests'); sys.argv=" + repr([str(script), "--label", "unlaunched", "--layer0-boundaries", *args]) + "; runpy.run_path(" + repr(str(script)) + ",run_name='__main__')"
            result = subprocess.run([sys.executable, "-c", code],
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Layer-0 boundaries require only", result.stderr)


if __name__ == "__main__":
    unittest.main()
