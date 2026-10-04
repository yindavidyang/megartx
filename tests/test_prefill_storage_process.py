"""Real CPU child processes; no model, GPU, runtime imports, or network."""
import base64
from contextlib import contextmanager
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from megartx import prefill_storage_process as client


class Evidence:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def write(self, name, value):
        self.calls.append((name, value))
        if self.error is not None:
            raise self.error


class BoundedClientTests(unittest.TestCase):
    @contextmanager
    def children(self):
        """Observe ownership/reaping without replacing real CPU subprocesses."""
        children = []
        original = subprocess.Popen

        def spawn(*args, **kwargs):
            process = original(*args, **kwargs)
            children.append(process)
            return process

        try:
            with patch.object(client.subprocess, 'Popen', side_effect=spawn) as factory:
                yield children, factory
        finally:
            # Prevent a regression in the helper from leaking test children.
            for process in children:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=2)
                if process.stdout is not None:
                    process.stdout.close()

    def assert_reaped(self, children):
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].returncode)
        with self.assertRaises(ChildProcessError):
            os.waitpid(children[0].pid, os.WNOHANG)

    def receipt(self, evidence, status, size, overflow=False):
        self.assertEqual(len(evidence.calls), 1)
        name, value = evidence.calls[0]
        self.assertEqual(name, 'client-console.json')
        self.assertEqual(value['schema'], client.CONSOLE_SCHEMA)
        self.assertEqual(value['status'], status)
        self.assertEqual(value['output_bytes'], size)
        self.assertIs(value['overflow'], overflow)
        self.assertLessEqual(size, 65536)
        self.assertLessEqual(len(value['console_base64']), 87384)
        self.assertLess(len(json.dumps(value).encode()), 2 << 20)
        for item in value.values():
            self.assertTrue(item is None or isinstance(item, (str, bool, int)))
        data = base64.b64decode(value['console_base64'], validate=True)
        self.assertEqual(len(data), size)
        return value, data

    def run_source(self, source, evidence, timeout=5):
        return client.run_bounded_client(
            [sys.executable, '-S', '-c', source], dict(os.environ),
            time.monotonic() + timeout, evidence,
        )

    def test_exact_cap_succeeds_and_merged_binary_is_lossless(self):
        evidence = Evidence()
        with self.children() as (children, factory):
            result = self.run_source(
                "import os; os.write(1, bytes(range(256))*128); "
                "os.write(2, bytes(range(255,-1,-1))*128)", evidence,
            )
            self.assert_reaped(children)
            self.assertEqual(factory.call_count, 1)
        expected = bytes(range(256)) * 128 + bytes(range(255, -1, -1)) * 128
        self.assertEqual(result.stdout, expected)
        self.assertEqual(result.returncode, 0)
        self.assertIsNone(result.stderr)
        receipt, data = self.receipt(evidence, 'completed', 65536)
        self.assertEqual(data, expected)
        self.assertEqual(receipt['returncode'], 0)

    def test_overflow_stops_running_child_before_later_marker(self):
        evidence = Evidence()
        with tempfile.TemporaryDirectory() as directory, self.children() as (children, factory):
            marker = Path(directory) / 'completed'
            source = ("import os,time,pathlib; os.write(1,b'x'*65537); "
                      "time.sleep(1.5); pathlib.Path(" + repr(str(marker)) + ").touch()")
            start = time.monotonic()
            with self.assertRaisesRegex(ValueError, 'console exceeded'):
                self.run_source(source, evidence)
            self.assertLess(time.monotonic() - start, 1.5)
            self.assertFalse(marker.exists())
            self.assert_reaped(children)
            self.assertLess(children[0].returncode, 0)
            self.assertEqual(factory.call_count, 1)
        _, data = self.receipt(evidence, 'overflow', 65536, overflow=True)
        self.assertEqual(data, b'x' * 65536)

    def test_read_sizes_are_incremental_remaining_plus_one(self):
        evidence = Evidence()
        sizes = []
        original_read = os.read
        original_popen = subprocess.Popen
        with self.children() as (children, factory):
            def bounded_read(fd, size):
                # Popen itself reads its private exec-error pipe; the console
                # invariant concerns only the child's registered stdout pipe.
                is_console = bool(children) and fd == children[0].stdout.fileno()
                if is_console:
                    retained = sum(item[1] for item in sizes)
                    self.assertEqual(size, min(4096, 65536 - retained + 1))
                chunk = original_read(fd, size)
                if is_console:
                    sizes.append((size, len(chunk)))
                return chunk

            with patch.object(client.os, 'read', side_effect=bounded_read), \
                    patch.object(client.subprocess, 'run', side_effect=AssertionError('run forbidden')), \
                    patch.object(original_popen, 'communicate',
                                 side_effect=AssertionError('communicate forbidden')):
                self.run_source("import os; os.write(1,b'x'*65536)", evidence)
            self.assertEqual(factory.call_count, 1)
            self.assert_reaped(children)
        self.assertGreater(len(sizes), 16)
        self.assertEqual(sizes[-1], (1, 0))
        self.receipt(evidence, 'completed', 65536)

    def test_timeout_reaps_child_and_preserves_partial_console(self):
        evidence = Evidence()
        with self.children() as (children, factory):
            with self.assertRaises(TimeoutError):
                self.run_source("import os,time; os.write(1,b'ready'); time.sleep(5)",
                                evidence, timeout=0.2)
            self.assert_reaped(children)
            self.assertEqual(factory.call_count, 1)
        _, data = self.receipt(evidence, 'timeout', 5)
        self.assertEqual(data, b'ready')

    def test_exact_cap_without_exit_times_out_instead_of_succeeding(self):
        evidence = Evidence()
        with self.children() as (children, _):
            with self.assertRaises(TimeoutError):
                self.run_source("import os,time; os.write(1,b'x'*65536); time.sleep(5)",
                                evidence, timeout=0.2)
            self.assert_reaped(children)
        self.receipt(evidence, 'timeout', 65536)

    def test_closed_output_still_waits_for_child_deadline(self):
        evidence = Evidence()
        with self.children() as (children, _):
            with self.assertRaises(TimeoutError):
                self.run_source("import os,time; os.close(1); os.close(2); time.sleep(5)",
                                evidence, timeout=0.2)
            self.assert_reaped(children)
        self.receipt(evidence, 'timeout', 0)

    def test_selector_interrupt_reaps_owned_child(self):
        evidence = Evidence()
        original_selector = selectors.DefaultSelector
        error = KeyboardInterrupt('injected selector interruption')

        def interrupting_selector():
            selector = original_selector()
            original_select = selector.select
            calls = 0

            def select(timeout=None):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise error
                return original_select(timeout)

            selector.select = select
            return selector

        with self.children() as (children, factory), \
                patch.object(client.selectors, 'DefaultSelector', side_effect=interrupting_selector):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_source("import os,time; os.write(1,b'ready'); time.sleep(5)", evidence)
            self.assertIs(caught.exception, error)
            self.assert_reaped(children)
            self.assertEqual(factory.call_count, 1)
        self.receipt(evidence, 'interrupted', 5)

    def test_timeout_kills_term_ignoring_child(self):
        evidence = Evidence()
        with self.children() as (children, _):
            with self.assertRaises(TimeoutError):
                self.run_source(
                    "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                    "os.write(1,b'ready'); time.sleep(5)", evidence, timeout=0.5,
                )
            self.assert_reaped(children)
            self.assertEqual(children[0].returncode, -signal.SIGKILL)
        self.receipt(evidence, 'timeout', 5)

    def test_signal_handler_interrupted_error_reaps_owned_child(self):
        evidence = Evidence()
        error = InterruptedError('injected launcher signal handler')
        selector = selectors.DefaultSelector()
        with self.children() as (children, factory), \
                patch.object(client.selectors, 'DefaultSelector', return_value=selector), \
                patch.object(selector, 'select', side_effect=error):
            with self.assertRaises(InterruptedError) as caught:
                self.run_source("import time; time.sleep(5)", evidence)
            self.assertIs(caught.exception, error)
            self.assert_reaped(children)
            self.assertEqual(factory.call_count, 1)
        self.receipt(evidence, 'interrupted', 0)

    def test_evidence_failure_never_masks_overflow_or_retries(self):
        evidence = Evidence(OSError('injected disk failure'))
        with self.children() as (children, factory):
            with self.assertRaisesRegex(ValueError, 'console exceeded') as caught:
                self.run_source("import os,time; os.write(1,b'x'*65537); time.sleep(5)", evidence)
            self.assert_reaped(children)
            self.assertEqual(factory.call_count, 1)
        self.assertTrue(any('evidence write failed' in note and 'disk failure' in note
                            for note in caught.exception.__notes__))
        self.receipt(evidence, 'overflow', 65536, overflow=True)

    def test_evidence_failure_never_masks_timeout(self):
        evidence = Evidence(OSError('injected disk failure'))
        with self.children() as (children, _):
            with self.assertRaises(TimeoutError) as caught:
                self.run_source("import time; time.sleep(5)", evidence, timeout=0.2)
            self.assert_reaped(children)
        self.assertTrue(any('evidence write failed' in note for note in caught.exception.__notes__))
        self.receipt(evidence, 'timeout', 0)

    def test_evidence_failure_after_success_is_reported_once(self):
        error = OSError('injected disk failure')
        evidence = Evidence(error)
        with self.children() as (children, factory):
            with self.assertRaises(OSError) as caught:
                self.run_source("print('ok')", evidence)
            self.assertIs(caught.exception, error)
            self.assert_reaped(children)
            self.assertEqual(factory.call_count, 1)
        self.receipt(evidence, 'completed', 3)

    def test_slow_evidence_write_cannot_report_late_success(self):
        evidence = Evidence()
        original_write = evidence.write

        def delayed_write(name, value):
            original_write(name, value)
            time.sleep(0.25)

        evidence.write = delayed_write
        with self.children() as (children, factory):
            with self.assertRaises(TimeoutError):
                self.run_source("print('ok')", evidence, timeout=0.2)
            self.assert_reaped(children)
            self.assertEqual(factory.call_count, 1)
        self.receipt(evidence, 'completed', 3)

    def test_selector_setup_failure_still_reaps_and_records_console(self):
        evidence = Evidence(OSError('injected disk failure'))
        error = RuntimeError('injected selector setup failure')
        with self.children() as (children, factory), \
                patch.object(client.selectors, 'DefaultSelector', side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                self.run_source("import time; time.sleep(5)", evidence)
            self.assertIs(caught.exception, error)
            self.assert_reaped(children)
            self.assertEqual(factory.call_count, 1)
        self.assertTrue(any('evidence write failed' in note for note in error.__notes__))
        self.receipt(evidence, 'error', 0)

    def test_cleanup_failure_is_not_primary_and_kill_still_reaps(self):
        evidence = Evidence()
        original = subprocess.Popen
        children = []

        def spawn(*args, **kwargs):
            process = original(*args, **kwargs)
            children.append(process)
            process.terminate = lambda: (_ for _ in ()).throw(OSError('injected TERM failure'))
            return process

        with patch.object(client.subprocess, 'Popen', side_effect=spawn):
            try:
                with self.assertRaises(TimeoutError) as caught:
                    self.run_source("import time; time.sleep(5)", evidence, timeout=0.2)
                self.assert_reaped(children)
                self.assertEqual(children[0].returncode, -signal.SIGKILL)
            finally:
                for process in children:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=2)
        self.assertTrue(any('terminate failed' in note and 'TERM failure' in note
                            for note in caught.exception.__notes__))
        self.receipt(evidence, 'timeout', 0)

    def test_nonzero_exit_is_returned_with_one_receipt(self):
        evidence = Evidence()
        with self.children() as (children, factory):
            result = self.run_source("import os; os.write(2,b'failure'); raise SystemExit(17)", evidence)
            self.assert_reaped(children)
            self.assertEqual(factory.call_count, 1)
        self.assertEqual(result.returncode, 17)
        self.assertEqual(result.stdout, b'failure')
        receipt, _ = self.receipt(evidence, 'completed', 7)
        self.assertEqual(receipt['returncode'], 17)

    def test_empty_success_and_passed_environment(self):
        evidence = Evidence()
        command = [sys.executable, '-S', '-c', "import os; assert os.environ['ONLY_VALUE']=='set'"]
        result = client.run_bounded_client(command, {'ONLY_VALUE': 'set'},
                                           time.monotonic() + 5, evidence)
        self.assertEqual(result.args, command)
        self.assertEqual(result.stdout, b'')
        self.assertEqual(result.returncode, 0)
        self.receipt(evidence, 'completed', 0)

    def test_expired_deadline_does_not_launch_and_writes_one_receipt(self):
        evidence = Evidence()
        with patch.object(client.subprocess, 'Popen') as factory:
            with self.assertRaises(TimeoutError):
                client.run_bounded_client(['never'], {}, time.monotonic() - 1, evidence)
            factory.assert_not_called()
        receipt, _ = self.receipt(evidence, 'timeout', 0)
        self.assertIsNone(receipt['returncode'])

    def test_invalid_deadlines_fail_before_launch(self):
        for deadline in (float('inf'), float('nan'), True, 'later'):
            with self.subTest(deadline=deadline), patch.object(client.subprocess, 'Popen') as factory:
                evidence = Evidence()
                with self.assertRaises(ValueError):
                    client.run_bounded_client(['never'], {}, deadline, evidence)
                factory.assert_not_called()
                self.receipt(evidence, 'error', 0)

    def test_launch_failure_is_preserved_and_not_retried(self):
        error = OSError('injected exec failure')
        evidence = Evidence(OSError('injected disk failure'))
        with patch.object(client.subprocess, 'Popen', side_effect=error) as factory:
            with self.assertRaises(OSError) as caught:
                client.run_bounded_client(['never'], {}, time.monotonic() + 5, evidence)
            self.assertIs(caught.exception, error)
            self.assertEqual(factory.call_count, 1)
        self.assertTrue(any('evidence write failed' in note for note in error.__notes__))
        self.receipt(evidence, 'error', 0)

    def test_import_is_cpu_only(self):
        evidence = Evidence()
        result = self.run_source(
            "import megartx.prefill_storage_process,sys; assert not any("
            "name.split('.')[0] in {'torch','vllm','numpy','requests'} for name in sys.modules)",
            evidence,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.receipt(evidence, 'completed', 0)


if __name__ == '__main__':
    unittest.main()
