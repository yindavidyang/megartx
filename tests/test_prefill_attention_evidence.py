"""CPU-only streaming, shared-budget and incomplete-capture controls."""
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from megartx import prefill_attention_plan as p
from megartx.prefill_attention_evidence import (
    StreamingEvidence, RAW_FILES, STATE_FILE, STATE_BYTES, _geometry,
)
from megartx.prefill_storage_plan import FAILURE_FILE, FAILURE_RESERVE_BYTES


def contribution_process(directory, offset, queue):
    try:
        evidence = StreamingEvidence(directory)
        evidence.contribute('attention-layer-00-k.bf16', offset, bytes(512))
        queue.put('ok')
    except Exception as error:
        queue.put(type(error).__name__)


class StreamingEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.evidence = StreamingEvidence(self.directory)
        self.evidence.create_capture('a' * 64)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_preallocated_extents_and_single_accounting(self):
        self.assertEqual(sum(RAW_FILES.values()), 5395952)
        self.assertEqual(len(RAW_FILES), 8)
        self.assertEqual({x.name: x.stat().st_size for x in self.directory.glob('*.bf16')}, RAW_FILES)
        self.assertEqual(self.evidence.sizes(), {'total_bytes': 5395952 + STATE_BYTES + 2,
                                               'metadata_bytes': STATE_BYTES + 2})
        self.assertEqual(sum(_geometry(name)[0] for name in RAW_FILES), 8316)

    def test_writer_reuses_selected_cpu_bytes_and_reader_requires_completion(self):
        with self.assertRaisesRegex(ValueError, 'Unwritten'):
            self.evidence.read_selection(0, 15)
        key = b'\x80\x3f' * (8 * 256)
        value = b'\x00\x40' * (8 * 256)
        self.evidence.writer_row(0, 15, key, value)
        self.assertEqual(self.evidence.read_selection(0, 15), (key[:512], value[:16]))
        with self.assertRaisesRegex(ValueError, 'zero-filled holes'):
            self.evidence.seal_raw()
        self.assertTrue((self.directory / FAILURE_FILE).exists())

    def test_duplicates_are_sticky_failure(self):
        name = 'attention-layer-05-q.bf16'
        self.evidence.contribute(name, 0, bytes(1024))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.evidence.contribute(name, 0, bytes(1024))
        with self.assertRaisesRegex(ValueError, 'Invalidated'):
            self.evidence.contribute(name, 1024, bytes(1024))

    def test_short_write_cannot_credit_completion(self):
        with patch('megartx.prefill_attention_evidence.os.pwrite', return_value=2):
            with self.assertRaisesRegex(OSError, 'Short raw'):
                self.evidence.contribute('attention-layer-00-k.bf16', 0, bytes(512))
        state = json.loads((self.directory / STATE_FILE).read_bytes())
        self.assertIsNotNone(state['pending'])
        self.assertFalse(int(state['files']['attention-layer-00-k.bf16']['completed'], 16))
        with self.assertRaisesRegex(ValueError, 'Invalidated'):
            self.evidence.seal_raw()

    def test_replaced_same_extent_is_rejected(self):
        name = 'attention-layer-00-k.bf16'
        old = self.directory / name
        replacement = self.directory / 'replacement.json'
        replacement.write_bytes(bytes(RAW_FILES[name]))
        os.replace(replacement, old)
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.evidence.contribute(name, 0, bytes(512))

    def test_unknown_old_raw_files_are_never_a_second_domain(self):
        (self.directory / 'head-position-2047.bf16').write_bytes(bytes(2))
        with self.assertRaisesRegex(ValueError, 'Previous-purpose'):
            self.evidence.sizes()
        with self.assertRaisesRegex(ValueError, 'tracked streaming'):
            self.evidence.raw('attention-layer-00-k.bf16', bytes(RAW_FILES['attention-layer-00-k.bf16']))

    def test_pending_state_after_interruption_cannot_be_sealed(self):
        state = json.loads((self.directory / STATE_FILE).read_bytes())
        state['pending'] = ['attention-layer-00-k.bf16', 0, 512]
        raw = json.dumps(state).encode()
        (self.directory / STATE_FILE).write_bytes(raw + b' ' * (STATE_BYTES - len(raw)))
        with self.assertRaisesRegex(ValueError, 'interrupted'):
            self.evidence.seal_raw()

    def test_metadata_budget_holds_failure_reserve_and_counts_temporaries(self):
        small = StreamingEvidence(self.directory, metadata_limit=STATE_BYTES + FAILURE_RESERVE_BYTES + 16)
        with self.assertRaisesRegex(ValueError, 'overflow'):
            small.write('test.json', {'message': 'x' * 32})
        small.write(FAILURE_FILE, {'status': 'failed'})
        self.assertLessEqual(small.sizes()['metadata_bytes'], small.metadata_limit)

    def test_attention_failure_blocks_seal(self):
        self.evidence.write('attention-failure.json', {'status': 'failed'})
        with self.assertRaisesRegex(ValueError, 'Invalidated'):
            self.evidence.seal_raw()

    def test_cross_process_contributions_share_one_completion_map(self):
        context = multiprocessing.get_context('fork')
        queue = context.Queue()
        children = [context.Process(target=contribution_process, args=(str(self.directory), x * 512, queue))
                    for x in (0, 1)]
        for child in children:
            child.start()
        for child in children:
            child.join(10)
            self.assertEqual(child.exitcode, 0)
        self.assertEqual([queue.get(timeout=2) for _ in children], ['ok', 'ok'])
        state = json.loads((self.directory / STATE_FILE).read_bytes())
        self.assertEqual(bytes.fromhex(state['files']['attention-layer-00-k.bf16']['completed'])[0], 3)
        queue.close()

    def test_full_stream_then_manifest_is_complete(self):
        # Only fsync is elided in this synthetic throughput fixture. The exact
        # contribution path, file writes, lock, bitmap and hashes remain real.
        with patch('megartx.prefill_attention_evidence.os.fsync'):
            for name, size in RAW_FILES.items():
                count, width = _geometry(name)
                raw = bytes(width)
                for row in range(count):
                    self.evidence.contribute(name, row * width, raw)
        manifest = self.evidence.seal_raw()
        self.assertEqual(manifest, {name: {'bytes': size, 'sha256': hashlib.sha256(bytes(size)).hexdigest()}
                                    for name, size in RAW_FILES.items()})
        self.assertEqual(self.evidence.seal_raw(), manifest)
        self.assertEqual(self.evidence.read_selection(5, 2048), (bytes(2048), bytes(32)))


if __name__ == '__main__':
    unittest.main()
