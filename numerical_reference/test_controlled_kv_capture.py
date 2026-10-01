"""CPU-only writer-slot ledger and stride-aware gather stubs, never live proof."""
import copy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from types import SimpleNamespace
import unittest

import numpy as np

from megartx.controlled_kv_capture import SlotLedger, gather_rows, nominal_layers, validate_context, validate_metadata


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tokens = tuple(range(100, 133))
        self.ledger = SlotLedger(self.tokens)

    def frame(self, positions):
        positions = tuple(positions)
        return (positions, tuple(self.tokens[p] for p in positions),
                {i: [i * 100 + p for p in positions] for i in range(30)},
                {i: 4000 for i in range(30)}, {i: (i, "stable storage") for i in range(30)})

    def test_full_chunked_and_handoff_bind_identical_absolute_slots(self):
        for parts in ((range(33),), (range(16), range(16, 32), range(32, 33)), (range(32), range(32, 33))):
            ledger = SlotLedger(self.tokens)
            for part in parts:
                ledger.record(*self.frame(part))
            self.assertTrue(ledger.complete)
            self.assertEqual([ledger.selected(i) for i in range(30)], [(i * 100 + 31, i * 100 + 32) for i in range(30)])

    def test_reordered_rows_use_positions_and_not_row_numbers(self):
        self.ledger.record(*self.frame(reversed(range(33))))
        self.assertEqual(self.ledger.selected(0), (31, 32))

    def test_incomplete_prefill_cannot_claim_committed_handoff(self):
        self.ledger.record(*self.frame(range(32)))
        with self.assertRaisesRegex(RuntimeError, "committed length"):
            self.ledger.selected(0)

    def test_wrong_token_rejects_entire_frame_without_partial_layer_commit(self):
        frame = list(self.frame(range(33)))
        frame[1] = tuple([999] + list(frame[1][1:]))
        with self.assertRaisesRegex(RuntimeError, "input tokens"):
            self.ledger.record(*frame)
        self.assertEqual(self.ledger.seen, set())
        self.assertTrue(all(not rows for rows in self.ledger.slots.values()))

    def test_duplicate_skipped_or_stale_absolute_positions_are_rejected(self):
        for positions in ((0, 0), (1,), (0, 2)):
            with self.subTest(positions=positions), self.assertRaisesRegex(RuntimeError, "positions"):
                self.ledger.record(*self.frame(positions))
        self.ledger.record(*self.frame(range(16)))
        with self.assertRaisesRegex(RuntimeError, "positions"):
            self.ledger.record(*self.frame(range(16)))

    def test_invalid_tokens_prefix_or_empty_rows_are_rejected(self):
        for tokens in (self.tokens[:-1], self.tokens + (200,), (True,) + self.tokens[1:]):
            with self.subTest(tokens=len(tokens)), self.assertRaisesRegex(RuntimeError, "33 exact"):
                SlotLedger(tokens)
        with self.assertRaisesRegex(RuntimeError, "bounded minimal"):
            self.ledger.record(*self.frame(()))

    def test_all_layers_are_required_before_commit(self):
        for index in (2, 3, 4):
            frame = list(self.frame(range(33)))
            frame[index].pop(29)
            with self.subTest(index=index), self.assertRaisesRegex(RuntimeError, "exact layer"):
                self.ledger.record(*frame)
        self.assertFalse(self.ledger.seen)

    def test_negative_outside_float_or_bool_writer_slots_are_rejected(self):
        for value in (-1, 4000, 0.0, True):
            frame = list(self.frame(range(33)))
            frame[2][12][0] = value
            with self.subTest(slot=value), self.assertRaisesRegex(RuntimeError, "writer slot"):
                self.ledger.record(*frame)
        self.assertFalse(self.ledger.seen)

    def test_duplicate_slots_and_later_reuse_are_rejected(self):
        frame = list(self.frame(range(16)))
        frame[2][0][1] = frame[2][0][0]
        with self.assertRaisesRegex(RuntimeError, "reused"):
            self.ledger.record(*frame)
        self.ledger.record(*self.frame(range(16)))
        frame = list(self.frame(range(16, 33)))
        frame[2][0][0] = 0
        with self.assertRaisesRegex(RuntimeError, "reused"):
            self.ledger.record(*frame)
        self.assertEqual(len(self.ledger.seen), 16)

    def test_cache_storage_owner_change_is_rejected_before_commit(self):
        self.ledger.record(*self.frame(range(32)))
        frame = list(self.frame((32,)))
        frame[4][3] = (3, "new allocation")
        with self.assertRaisesRegex(RuntimeError, "owner changed"):
            self.ledger.record(*frame)
        self.assertEqual(len(self.ledger.seen), 32)

    def test_forward_count_is_bounded_independently(self):
        for start in (0, 8, 16, 24):
            self.ledger.record(*self.frame(range(start, start + 8)))
        with self.assertRaisesRegex(RuntimeError, "bounded minimal"):
            self.ledger.record(*self.frame((32,)))


class MetadataTests(unittest.TestCase):
    def meta(self, rows=32, decode=False):
        return SimpleNamespace(num_actual_tokens=rows, num_decodes=int(decode), num_prefills=int(not decode),
                               num_decode_tokens=rows if decode else 0, num_prefill_tokens=0 if decode else rows,
                               causal=True, use_cascade=False)

    def test_single_prefill_and_nonspec_decode(self):
        validate_metadata(self.meta(), 32)
        validate_metadata(self.meta(1, True), 1)

    def test_padded_noncausal_cascade_multi_request_or_ambiguous_counts(self):
        for key, value in (("num_actual_tokens", 33), ("causal", False), ("use_cascade", True),
                           ("num_prefills", 2), ("num_actual_tokens", True), ("num_prefill_tokens", 0)):
            meta = self.meta()
            setattr(meta, key, value)
            with self.subTest(field=key), self.assertRaisesRegex(RuntimeError, "causal unpadded"):
                validate_metadata(meta, 32)

    def test_multi_token_decode_cannot_disguise_speculation(self):
        with self.assertRaisesRegex(RuntimeError, "single nonspec"):
            validate_metadata(self.meta(2, True), 2)

    def context(self):
        return SimpleNamespace(slot_mapping={}, attn_metadata={}, no_compile_layers={}, ubatch_slices=None,
                               cudagraph_runtime_mode=SimpleNamespace(name="NONE"))

    def test_eager_dictionary_context_and_reject_lists_graphs_or_microbatches(self):
        validate_context(self.context())
        for key, value in (("slot_mapping", []), ("attn_metadata", []), ("no_compile_layers", []),
                           ("ubatch_slices", [1]), ("cudagraph_runtime_mode", SimpleNamespace(name="FULL"))):
            context = self.context()
            setattr(context, key, value)
            with self.subTest(field=key), self.assertRaisesRegex(RuntimeError, "eager, unsplit"):
                validate_context(context)

    def test_pinned_nominal_geometry_is_separate_k_and_v(self):
        layers = nominal_layers()
        self.assertEqual(len(layers), 30)
        self.assertEqual((layers[0]["kv_heads"], layers[0]["head_dim"], layers[0]["window_size"]), (8, 256, 1024))
        self.assertEqual((layers[5]["kv_heads"], layers[5]["head_dim"], layers[5]["window_size"]), (2, 512, None))


class TinyTensor:
    def __init__(self, bits, whole_cache=False):
        self.bits, self.dtype, self.whole_cache = bits, "bf16", whole_cache

    @property
    def shape(self):
        return self.bits.shape

    def stride(self):
        return tuple(i // self.bits.itemsize for i in self.bits.strides)

    def __getitem__(self, index):
        return TinyTensor(self.bits[index])

    def detach(self):
        if self.whole_cache:
            raise AssertionError("Whole-cache detach/copy is forbidden")
        return self

    def contiguous(self):
        if self.whole_cache:
            raise AssertionError("Whole-cache contiguous is forbidden")
        return TinyTensor(np.ascontiguousarray(self.bits))

    def view(self, dtype):
        return TinyTensor(self.bits.view(dtype))

    def cpu(self):
        if self.whole_cache:
            raise AssertionError("Whole-cache CPU transfer is forbidden")
        return self

    def numpy(self):
        return self.bits


class TorchStub:
    bfloat16, int16 = "bf16", np.int16

    @staticmethod
    def stack(rows):
        return TinyTensor(np.stack([r.bits for r in rows]))


class GatherTests(unittest.TestCase):
    def setUp(self):
        # Physical NHC storage, logical BHNC view; rows31/32 cross blocks.
        physical = (np.arange(3 * 16 * 2 * 8, dtype=np.uint16).reshape(3, 16, 2, 8) + 0x3000)
        self.raw = physical.transpose(0, 2, 1, 3)
        self.cache = TinyTensor(self.raw, whole_cache=True)

    def test_kernel_slot_decomposition_noncontiguous_strides_and_k_v_slices(self):
        before = self.raw.copy()
        keys, values = gather_rows(self.cache, (31, 32), 4, TorchStub)
        np.testing.assert_array_equal(keys, np.stack([self.raw[1, :, 15, :4], self.raw[2, :, 0, :4]]))
        np.testing.assert_array_equal(values, np.stack([self.raw[1, :, 15, 4:], self.raw[2, :, 0, 4:]]))
        np.testing.assert_array_equal(self.raw, before)
        self.assertFalse(np.array_equal(keys, values))

    def test_raw_signed_zero_is_preserved(self):
        self.raw[1, 0, 15, 0] = 0x8000
        keys, _ = gather_rows(self.cache, (31, 32), 4, TorchStub)
        self.assertEqual(int(keys[0, 0, 0]), 0x8000)

    def test_nan_and_infinity_are_rejected_by_raw_bf16_bits(self):
        for value in (0x7F80, 0xFF80, 0x7FC1):
            self.raw[1, 0, 15, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "nonfinite"):
                gather_rows(self.cache, (31, 32), 4, TorchStub)

    def test_wrong_dtype_content_extent_or_content_stride_is_rejected(self):
        for cache in (TinyTensor(self.raw[:, :, :, :6], True), TinyTensor(self.raw[:, :, :, ::2], True)):
            with self.assertRaisesRegex(RuntimeError, "content layout"):
                gather_rows(cache, (31, 32), 4, TorchStub)
        self.cache.dtype = "fp8"
        with self.assertRaisesRegex(RuntimeError, "content layout"):
            gather_rows(self.cache, (31, 32), 4, TorchStub)

    def test_duplicate_negative_outside_or_fractional_slots_are_rejected(self):
        for slots in ((31, 31), (-1, 32), (31, 48), (31.0, 32), (True, 32)):
            with self.subTest(slots=slots), self.assertRaisesRegex(RuntimeError, "distinct populated"):
                gather_rows(self.cache, slots, 4, TorchStub)

    def test_bounded_snapshot_is_a_copy_not_an_alias(self):
        keys, values = gather_rows(self.cache, (31, 32), 4, TorchStub)
        before = self.raw.copy()
        keys[:] = 0
        values[:] = 0
        np.testing.assert_array_equal(self.raw, before)


if __name__ == "__main__":
    unittest.main()
