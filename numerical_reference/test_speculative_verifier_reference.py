"""Independent serial-sequence/cache oracles, adversarial controls and exact laws.

Synthetic K/V bytes and logits are witnesses for ownership and row alignment,
not approximations of Gemma and not mock GPU qualification.
"""

from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import itertools
import math
import struct
import unittest

from speculative_verifier_reference import (
    CacheGeometry, CacheSnapshot, Decision, KVRecord, ReferenceCache,
    SpecContractError, VerifyBlock, acceptance_probability, commit_group,
    distribution, greedy_token, residual_distribution, sample_distribution,
    select_greedy, select_stochastic,
)


VOCAB = 9


@lru_cache(maxsize=128)
def weighted_prefix(prefix):
    return tuple(itertools.accumulate((index + 1) * (token + 1)
                                      for index, token in enumerate(prefix)))


def payload(prefix, position, layer):
    """Oracle bytes indexed by full history, separate from cache implementation."""
    weighted = weighted_prefix(tuple(prefix))[position]
    key = struct.pack(">4q", position, layer, prefix[position], weighted)
    value = struct.pack(">4q", -position - 1, layer + 10, prefix[position] + 1, -weighted - 7)
    return key, value


def record(prefix, position, layer):
    key, value = payload(prefix, position, layer)
    return KVRecord(position, prefix[position], position, key, value)


def payload_logits(layer_rows):
    # Every byte contributes to every score; compare full scores as well as
    # emitted IDs, so an accidental argmax collision cannot hide corruption.
    raw = b"".join(row.k + row.v for rows in layer_rows for row in rows)
    return tuple(int.from_bytes(hashlib.sha256(raw + bytes([token])).digest()[:6], "big")
                 for token in range(VOCAB))


def serial_logits(prefix, geometries):
    query = len(prefix) - 1
    rows = []
    for layer, geometry in enumerate(geometries):
        first = 0 if geometry.window is None else max(0, query - geometry.window + 1)
        # Rebuild absolute records directly from the full sequence, without
        # rings, page addresses, transactions or the reference mask method.
        rows.append(tuple(record(prefix, position, layer) for position in range(first, query + 1)))
    return payload_logits(rows)


def serial_tokens(prefix, count, geometries):
    sequence = tuple(prefix)
    emitted = []
    for _ in range(count):
        logits = serial_logits(sequence, geometries)
        best = sorted(range(VOCAB), key=lambda token: (-logits[token], token))[0]
        emitted.append(best)
        sequence += (best,)
    return tuple(emitted)


def one_hot_logits(*tokens, vocabulary=VOCAB):
    return tuple(tuple(1 if index == token else 0 for index in range(vocabulary)) for token in tokens)


class GreedyContractTests(unittest.TestCase):
    def test_absolute_anchor_row_and_prediction_positions(self):
        block = VerifyBlock(1023, 2, (3, 4))
        self.assertEqual(block.inputs, (2, 3, 4))
        self.assertEqual(block.input_positions, (1023, 1024, 1025))
        self.assertEqual(block.prediction_positions, (1024, 1025, 1026))

    def test_every_acceptance_length_including_zero_and_full(self):
        for count in (0, 1, 2, 4, 7):
            correct = tuple((index + 3) % VOCAB for index in range(count + 1))
            for accepted in range(count + 1):
                with self.subTest(proposals=count, accepted=accepted):
                    drafts = list(correct[:count])
                    if accepted < count:
                        drafts[accepted] = (drafts[accepted] + 1) % VOCAB
                    result = select_greedy(VerifyBlock(64, 2, tuple(drafts)), one_hot_logits(*correct))
                    self.assertEqual(result.emitted, correct[:accepted + 1])
                    self.assertEqual(result.accepted_draft_tokens, accepted)
                    self.assertEqual(result.commit_count, accepted + 1)
                    self.assertEqual(result.pending_anchor, correct[accepted])
                    self.assertEqual(result.target_token_kind, "bonus" if accepted == count else "fallback")

    def test_ties_and_processed_masked_logits(self):
        self.assertEqual(greedy_token((-math.inf, 4, 4)), 1)
        result = select_greedy(VerifyBlock(0, 0, (1,)), ((-math.inf, 4, 4), (5, 1, 2)))
        self.assertEqual(result.emitted, (1, 0))

    def test_invalid_rows_and_nonfinite_logits_rejected(self):
        cases = ((), ((1, 2),), ((1, 2), (1,)),
                 ((math.nan, 1), (1, 2)), ((math.inf, 1), (1, 2)),
                 ((-math.inf, -math.inf), (1, 2)))
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(SpecContractError):
                select_greedy(VerifyBlock(0, 0, (1,)), rows)

    def test_accepted_eos_stops_at_each_position_without_bonus(self):
        for eos_index in range(3):
            drafts = list((2, 3, 4))
            drafts[eos_index] = 8
            result = select_greedy(VerifyBlock(17, 1, tuple(drafts)),
                                   one_hot_logits(*drafts, 7), eos_token_ids=(8,))
            self.assertEqual(result.emitted, tuple(drafts[:eos_index + 1]))
            self.assertEqual(result.commit_count, eos_index + 1)
            self.assertEqual(result.accepted_draft_tokens, eos_index + 1)
            self.assertIsNone(result.target_token_kind)
            self.assertEqual(result.stop_reason, "eos")

    def test_rejected_eos_is_not_emitted(self):
        result = select_greedy(VerifyBlock(17, 1, (8, 8)), one_hot_logits(3, 8, 8), eos_token_ids=(8,))
        self.assertEqual(result.emitted, (3,))
        self.assertIsNone(result.stop_reason)

    def test_fallback_bonus_and_pending_anchor_eos(self):
        fallback = select_greedy(VerifyBlock(17, 1, (2,)), one_hot_logits(8, 3), eos_token_ids=(8,))
        bonus = select_greedy(VerifyBlock(17, 1, (2,)), one_hot_logits(2, 8), eos_token_ids=(8,))
        terminal = select_greedy(VerifyBlock(17, 8, (2,)), (), eos_token_ids=(8,))
        self.assertEqual((fallback.emitted, fallback.target_token_kind, fallback.stop_reason), ((8,), "fallback", "eos"))
        self.assertEqual((bonus.emitted, bonus.target_token_kind, bonus.stop_reason), ((2, 8), "bonus", "eos"))
        self.assertEqual((terminal.emitted, terminal.commit_count), ((), 0))

    def test_output_cap_keeps_final_token_pending(self):
        for remaining in range(5):
            result = select_greedy(VerifyBlock(33, 1, (2, 3, 4)), one_hot_logits(2, 3, 4, 5),
                                   max_new_tokens=remaining)
            self.assertEqual(result.emitted, (2, 3, 4, 5)[:remaining])
            self.assertEqual(result.commit_count, remaining)
            self.assertEqual(result.stop_reason, "budget")

    def test_negative_shifted_rows_and_duplicate_anchor_are_detected(self):
        expected = (3, 4, 5)
        shifted = select_greedy(VerifyBlock(64, 2, (3, 4)), one_hot_logits(4, 5, 6))
        with self.assertRaises(AssertionError):
            self.assertEqual(shifted.emitted, expected)
        correct = select_greedy(VerifyBlock(64, 2, (3, 4)), one_hot_logits(3, 4, 5))
        for broken in ((2,) + correct.emitted, correct.emitted[:-1]):
            with self.assertRaises(AssertionError):
                self.assertEqual(broken, expected)

    def test_block_and_limit_validation(self):
        for arguments in ((True, 1, ()), (-1, 1, ()), (0, 1, [2]), (0, -1, ())):
            with self.assertRaises(SpecContractError):
                VerifyBlock(*arguments)
        with self.assertRaises(SpecContractError):
            select_greedy(VerifyBlock(0, 0, ()), ((1,),), max_new_tokens=-1)


class TransactionTests(unittest.TestCase):
    def make_caches(self, prefix, geometries):
        return tuple(ReferenceCache(geometry, tuple(record(prefix, pos, layer) for pos in range(len(prefix))))
                     for layer, geometry in enumerate(geometries))

    def assert_cache_prefix(self, caches, prefix):
        # Independently reconstruct physical storage, including the old bytes
        # in every slot. Checking length/visibility alone cannot prove rollback.
        for layer, cache in enumerate(caches):
            expected = [None] * cache.geometry.capacity
            for position in range(len(prefix)):
                slot = position if cache.geometry.window is None else position % cache.geometry.capacity
                expected[slot] = record(prefix, position, layer)
            self.assertEqual(cache.snapshot.length, len(prefix))
            self.assertEqual(cache.snapshot.slots, tuple(expected))

    def stage_block(self, caches, history, drafts):
        block = VerifyBlock(len(history) - 1, history[-1], tuple(drafts))
        transactions = tuple(cache.begin() for cache in caches)
        proposal_history = tuple(history[:-1]) + block.inputs
        for layer, tx in enumerate(transactions):
            for position in block.input_positions:
                tx.stage(record(proposal_history, position, layer))
        geometries = tuple(cache.geometry for cache in caches)
        rows = []
        for query in block.input_positions:
            actual = payload_logits(tuple(tx.visible(query) for tx in transactions))
            self.assertEqual(actual, serial_logits(proposal_history[:query + 1], geometries))
            rows.append(actual)
        return block, transactions, tuple(rows)

    def run_cycle(self, caches, history, count, accepted):
        geometries = tuple(cache.geometry for cache in caches)
        gold = serial_tokens(history, count + 1, geometries)
        drafts = list(gold[:count])
        if accepted < count:
            drafts[accepted] = (drafts[accepted] + 1) % VOCAB
            # An adversarial suffix is materialized, but cannot survive commit.
            drafts[accepted + 1:] = [8] * (count - accepted - 1)
        before = tuple(cache.snapshot for cache in caches)
        block, transactions, rows = self.stage_block(caches, history, drafts)
        self.assertEqual(tuple(cache.snapshot for cache in caches), before)
        result = select_greedy(block, rows)
        self.assertEqual(result.emitted, gold[:accepted + 1])
        self.assertTrue(commit_group(transactions, [result.commit_count] * len(caches)))
        next_history = tuple(history) + result.emitted
        self.assert_cache_prefix(caches, next_history[:-1])
        # Materialize the fallback/bonus at its absolute position next time;
        # compare full future scores to fresh serial state after rollback.
        _, next_transactions, next_rows = self.stage_block(caches, next_history, ())
        self.assertEqual(next_rows[0], serial_logits(next_history, geometries))
        for tx in next_transactions:
            tx.abort()
        return next_history

    def test_all_acceptance_lengths_at_page_ring_and_window_boundaries(self):
        for length in (7, 8, 9, 63, 64, 65, 1023, 1024, 1025, 2049):
            geometries = (CacheGeometry(length + 32, 64), CacheGeometry(1024, 64, 1024))
            prompt = tuple(index % VOCAB for index in range(length))
            anchor = serial_tokens(prompt, 1, geometries)[0]
            for count in (0, 1, 2, 4, 7):
                for accepted in range(count + 1):
                    with self.subTest(length=length, proposals=count, accepted=accepted):
                        caches = self.make_caches(prompt, geometries)
                        self.run_cycle(caches, prompt + (anchor,), count, accepted)

    def test_repeated_wraps_and_rejections_replay_like_serial_decode(self):
        geometries = (CacheGeometry(160, 8), CacheGeometry(8, 4, 8))
        prompt = tuple(index % VOCAB for index in range(15))
        history = prompt + serial_tokens(prompt, 1, geometries)
        caches = self.make_caches(prompt, geometries)
        for step in range(18):
            count = (0, 1, 2, 4, 7)[step % 5]
            history = self.run_cycle(caches, history, count, step % (count + 1))

    def test_long_staging_block_cannot_destroy_small_committed_ring(self):
        geometry = CacheGeometry(4, 2, 4)
        prefix = tuple(index % VOCAB for index in range(13))
        cache = self.make_caches(prefix, (geometry,))[0]
        before = cache.snapshot
        _, transactions, _ = self.stage_block((cache,), prefix + (1,), (2,) * 12)
        self.assertEqual(cache.snapshot, before)
        self.assertTrue(commit_group(transactions, (1,)))
        self.assert_cache_prefix((cache,), prefix + (1,))

    def test_cancel_before_during_and_after_verification_preserves_all_bytes(self):
        geometries = (CacheGeometry(32, 4), CacheGeometry(8, 4, 8))
        prefix = tuple(index % VOCAB for index in range(8))
        for stage_count in (0, 1, 4):
            caches = self.make_caches(prefix, geometries)
            before = tuple(cache.snapshot for cache in caches)
            transactions = tuple(cache.begin() for cache in caches)
            attempted = prefix + (2, 3, 4, 5)
            for layer, tx in enumerate(transactions):
                for position in range(8, 8 + stage_count):
                    tx.stage(record(attempted, position, layer))
            decision = Decision((3,), 0, "fallback", None)
            published = commit_group(transactions, (decision.commit_count,) * 2, cancelled=True)
            emitted = decision.emitted if published else ()
            self.assertEqual(emitted, ())
            self.assertEqual(tuple(cache.snapshot for cache in caches), before)
            for tx in transactions:
                with self.assertRaises(SpecContractError):
                    tx.prepare_commit(0)

    def test_terminal_eos_and_output_limit_do_not_materialize_final_token(self):
        prefix = (1, 2, 3, 4)
        for remaining, eos in ((None, (7,)), (2, ())):
            caches = self.make_caches(prefix, (CacheGeometry(16, 4), CacheGeometry(4, 2, 4)))
            block, transactions, _ = self.stage_block(caches, prefix + (5,), (6, 7, 8))
            result = select_greedy(block, one_hot_logits(6, 7, 8, 0), eos_token_ids=eos,
                                   max_new_tokens=remaining)
            self.assertEqual(result.emitted, (6, 7))
            commit_group(transactions, (result.commit_count,) * 2)
            self.assert_cache_prefix(caches, prefix + (5, 6))
            self.assertEqual(result.pending_anchor, 7)

    def test_cancel_selected_prefix_then_retry_reproduces_serial_sequence(self):
        geometries = (CacheGeometry(32, 4), CacheGeometry(8, 4, 8))
        prefix = tuple(index % VOCAB for index in range(8))
        caches = self.make_caches(prefix, geometries)
        history = prefix + serial_tokens(prefix, 1, geometries)
        gold = serial_tokens(history, 5, geometries)
        block, transactions, rows = self.stage_block(caches, history, gold[:4])
        decision = select_greedy(block, rows)
        self.assertEqual(decision.emitted, gold)
        before = tuple(cache.snapshot for cache in caches)
        self.assertFalse(commit_group(transactions, (decision.commit_count,) * 2, cancelled=True))
        self.assertEqual(tuple(cache.snapshot for cache in caches), before)
        retried = self.run_cycle(caches, history, 4, 4)
        self.assertEqual(retried, history + gold)

    def test_cancellation_after_publication_cannot_rollback_published_prefix(self):
        prefix = (1, 2, 3, 4)
        caches = self.make_caches(prefix, (CacheGeometry(16, 4), CacheGeometry(4, 2, 4)))
        _, transactions, _ = self.stage_block(caches, prefix + (5,), (6,))
        commit_group(transactions, (2, 2))
        published = tuple(cache.snapshot for cache in caches)
        with self.assertRaises(SpecContractError):
            commit_group(transactions, (0, 0), cancelled=True)
        next_attempt = tuple(cache.begin() for cache in caches)
        self.assertFalse(commit_group(next_attempt, (0, 0), cancelled=True))
        self.assertEqual(tuple(cache.snapshot for cache in caches), published)
        self.assert_cache_prefix(caches, prefix + (5, 6))

    def test_prepare_failure_in_any_layer_publishes_nothing(self):
        prefix = (1, 2, 3, 4)
        caches = self.make_caches(prefix, (CacheGeometry(16, 2), CacheGeometry(5, 2)))
        _, transactions, _ = self.stage_block(caches, prefix + (5,), (6,))
        before = tuple(cache.snapshot for cache in caches)
        with self.assertRaises(SpecContractError):
            commit_group(transactions, (2, 2))
        self.assertEqual(tuple(cache.snapshot for cache in caches), before)
        for tx in transactions:
            tx.abort()

    def test_stale_duplicate_noncontiguous_and_mismatched_layer_commits_rejected(self):
        prefix = (1, 2)
        caches = self.make_caches(prefix, (CacheGeometry(8, 2), CacheGeometry(8, 2)))
        old = caches[0].begin()
        block, transactions, _ = self.stage_block(caches, prefix + (3,), (4,))
        before = tuple(cache.snapshot for cache in caches)
        for group, counts in ((transactions, (1, 2)), ((transactions[0], transactions[0]), (1, 1))):
            with self.assertRaises(SpecContractError):
                commit_group(group, counts)
        self.assertEqual(tuple(cache.snapshot for cache in caches), before)
        commit_group(transactions, (1, 1))
        with self.assertRaises(SpecContractError):
            old.stage(record(prefix + block.inputs, 2, 0))
        current = caches[0].begin()
        with self.assertRaises(SpecContractError):
            current.stage(record(prefix + block.inputs, 2, 0))
        current.abort()
        old.abort()

    def test_zero_commit_and_abort_are_exact_snapshots(self):
        cache = self.make_caches((1, 2, 3, 4), (CacheGeometry(4, 2, 4),))[0]
        before = cache.snapshot
        tx = cache.begin()
        tx.stage(record((1, 2, 3, 4, 5), 4, 0))
        commit_group((tx,), (0,))
        self.assertEqual(cache.snapshot, before)

    def test_page_addresses_and_absolute_rope_tags(self):
        ring = CacheGeometry(1024, 64, 1024)
        self.assertEqual([ring.address(pos) for pos in (63, 64, 1023, 1024, 2049)],
                         [(0, 63), (1, 0), (15, 63), (0, 0), (0, 1)])
        with self.assertRaises(SpecContractError):
            KVRecord(1024, 1, 0, b"K", b"V")
        with self.assertRaises(SpecContractError):
            CacheGeometry(8, 4, 9)

    def test_negative_rejected_suffix_commit_and_bonus_off_by_one_detected(self):
        prefix = (1, 2, 3, 4)
        for wrong_count in (1, 3, 4):
            caches = self.make_caches(prefix, (CacheGeometry(16, 2), CacheGeometry(4, 2, 4)))
            _, transactions, _ = self.stage_block(caches, prefix + (5,), (6, 7, 8))
            # Intended prefix is anchor + one accepted draft (count=2).
            commit_group(transactions, (wrong_count,) * 2)
            with self.assertRaises(AssertionError):
                self.assert_cache_prefix(caches, prefix + (5, 6))

    def test_negative_tag_mask_does_not_restore_overwritten_committed_ring(self):
        prefix = tuple(range(8))
        cache = self.make_caches(prefix, (CacheGeometry(8, 4, 8),))[0]
        before = cache.snapshot
        attempted = prefix + (8, 1, 2, 3)
        destroyed = list(before.slots)
        for position in range(8, 12):
            destroyed[position % 8] = record(attempted, position, 0)
        # Emulate direct speculative writes followed by length-only rollback.
        cache._state = CacheSnapshot(before.length, before.generation, tuple(destroyed))
        with self.assertRaises(AssertionError):
            self.assert_cache_prefix((cache,), prefix)
        masked = [row.position for row in destroyed if row.position < before.length]
        self.assertNotIn(1, masked)  # needed at query 8, physically lost
        retry = cache.begin()
        retry.stage(record(attempted, 8, 0))
        with self.assertRaises(SpecContractError):
            retry.visible(8)
        retry.abort()

    def test_future_suffix_is_invisible_to_earlier_verifier_rows(self):
        prefix = (1, 2, 3, 4)
        cache = self.make_caches(prefix, (CacheGeometry(8, 2, 8),))[0]
        _, transactions, _ = self.stage_block((cache,), prefix + (5,), (6, 7))
        tx = transactions[0]
        visible = tx.visible(4)
        self.assertEqual(tuple(row.position for row in visible), (0, 1, 2, 3, 4))
        poisoned = visible + (record(prefix + (5, 6, 7), 6, 0),)
        with self.assertRaises(AssertionError):
            self.assertEqual(payload_logits((poisoned,)), serial_logits(prefix + (5,), (cache.geometry,)))
        tx.abort()

    def test_negative_rope_payload_uses_ring_slot_despite_correct_absolute_tag(self):
        prefix = tuple(index % VOCAB for index in range(1025))
        cache = self.make_caches(prefix, (CacheGeometry(1024, 64, 1024),))[0]
        original = cache.snapshot.slots[0]
        key = struct.pack(">q", 0) + original.k[8:]
        wrong = KVRecord(1024, original.token_id, 1024, key, original.v)
        slots = list(cache.snapshot.slots)
        slots[0] = wrong
        cache._state = CacheSnapshot(cache.snapshot.length, cache.snapshot.generation, tuple(slots))
        with self.assertRaises(AssertionError):
            self.assert_cache_prefix((cache,), prefix)

    def test_unequal_base_lengths_cannot_be_hidden_by_different_commit_counts(self):
        first = ReferenceCache(CacheGeometry(8, 2), (record((1,), 0, 0),))
        second = ReferenceCache(CacheGeometry(8, 2),
                                tuple(record((1, 2), pos, 1) for pos in range(2)))
        tx1, tx2 = first.begin(), second.begin()
        tx1.stage(record((1, 2, 3), 1, 0))
        tx1.stage(record((1, 2, 3), 2, 0))
        tx2.stage(record((1, 2, 3), 2, 1))
        before = (first.snapshot, second.snapshot)
        with self.assertRaises(SpecContractError):
            commit_group((tx1, tx2), (2, 1))
        self.assertEqual((first.snapshot, second.snapshot), before)


class SamplingTests(unittest.TestCase):
    def test_acceptance_and_residual_exact_mass(self):
        p, q = (F(3, 4), F(1, 4)), (F(1, 4), F(3, 4))
        self.assertEqual(acceptance_probability(p, q, 0), 1)
        self.assertEqual(acceptance_probability(p, q, 1), F(1, 3))
        self.assertEqual(residual_distribution(p, q), (1, 0))
        # Enumerate the output law algebraically, independent of selectors.
        accepted = (F(1, 4), F(1, 4))
        rejected = F(1, 2)
        self.assertEqual(tuple(a + rejected * r for a, r in zip(accepted, residual_distribution(p, q))), p)

    def test_zero_support_equal_laws_and_deterministic_draft(self):
        self.assertEqual(acceptance_probability((1, 0), (0, 1), 1), 0)
        self.assertEqual(residual_distribution((1, 0), (0, 1)), (1, 0))
        with self.assertRaises(SpecContractError):
            acceptance_probability((1, 0), (1, 0), 1)
        with self.assertRaises(SpecContractError):
            residual_distribution((F(1, 2), F(1, 2)), (F(1, 2), F(1, 2)))
        p = (F(1, 4), F(3, 4))
        self.assertEqual(acceptance_probability(p, (1, 0), 0), F(1, 4))
        self.assertEqual(residual_distribution(p, (1, 0)), (0, 1))

    def test_exact_four_token_law_with_multitoken_residual_support(self):
        p, q = (F(1, 2), F(1, 4), F(1, 4), 0), (F(1, 4), F(1, 8), F(3, 8), F(1, 4))
        self.assertEqual(residual_distribution(p, q), (F(2, 3), F(1, 3), 0, 0))
        # Eight equally likely proposal cells represent q exactly. The 3-cell
        # acceptance and 12-cell target grids integrate all rational cutoffs.
        proposals = (0, 0, 1, 2, 2, 2, 3, 3)
        actual = defaultdict(F)
        for proposal, accept_grid, target_grid in itertools.product(proposals, range(3), range(12)):
            result = select_stochastic(VerifyBlock(0, 0, (proposal,)), (p, p), (q,),
                                       (F(2 * accept_grid + 1, 6),), F(2 * target_grid + 1, 24),
                                       max_new_tokens=1)
            actual[result.emitted[0]] += F(1, 288)
        self.assertEqual(tuple(actual[token] for token in range(4)), p)

    def test_categorical_zero_mass_and_boundaries(self):
        p = (0, F(1, 4), 0, F(3, 4))
        self.assertEqual(sample_distribution(p, 0), 1)
        self.assertEqual(sample_distribution(p, F(1, 4)), 3)
        self.assertEqual(sample_distribution(p, F(999, 1000)), 3)

    def test_invalid_probabilities_uniforms_and_row_alignment(self):
        for p in ((), (F(1, 2),), (-1, 2), (0.5, 0.5), (True, 0)):
            with self.assertRaises(SpecContractError):
                distribution(p)
        for uniform in (-1, 1, 0.2):
            with self.assertRaises(SpecContractError):
                sample_distribution((1, 0), uniform)
        with self.assertRaises(SpecContractError):
            select_stochastic(VerifyBlock(0, 0, (1,)), ((1, 0),), ((0, 1),), (0,), 0)

    def test_first_reject_uses_residual_and_full_accept_uses_last_target_row(self):
        block = VerifyBlock(7, 0, (1, 0))
        p = ((F(3, 4), F(1, 4)), (1, 0), (0, 1))
        q = ((F(1, 4), F(3, 4)), (1, 0))
        rejected = select_stochastic(block, p, q, (F(1, 3), 0), 0)
        accepted = select_stochastic(block, p, q, (0, 0), 0)
        self.assertEqual(rejected.emitted, (0,))
        self.assertEqual(accepted.emitted, (1, 0, 1))
        self.assertEqual((rejected.target_token_kind, accepted.target_token_kind), ("fallback", "bonus"))

    def test_stochastic_eos_budget_and_no_proposal_step(self):
        block = VerifyBlock(7, 0, (1, 0))
        p = ((0, 1), (1, 0), (0, 1))
        q = ((0, 1), (1, 0))
        for eos, budget, reason in (((1,), None, "eos"), ((), 1, "budget")):
            result = select_stochastic(block, p, q, (0, 0), 0,
                                       eos_token_ids=eos, max_new_tokens=budget)
            self.assertEqual((result.emitted, result.commit_count, result.stop_reason), ((1,), 1, reason))
        zero = select_stochastic(VerifyBlock(7, 0, ()), ((0, 1),), (), (), 0)
        self.assertEqual((zero.emitted, zero.target_token_kind), ((1,), "bonus"))

    def test_fixed_uniform_replay_is_reproducible(self):
        args = (VerifyBlock(7, 0, (1,)), ((F(3, 4), F(1, 4)), (1, 0)),
                ((F(1, 4), F(3, 4)),), (F(1, 2),), F(1, 4))
        self.assertEqual(select_stochastic(*args), select_stochastic(*args))

    def test_exact_three_token_joint_law_over_multiple_cycles(self):
        # Finite uniform grids exactly integrate this conditional two-token
        # system. A rejection may require another cycle to reach three outputs.
        # Expected law is a separate serial product over target conditionals.
        def target(history):
            return (F(3, 4), F(1, 4)) if history[-1] == 0 else (F(1, 4), F(3, 4))

        @lru_cache(None)
        def speculative_law(history, remaining):
            if remaining == 0:
                return {(): F(1)}
            cycle = defaultdict(F)
            for x0, x1, u0, u1, ut in itertools.product(range(2), range(2), range(2), range(2), range(4)):
                drafts = (x0, x1)  # exact draws under the actual q=(1/2,1/2)
                p = (target(history), target(history + (x0,)), target(history + drafts))
                result = select_stochastic(VerifyBlock(len(history) - 1, history[-1], drafts), p,
                                           ((F(1, 2), F(1, 2)),) * 2,
                                           (F(2 * u0 + 1, 4), F(2 * u1 + 1, 4)), F(2 * ut + 1, 8),
                                           max_new_tokens=remaining)
                cycle[result.emitted] += F(1, 64)
            output = defaultdict(F)
            for emitted, mass in cycle.items():
                for suffix, tail_mass in speculative_law(history + emitted, remaining - len(emitted)).items():
                    output[emitted + suffix] += mass * tail_mass
            return dict(output)

        expected = {}
        for tokens in itertools.product(range(2), repeat=3):
            probability, history = F(1), (0,)
            for token in tokens:
                probability *= target(history)[token]
                history += (token,)
            expected[tokens] = probability
        actual = speculative_law((0,), 3)
        self.assertEqual(sum(actual.values()), 1)
        self.assertEqual(actual, expected)

    def test_negative_independent_token_agreement_is_not_a_distribution_proof(self):
        p, q = (F(3, 4), F(1, 4)), (F(1, 4), F(3, 4))
        # Wrong algorithm: keep draft only when an independent target draw
        # agrees, otherwise resample p without residual correction.
        rejection_mass = 1 - sum(a * b for a, b in zip(p, q))
        wrong_law = tuple(target * draft + rejection_mass * target for target, draft in zip(p, q))
        self.assertEqual(wrong_law, (F(21, 32), F(11, 32)))
        with self.assertRaises(AssertionError):
            self.assertEqual(wrong_law, p)

    def test_negative_length_selected_from_sampled_proposal_can_bias_output(self):
        p, q = (F(3, 4), F(1, 4)), (F(1, 2), F(1, 2))
        wrong = defaultdict(F)
        for sampled_draft, target_grid in itertools.product(range(2), range(4)):
            # Wrong policy: verify x=0, discard x=1 and switch to target-only.
            # It uses unconditional q even though the scheduled trace is selected.
            drafts = (0,) if sampled_draft == 0 else ()
            result = select_stochastic(VerifyBlock(0, 0, drafts), (p,) * (len(drafts) + 1),
                                       (q,) * len(drafts), (F(1, 4),) * len(drafts),
                                       F(2 * target_grid + 1, 8), max_new_tokens=1)
            wrong[result.emitted[0]] += F(1, 8)
        self.assertEqual((wrong[0], wrong[1]), (F(7, 8), F(1, 8)))
        with self.assertRaises(AssertionError):
            self.assertEqual((wrong[0], wrong[1]), p)


if __name__ == "__main__":
    unittest.main()
