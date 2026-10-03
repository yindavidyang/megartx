"""CPU interface/transaction tests against PR18 and an independent serial target.

All byte/logit witnesses are synthetic; these are not GPU/model qualification.
"""

from collections import defaultdict
from dataclasses import replace
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import importlib.util
import itertools
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from megartx import speculative_verifier as v


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("speculative_cpu_oracle", ROOT / "numerical_reference/speculative_verifier_reference.py")
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)
VOCAB = 9
GEOMETRIES = ((8192, 16, None), (8, 3, 8))


@lru_cache(maxsize=256)
def weighted_prefix(history):
    return tuple(itertools.accumulate((p + 1) * (t + 1) for p, t in enumerate(history)))


def record(history, position, layer):
    material = repr((weighted_prefix(history)[position], history[position], position, layer)).encode()
    return v.KVRow(position, history[position], position,
                   hashlib.sha256(b"K" + material).digest(), hashlib.sha256(b"V" + material).digest())


def scores(rows):
    material = b"".join(row.k + row.v for layer in rows for row in layer)
    return tuple(int.from_bytes(hashlib.sha256(material + bytes([t])).digest()[:6], "big")
                 for t in range(VOCAB))


def serial_scores(history, geometries=GEOMETRIES):
    query = len(history) - 1
    return scores(tuple(tuple(record(history, p, layer) for p in range(
        0 if window is None else max(0, query - window + 1), query + 1))
        for layer, (_, _, window) in enumerate(geometries)))


def serial_tokens(history, count, geometries=GEOMETRIES):
    emitted = ()
    for _ in range(count):
        row = serial_scores(history + emitted, geometries)
        emitted += (sorted(range(VOCAB), key=lambda t: (-row[t], t))[0],)
    return emitted


def session(history, geometries=GEOMETRIES):
    layers = []
    for layer, (capacity, page, window) in enumerate(geometries):
        slots = [None] * capacity
        for position in range(len(history) - 1):
            slots[position % capacity if window else position] = record(history, position, layer)
        layers.append(v.LayerCache(capacity, page, window, tuple(slots)))
    return v.Session("synthetic-request", 7, 0, history, tuple(layers),
                     v.RngState.seeded(1234), "synthetic-policy-only")


class ByteTarget:
    execution_kind = "cpu_contract"
    supported_rows = tuple(range(1, 9))
    vocabulary_size = VOCAB
    max_kv_bytes_per_row = 128

    def __init__(self, logits=None, laws=None, mutate=lambda x: x):
        self.logits, self.laws, self.mutate = logits, laws, mutate
        self.calls, self.requests = 0, []

    def forward(self, request):
        self.calls += 1
        self.requests.append(request)
        history = request.session.history + request.candidates
        layers = tuple(tuple(record(history, p, layer) for p in request.input_positions)
                       for layer in range(len(request.session.layers)))
        rows = tuple(scores(tuple(request.visible(layer, j, staged)
                                 for layer, staged in enumerate(layers)))
                     for j in range(len(request.inputs)))
        batch = v.TargetBatch(request.session.request_id, request.session.epoch,
                              request.session.generation, request.session.policy_id,
                              request.input_positions, request.prediction_positions,
                              layers, self.logits if self.logits is not None else rows,
                              self.laws if self.laws is not None else ())
        return self.mutate(batch)


def one_hot(*tokens):
    return tuple(tuple(int(t == chosen) for t in range(VOCAB)) for chosen in tokens)


class TransactionTests(unittest.TestCase):
    def test_default_off_and_native_rejected_before_forward(self):
        initial, target = session((2, 3)), ByteTarget()
        harness = v.TargetVerifier(initial, target)
        with self.assertRaisesRegex(v.VerificationError, "disabled"):
            harness.verify((4,), remaining_budget=5)
        target.execution_kind = "native"
        with self.assertRaisesRegex(v.VerificationError, "no native"):
            v.TargetVerifier(initial, target, enabled=True)
        self.assertEqual(target.calls, 0)

    def test_all_acceptance_counts_exact_rows_bytes_and_next_forward(self):
        # Page edges, 1024 window boundaries, repeated wraps and k=0 control.
        for c, k in itertools.product((0, 1, 7, 8, 15, 16, 1023, 1024, 1025, 2049), (0, 1, 2, 4, 7)):
            initial = session(tuple(i % VOCAB for i in range(c + 1)))
            correct = serial_tokens(initial.history, k + 1)
            for accepted in range(k + 1):
                candidates = list(correct[:k])
                if accepted < k:
                    candidates[accepted] = (candidates[accepted] + 1) % VOCAB
                candidates = tuple(candidates)
                target = ByteTarget()
                harness = v.TargetVerifier(initial, target, enabled=True)
                result = harness.verify(candidates, remaining_budget=32)
                rows = tuple(serial_scores(initial.history + candidates[:j]) for j in range(k + 1))
                expected = oracle.select_greedy(oracle.VerifyBlock(c, initial.anchor, candidates), rows, max_new_tokens=32)
                self.assertEqual((result.emitted, result.accepted, result.target_kind, result.stop_reason),
                                 (expected.emitted, expected.accepted_draft_tokens, expected.target_token_kind, expected.stop_reason))
                self.assertEqual(result.emitted, correct[:accepted + 1])
                after = harness.session
                self.assertEqual(after.cached_length, c + len(result.emitted))
                rebuilt = session(initial.history + result.emitted)
                self.assertEqual(after.layers, rebuilt.layers)
                # Compare every full synthetic logit row, not just argmax IDs.
                request = target.requests[-1]
                fresh = ByteTarget().forward(request)
                self.assertEqual(fresh.logits, rows)
                next_result = harness.verify((), remaining_budget=1)
                self.assertEqual(next_result.emitted, serial_tokens(after.history, 1))
                self.assertEqual(harness.session.layers, session(after.history + next_result.emitted).layers)

    def test_prefill_sampled_anchor_transition_is_not_reemitted(self):
        prompt = tuple(i % VOCAB for i in range(16))
        first = serial_tokens(prompt, 1)[0]  # prefill last logits predict P
        initial = session(prompt + (first,))  # only P prompt inputs cached
        target = ByteTarget()
        harness = v.TargetVerifier(initial, target, enabled=True)
        result = harness.verify((), remaining_budget=1)  # first already counted
        self.assertEqual(target.requests[0].inputs, (first,))
        self.assertEqual(target.requests[0].input_positions, (len(prompt),))
        self.assertEqual(target.requests[0].prediction_positions, (len(prompt) + 1,))
        self.assertEqual(result.emitted, serial_tokens(prompt, 2)[1:])

    def test_staging_longer_than_ring_and_page_addresses(self):
        geometry = ((64, 16, None), (4, 3, 4))
        initial = session(tuple(i % VOCAB for i in range(18)), geometry)
        correct = serial_tokens(initial.history, 8, geometry)
        harness = v.TargetVerifier(initial, ByteTarget(), enabled=True)
        result = harness.verify(correct[:7], remaining_budget=8)
        self.assertEqual(result.emitted, correct)
        self.assertEqual(harness.session.layers, session(initial.history + correct, geometry).layers)
        self.assertEqual(initial.layers[0].address(16), (1, 0))
        self.assertEqual(initial.layers[1].address(19), (1, 0))

    def test_budget_eos_and_terminal_no_forward(self):
        for candidates, choices, eos, budget in (
                ((3, 4), (3, 4, 5), (), 1),  # accepted budget stop, no bonus
                ((3, 4), (3, 4, 5), (3,), 8),
                ((3, 4), (3, 4, 5), (4,), 8),
                ((3, 4), (6, 4, 5), (6,), 8),  # fallback EOS
                ((3, 4), (3, 4, 5), (5,), 8),  # bonus EOS
                ((3, 4), (6, 4, 5), (3,), 8)):  # rejected draft EOS
            target = ByteTarget(logits=one_hot(*choices))
            harness = v.TargetVerifier(session((2,)), target, enabled=True)
            result = harness.verify(candidates, eos_token_ids=eos, remaining_budget=budget)
            expected = oracle.select_greedy(oracle.VerifyBlock(0, 2, candidates), one_hot(*choices),
                                            eos_token_ids=eos, max_new_tokens=budget)
            self.assertEqual(result.emitted, expected.emitted)
            self.assertEqual(result.stop_reason, expected.stop_reason)
            if result.stop_reason:
                self.assertEqual(harness.verify((), remaining_budget=5).emitted, ())
                self.assertEqual(target.calls, 1)
        for history, budget, eos in (((2,), 0, ()), ((3,), 4, (3,))):
            target = ByteTarget()
            initial = session(history)
            harness = v.TargetVerifier(initial, target, enabled=True)
            self.assertIsNone(harness.verify((), remaining_budget=budget, eos_token_ids=eos, cancelled=lambda: True))
            self.assertIs(harness.session, initial)
            self.assertEqual(harness.verify((), remaining_budget=budget, eos_token_ids=eos).emitted, ())
            self.assertEqual(target.calls, 0)

    def test_capacity_shape_and_staging_guards_before_dispatch(self):
        cases = (
            (session((2,)), {"max_candidates": 1}, (3, 4), 4),
            (session((2,)), {"staging_rows": 1}, (3,), 4),
            (session((2,)), {"staging_bytes": 127}, (), 1),
            (session((2, 3), ((2, 1, None), (8, 3, 8))), {}, (3, 4), 4),
        )
        for initial, kwargs, candidates, budget in cases:
            target = ByteTarget()
            harness = v.TargetVerifier(initial, target, enabled=True, **kwargs)
            with self.assertRaises(v.VerificationError):
                harness.verify(candidates, remaining_budget=budget)
            self.assertIs(harness.session, initial)
            self.assertEqual(target.calls, 0)
        # A discarded suffix can live in staging beyond final global capacity.
        target = ByteTarget(logits=one_hot(3, 4, 5))
        harness = v.TargetVerifier(session((2, 3), ((2, 1, None), (8, 3, 8))), target, enabled=True)
        harness.verify((3, 4), remaining_budget=1)
        self.assertEqual(harness.session.cached_length, 2)
        target = ByteTarget()
        target.supported_rows = (1,)
        with self.assertRaises(v.VerificationError):
            v.TargetVerifier(session((2,)), target, enabled=True).verify((3,), remaining_budget=5)
        self.assertEqual(target.calls, 0)

    def test_bad_batches_leave_all_bytes_frontier_and_rng_untouched(self):
        mutations = (
            lambda b: replace(b, generation=b.generation + 1),
            lambda b: replace(b, generation=False),
            lambda b: replace(b, request_id="other"),
            lambda b: replace(b, epoch=8),
            lambda b: replace(b, policy_id="other"),
            lambda b: replace(b, input_positions=tuple(p + 1 for p in b.input_positions)),
            lambda b: replace(b, prediction_positions=b.input_positions),
            lambda b: replace(b, layers=b.layers[:-1]),
            lambda b: replace(b, layers=(b.layers[0][:-1], b.layers[1])),
            lambda b: replace(b, layers=((replace(b.layers[0][0], token=8),) + b.layers[0][1:], b.layers[1])),
            lambda b: replace(b, layers=((replace(b.layers[0][0], k=b"x" * 500),) + b.layers[0][1:], b.layers[1])),
            # Total bytes fit, but one row violates the advertised per-row bound.
            lambda b: replace(b, layers=tuple(tuple(
                replace(row, k=b"x" * 64) if layer == 0 and j == 0 else
                replace(row, k=b"x", v=b"y") if j > 0 else row
                for j, row in enumerate(rows)) for layer, rows in enumerate(b.layers))),
            lambda b: replace(b, logits=b.logits[:-1]),
            lambda b: replace(b, logits=(tuple([math.nan] * VOCAB),) + b.logits[1:]),
            lambda b: replace(b, logits=b.logits[:-1] + (tuple([-math.inf] * VOCAB),)),
            lambda b: replace(b, logits=(b.logits[0][:-1],) + b.logits[1:]),
        )
        initial = session(tuple(i % VOCAB for i in range(17)))
        for mutate in mutations:
            harness = v.TargetVerifier(initial, ByteTarget(mutate=mutate), enabled=True)
            with self.assertRaises(v.VerificationError):
                harness.verify((3, 4), remaining_budget=5)
            self.assertIs(harness.session, initial)
        with self.assertRaises(v.VerificationError):
            replace(initial.layers[1].slots[0], rope_position=0)

    def test_cancellation_each_boundary_retry_and_backend_failure(self):
        initial = session(tuple(i % VOCAB for i in range(17)))
        for cancel_at in range(3):
            target = ByteTarget()
            harness = v.TargetVerifier(initial, target, enabled=True)
            checks = iter(i == cancel_at for i in range(3))
            self.assertIsNone(harness.verify((3, 4), remaining_budget=5, cancelled=lambda: next(checks)))
            self.assertIs(harness.session, initial)
            retry = harness.verify((3, 4), remaining_budget=5)
            fresh = v.TargetVerifier(initial, ByteTarget(), enabled=True)
            self.assertEqual(retry, fresh.verify((3, 4), remaining_budget=5))
            self.assertEqual(harness.session, fresh.session)
        def fail(batch):
            raise RuntimeError("synthetic backend fault")
        harness = v.TargetVerifier(initial, ByteTarget(mutate=fail), enabled=True)
        with self.assertRaises(RuntimeError):
            harness.verify((3,), remaining_budget=5)
        self.assertIs(harness.session, initial)

    def test_row_processors_receive_hypothetical_prefix_and_suffix_is_hidden(self):
        initial = session(tuple(i % VOCAB for i in range(9)))
        request = v.ForwardRequest(initial, (3, 4, 5), "greedy")
        batch = ByteTarget().forward(request)
        for row in range(4):
            self.assertEqual(request.prefix(row), initial.history + (3, 4, 5)[:row])
            visible = request.visible(1, row, batch.layers[1])
            self.assertEqual(tuple(r.position for r in visible), tuple(range(1 + row, 9 + row)))
        # Deliberately exposing the whole future block changes row0 scores.
        wrong = scores(tuple(request.visible(i, 3, staged) for i, staged in enumerate(batch.layers)))
        self.assertNotEqual(wrong, batch.logits[0])

    def test_reentrant_call_does_not_publish(self):
        target = ByteTarget()
        harness = v.TargetVerifier(session((2,)), target, enabled=True)
        target.mutate = lambda batch: harness.verify((), remaining_budget=1)
        initial = harness.session
        with self.assertRaisesRegex(v.VerificationError, "reentrant"):
            harness.verify((), remaining_budget=1)
        self.assertIs(harness.session, initial)


class GridRng:
    """Exact enumerated independent RNG fixture, not a native RNG simulation."""
    def __init__(self, uniform, state):
        self.uniform, self.state = uniform, state

    def randrange(self, denominator):
        return int(self.uniform * denominator)

    def getstate(self):
        return self.state


class StochasticTests(unittest.TestCase):
    def test_exact_two_token_joint_law_over_independent_proposal_accept_target_grids(self):
        p0, q = (F(3, 4), F(1, 4)), (F(1, 2), F(1, 2))
        joint = defaultdict(F)
        states = v.RngState.seeded(12)
        # Each target draw is independent of proposal and acceptance. Following
        # fallback, a new independent target-only cycle finishes the joint law.
        for y, a, t, next_t in itertools.product(range(2), range(4), range(4), range(4)):
            p1 = (F(1, 2), F(1, 2)) if y == 0 else (F(1, 4), F(3, 4))
            rows = tuple(row + (F(0),) * 7 for row in (p0, p1))
            harness = v.TargetVerifier(session((2,)), ByteTarget(laws=rows), enabled=True)
            with patch.object(v.RngState, "generators", return_value=(GridRng(F(a, 4), states.acceptance),
                                                                       GridRng(F(t, 4), states.target))):
                result = harness.verify((y,), mode="stochastic", q_rows=(q + (F(0),) * 7,), remaining_budget=2)
            emitted = result.emitted
            if len(emitted) == 1:
                next_law = (F(1, 2), F(1, 2)) if emitted[0] == 0 else (F(1, 4), F(3, 4))
                target = ByteTarget(laws=(next_law + (F(0),) * 7,))
                continuation = v.TargetVerifier(harness.session, target, enabled=True)
                with patch.object(v.RngState, "generators", return_value=(GridRng(F(0), states.acceptance),
                                                                           GridRng(F(next_t, 4), states.target))):
                    emitted += continuation.verify((), mode="stochastic", remaining_budget=1).emitted
            joint[emitted] += F(1, 128)
        self.assertEqual(dict(joint), {(0, 0): F(3, 8), (0, 1): F(3, 8),
                                      (1, 0): F(1, 16), (1, 1): F(3, 16)})

    def test_fixed_seed_decisions_match_exact_reference_and_retry(self):
        for seed in range(50):
            for k in (0, 1, 2, 4, 7):
                p = (F(3, 4), F(1, 4)) + (F(0),) * 7
                q = (F(1, 2), F(1, 2)) + (F(0),) * 7
                candidates = tuple(j % 2 for j in range(k))
                initial = replace(session((2,)), rng=v.RngState.seeded(seed))
                arng, trng = initial.rng.generators()
                uniforms = tuple(F(arng.randrange((min(F(1), p[y] / q[y])).denominator),
                                   (min(F(1), p[y] / q[y])).denominator) for y in candidates)
                rejected = next((j for j, y in enumerate(candidates) if uniforms[j] >= min(F(1), p[y] / q[y])), None)
                target_law = p if rejected is None else oracle.residual_distribution(p, q)
                denominator = math.lcm(*(mass.denominator for mass in target_law))
                tu = F(trng.randrange(denominator), denominator)
                expected = oracle.select_stochastic(oracle.VerifyBlock(0, 2, candidates), (p,) * (k + 1),
                                                     (q,) * k, uniforms, tu, max_new_tokens=20)
                harness = v.TargetVerifier(initial, ByteTarget(laws=(p,) * (k + 1)), enabled=True)
                result = harness.verify(candidates, mode="stochastic", q_rows=(q,) * k, remaining_budget=20)
                self.assertEqual((result.emitted, result.accepted, result.target_kind),
                                 (expected.emitted, expected.accepted_draft_tokens, expected.target_token_kind))
                fresh = v.TargetVerifier(initial, ByteTarget(laws=(p,) * (k + 1)), enabled=True)
                self.assertEqual(result, fresh.verify(candidates, mode="stochastic", q_rows=(q,) * k, remaining_budget=20))
                self.assertEqual(harness.session, fresh.session)

    def test_one_hot_supplied_candidates_zeros_equal_laws_and_eos(self):
        zero_one = tuple(F(x) for x in one_hot(0)[0])
        one_zero = tuple(F(x) for x in one_hot(1)[0])
        for laws, candidates, q_rows, eos, expected in (
                ((one_zero, zero_one), (0,), None, (), (1,)),
                ((zero_one, one_zero), (0,), None, (), (0, 1)),
                ((zero_one, one_zero), (0,), (zero_one,), (0,), (0,))):
            harness = v.TargetVerifier(session((2,)), ByteTarget(laws=laws), enabled=True)
            result = harness.verify(candidates, mode="stochastic", q_rows=q_rows, eos_token_ids=eos, remaining_budget=8)
            self.assertEqual(result.emitted, expected)
            self.assertEqual(harness.session.layers, session((2,) + expected).layers)
        initial = session((2,))
        for rows in ((one_zero,), ((F(0.5), F(0.4)) + (F(0),) * 7,), ((0.5, 0.5) + (0,) * 7,)):
            target = ByteTarget(laws=(zero_one, one_zero))
            harness = v.TargetVerifier(initial, target, enabled=True)
            with self.assertRaises(v.VerificationError):
                harness.verify((0,), mode="stochastic", q_rows=rows, remaining_budget=5)
            self.assertEqual(target.calls, 0)
            self.assertIs(harness.session, initial)

    def test_stochastic_cancellation_restores_rng_and_independent_domains(self):
        initial = session((2,))
        self.assertNotEqual(initial.rng.acceptance, initial.rng.target)
        with self.assertRaises(v.VerificationError):
            replace(initial, rng=v.RngState(initial.rng.target, initial.rng.target))
        law = (F(1, 2), F(1, 2)) + (F(0),) * 7
        harness = v.TargetVerifier(initial, ByteTarget(laws=(law, law)), enabled=True)
        checks = iter((False, False, True))
        self.assertIsNone(harness.verify((1,), mode="stochastic", remaining_budget=8, cancelled=lambda: next(checks)))
        self.assertIs(harness.session, initial)
        retry = harness.verify((1,), mode="stochastic", remaining_budget=8)
        fresh = v.TargetVerifier(initial, ByteTarget(laws=(law, law)), enabled=True)
        self.assertEqual(retry, fresh.verify((1,), mode="stochastic", remaining_budget=8))
        self.assertEqual(harness.session.rng, fresh.session.rng)


if __name__ == "__main__":
    unittest.main()
