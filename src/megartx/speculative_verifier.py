"""Opt-in, single-thread CPU harness for supplied speculative candidates.

Backends return processed logits/laws and immutable *staged* KV witnesses. This
is an executable interface contract, not a vLLM plugin or native cache adapter.
No model loading, device execution, softmax or logit processors live here.
The independent oracle remains in numerical_reference, never imported here.
"""

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import math
import random
from typing import Optional, Protocol


class VerificationError(ValueError):
    """Unsupported or inconsistent verification contract; no state published."""


def _int(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise VerificationError(f"{label} must be an integer >= {minimum}")


def _tuple(value, label):
    if type(value) is not tuple:
        raise VerificationError(f"{label} must be an immutable tuple")


def _law(row):
    _tuple(row, "probability row")
    if not row or any(type(v) not in (int, Fraction) for v in row):
        raise VerificationError("only exact int/Fraction laws are supported")
    row = tuple(Fraction(v) for v in row)
    if any(v < 0 for v in row) or sum(row) != 1:
        raise VerificationError("laws must be nonnegative and exactly normalized")
    return row


def _best(row):
    _tuple(row, "logit row")
    if not row or any(type(v) not in (int, float) or (type(v) is float and math.isnan(v))
                      or v == math.inf for v in row) or max(row) == -math.inf:
        raise VerificationError("invalid or entirely masked processed logits")
    return max(range(len(row)), key=row.__getitem__)  # lowest ID on ties


def _sample(law, rng):
    # Integer rejection sampling avoids a rounded 53-bit uniform grid. Exact
    # probabilities are conditional on an ideal independent randrange source;
    # Python's deterministic PRNG is a CPU replay facility, not native proof.
    denominator = math.lcm(*(p.denominator for p in law))
    draw = rng.randrange(denominator)
    for token, mass in enumerate(law):
        draw -= mass.numerator * (denominator // mass.denominator)
        if draw < 0:
            return token
    raise AssertionError("normalized law must select a token")


@dataclass(frozen=True)
class RngState:
    acceptance: tuple
    target: tuple

    @classmethod
    def seeded(cls, seed):
        _int(seed, "seed")
        states = []
        for domain in ("acceptance", "correction-bonus"):
            material = hashlib.sha256(f"megartx-verifier-v1:{domain}:{seed}".encode()).digest()
            states.append(random.Random(int.from_bytes(material, "big")).getstate())
        return cls(*states)

    def generators(self):
        if self.acceptance == self.target:
            raise VerificationError("acceptance and target RNG streams must be distinct")
        generators = random.Random(), random.Random()
        for rng, state in zip(generators, (self.acceptance, self.target)):
            rng.setstate(state)
        return generators


@dataclass(frozen=True)
class KVRow:
    position: int
    token: int
    rope_position: int
    k: bytes
    v: bytes

    def __post_init__(self):
        _int(self.position, "KV position")
        _int(self.token, "KV token")
        if type(self.rope_position) is not int or self.rope_position != self.position:
            raise VerificationError("RoPE must use the absolute position")
        if type(self.k) is not bytes or type(self.v) is not bytes or not self.k or not self.v:
            raise VerificationError("K and V must be separate nonempty immutable byte payloads")


@dataclass(frozen=True)
class LayerCache:
    capacity: int
    page_size: int
    window: Optional[int]
    slots: tuple[Optional[KVRow], ...]

    def __post_init__(self):
        _int(self.capacity, "capacity", 1)
        _int(self.page_size, "page size", 1)
        if self.window is not None:
            _int(self.window, "window", 1)
            if self.capacity < self.window:
                raise VerificationError("ring must retain the whole window")
        _tuple(self.slots, "cache slots")
        if len(self.slots) != self.capacity:
            raise VerificationError("physical slot count differs from capacity")
        for slot, row in enumerate(self.slots):
            if row is not None and (type(row) is not KVRow or self.slot(row.position) != slot):
                raise VerificationError("invalid absolute tag/physical address")

    def slot(self, absolute):
        _int(absolute, "absolute position")
        if self.window is not None:
            return absolute % self.capacity
        if absolute >= self.capacity:
            raise VerificationError("global cache capacity exhausted")
        return absolute

    def address(self, absolute):
        return divmod(self.slot(absolute), self.page_size)


@dataclass(frozen=True)
class Session:
    request_id: str
    epoch: int
    generation: int
    history: tuple[int, ...]  # cached prefix + already emitted, uncached anchor
    layers: tuple[LayerCache, ...]
    rng: RngState
    policy_id: str           # frozen model/lane/tokenizer/processor identity
    stop_reason: Optional[str] = None

    def __post_init__(self):
        if (type(self.request_id) is not str or not self.request_id
                or type(self.policy_id) is not str or not self.policy_id):
            raise VerificationError("request and frozen policy identities are required")
        _int(self.epoch, "epoch")
        _int(self.generation, "generation")
        _tuple(self.history, "history")
        if not self.history:
            raise VerificationError("history must include a pending anchor")
        for token in self.history:
            _int(token, "history token")
        _tuple(self.layers, "layers")
        if not self.layers or any(type(layer) is not LayerCache for layer in self.layers):
            raise VerificationError("complete cache layer set required")
        if type(self.rng) is not RngState or self.stop_reason not in (None, "eos", "budget"):
            raise VerificationError("invalid RNG or stop state")
        self.rng.generators()  # validate replay states before any dispatch
        for layer in self.layers:
            first = 0 if layer.window is None else max(0, self.cached_length - layer.window)
            for position in range(first, self.cached_length):
                row = layer.slots[layer.slot(position)]
                if row is None or row.position != position or row.token != self.history[position]:
                    raise VerificationError("missing/mismatched committed prefix KV")
            if any(row is not None and row.position >= self.cached_length for row in layer.slots):
                raise VerificationError("pending anchor/suffix must not have committed KV")

    @property
    def cached_length(self):
        return len(self.history) - 1

    @property
    def anchor(self):
        return self.history[-1]


@dataclass(frozen=True)
class ForwardRequest:
    session: Session
    candidates: tuple[int, ...]
    mode: str

    @property
    def inputs(self):
        return (self.session.anchor,) + self.candidates

    @property
    def input_positions(self):
        return tuple(range(self.session.cached_length, self.session.cached_length + len(self.inputs)))

    @property
    def prediction_positions(self):
        return tuple(p + 1 for p in self.input_positions)

    def prefix(self, row):
        """History for row-specific processors, including the consumed input."""
        _int(row, "row")
        if row >= len(self.inputs):
            raise VerificationError("processor row outside block")
        return self.session.history + self.candidates[:row]

    def visible(self, layer_id, row, staged):
        """CPU causal/window view; staged rows cannot overwrite old ring slots."""
        _int(layer_id, "layer ID")
        _int(row, "row")
        _tuple(staged, "staged rows")
        if layer_id >= len(self.session.layers) or row >= len(staged) or row >= len(self.inputs):
            raise VerificationError("attention view outside layer/block")
        layer = self.session.layers[layer_id]
        query = self.input_positions[row]
        first = 0 if layer.window is None else max(0, query - layer.window + 1)
        rows = []
        for p in range(first, query + 1):
            record = (staged[p - self.session.cached_length] if p >= self.session.cached_length
                      else layer.slots[layer.slot(p)])
            expected = self.inputs[p - self.session.cached_length] if p >= self.session.cached_length else self.session.history[p]
            if type(record) is not KVRow or record.position != p or record.token != expected:
                raise VerificationError("missing or misaligned visible KV")
            rows.append(record)
        return tuple(rows)


@dataclass(frozen=True)
class TargetBatch:
    request_id: str
    epoch: int
    generation: int
    policy_id: str
    input_positions: tuple[int, ...]
    prediction_positions: tuple[int, ...]
    layers: tuple[tuple[KVRow, ...], ...]
    logits: tuple[tuple, ...] = ()
    laws: tuple[tuple, ...] = ()


class CpuTargetBackend(Protocol):
    """Backend must be side-effect-free and return all-layer staged bytes.

    Native asynchronous writers, tensors and device addresses are unsupported.
    The snapshot, request history, absolute positions and visible() helper are
    sufficient for a small independent CPU target or captured fixture replay.
    """

    execution_kind: str
    supported_rows: tuple[int, ...]
    vocabulary_size: int
    max_kv_bytes_per_row: int  # all layers combined; reserve before dispatch

    def forward(self, request: ForwardRequest) -> TargetBatch: ...


@dataclass(frozen=True)
class CycleResult:
    emitted: tuple[int, ...]
    accepted: int
    target_kind: Optional[str]
    stop_reason: Optional[str]
    cached_length: int
    next_anchor: int
    generation: int


class TargetVerifier:
    """Default-off CPU transaction owner. One assignment publishes all state.

    k is chosen before proposal sampling in stochastic mode. Absent q_rows,
    supplied tokens are deterministic proposals with one-hot actual laws.
    General q_rows require the caller to supply the actual conditional proposal
    law and to generate proposals independently of this harness's two RNGs.
    """

    def __init__(self, session, backend, *, enabled=False, max_candidates=7,
                 staging_rows=8, staging_bytes=1 << 20):
        if type(session) is not Session or type(enabled) is not bool:
            raise VerificationError("immutable session and boolean opt-in required")
        _int(max_candidates, "candidate bound")
        _int(staging_rows, "staging row bound", 1)
        _int(staging_bytes, "staging byte bound", 1)
        if backend.execution_kind != "cpu_contract":
            raise VerificationError("no native/GPU adapter is implemented")
        _int(backend.vocabulary_size, "vocabulary size", 1)
        _int(backend.max_kv_bytes_per_row, "staged per-row byte bound", 1)
        _tuple(backend.supported_rows, "supported row counts")
        if not backend.supported_rows:
            raise VerificationError("backend must advertise supported row counts")
        for count in backend.supported_rows:
            _int(count, "supported row count", 1)
        if any(t >= backend.vocabulary_size for t in session.history):
            raise VerificationError("history outside target vocabulary")
        self.session, self.backend, self.enabled = session, backend, enabled
        self.max_candidates, self.staging_rows, self.staging_bytes = max_candidates, staging_rows, staging_bytes
        self._busy = False

    def verify(self, candidates, *, remaining_budget, eos_token_ids=(),
               mode="greedy", q_rows=None, cancelled=lambda: False):
        if not self.enabled:
            raise VerificationError("supplied-candidate verification is disabled")
        if self._busy:
            raise VerificationError("concurrent/reentrant verification is unsupported")
        self._busy = True
        try:
            return self._verify(candidates, remaining_budget, eos_token_ids, mode, q_rows, cancelled)
        finally:
            self._busy = False

    def _verify(self, candidates, budget, eos_ids, mode, q_rows, cancelled):
        before = self.session
        _tuple(candidates, "candidates")
        _int(budget, "remaining output budget")
        eos = frozenset(eos_ids)
        for token in eos:
            _int(token, "EOS token")
            if token >= self.backend.vocabulary_size:
                raise VerificationError("EOS outside target vocabulary")
        if mode not in ("greedy", "stochastic") or (mode == "greedy" and q_rows is not None):
            raise VerificationError("unsupported sampler contract")
        for token in candidates:
            _int(token, "candidate")
            if token >= self.backend.vocabulary_size:
                raise VerificationError("candidate outside target vocabulary")
        if len(candidates) > self.max_candidates:
            raise VerificationError("candidate count exceeds bound")
        reason = before.stop_reason or ("eos" if before.anchor in eos else "budget" if budget == 0 else None)
        if reason:
            if cancelled():
                return None
            self.session = replace(before, stop_reason=reason)
            return CycleResult((), 0, None, reason, before.cached_length, before.anchor, before.generation)
        # Reject unsupported shapes before dispatch. Reserve for the maximum
        # possible consumed prefix; discarded staging need not fit global KV.
        count = len(candidates) + 1
        if count > self.staging_rows or count not in self.backend.supported_rows:
            raise VerificationError("unsupported/unreserved target row shape")
        if count * self.backend.max_kv_bytes_per_row > self.staging_bytes:
            raise VerificationError("insufficient staged byte reservation before dispatch")
        for layer in before.layers:
            layer.slot(before.cached_length + min(count, budget) - 1)
        if mode == "stochastic":
            if q_rows is None:
                q_rows = tuple(tuple(int(v == t) for v in range(self.backend.vocabulary_size))
                               for t in candidates)
            _tuple(q_rows, "draft laws")
            if len(q_rows) != len(candidates):
                raise VerificationError("need one actual draft law per candidate")
            q_rows = tuple(_law(row) for row in q_rows)
            if any(len(row) != self.backend.vocabulary_size or row[t] == 0
                   for row, t in zip(q_rows, candidates)):
                raise VerificationError("candidate must have positive mass in target-vocabulary q")
        if cancelled():
            return None
        request = ForwardRequest(before, candidates, mode)
        batch = self.backend.forward(request)
        self._validate_batch(request, batch)
        if cancelled():
            return None
        accept_rng, target_rng = before.rng.generators()
        choices = tuple(_best(row) for row in batch.logits) if mode == "greedy" else ()
        laws = tuple(_law(row) for row in batch.laws) if mode == "stochastic" else ()
        emitted, accepted, kind, reason = [], 0, None, None
        for j in range(count):
            if j == len(candidates):
                token = choices[j] if mode == "greedy" else _sample(laws[j], target_rng)
                kind = "bonus"
            else:
                token = candidates[j]
                if mode == "greedy":
                    keep = token == choices[j]
                else:
                    alpha = min(Fraction(1), laws[j][token] / q_rows[j][token])
                    keep = accept_rng.randrange(alpha.denominator) < alpha.numerator
                if keep:
                    accepted += 1
                else:
                    if mode == "greedy":
                        token = choices[j]
                    else:
                        residual = tuple(max(Fraction(0), p - q) for p, q in zip(laws[j], q_rows[j]))
                        total = sum(residual)
                        if total == 0:
                            raise VerificationError("impossible rejection with empty residual")
                        token = _sample(tuple(p / total for p in residual), target_rng)
                    kind = "fallback"
            emitted.append(token)
            reason = "eos" if token in eos else "budget" if len(emitted) == budget else None
            if kind is not None or reason is not None:
                break
        committed = len(emitted)  # anchor + emissions[:-1]; final token pending
        layers = []
        for layer, staged in zip(before.layers, batch.layers):
            slots = list(layer.slots)
            for row in staged[:committed]:
                slots[layer.slot(row.position)] = row
            layers.append(replace(layer, slots=tuple(slots)))
        rng = RngState(accept_rng.getstate(), target_rng.getstate())
        after = replace(before, generation=before.generation + 1,
                        history=before.history + tuple(emitted), layers=tuple(layers),
                        rng=rng, stop_reason=reason)
        result = CycleResult(tuple(emitted), accepted, kind, reason, after.cached_length,
                             after.anchor, after.generation)
        if cancelled():
            return None
        if self.session is not before:
            raise VerificationError("session changed before publication")
        self.session = after  # CPU linearization point; no native atomicity claim
        return result

    def _validate_batch(self, request, batch):
        before = request.session
        if (type(batch) is not TargetBatch or type(batch.epoch) is not int
                or type(batch.generation) is not int or (batch.request_id, batch.epoch, batch.generation, batch.policy_id) != (
                before.request_id, before.epoch, before.generation, before.policy_id)):
            raise VerificationError("stale/wrong request, generation or policy")
        for positions in (batch.input_positions, batch.prediction_positions):
            _tuple(positions, "returned absolute positions")
            for position in positions:
                _int(position, "returned absolute position")
        if batch.input_positions != request.input_positions or batch.prediction_positions != request.prediction_positions:
            raise VerificationError("logits-position contract mismatch")
        _tuple(batch.layers, "staged layer set")
        if len(batch.layers) != len(before.layers):
            raise VerificationError("incomplete staged layer set")
        row_sizes = [0] * len(request.inputs)
        for rows in batch.layers:
            _tuple(rows, "staged layer rows")
            if len(rows) != len(request.inputs):
                raise VerificationError("need k+1 staged rows in every layer")
            for j, (row, p, t) in enumerate(zip(rows, request.input_positions, request.inputs)):
                if type(row) is not KVRow or row.position != p or row.token != t:
                    raise VerificationError("staged KV position/token mismatch")
                row_sizes[j] += len(row.k) + len(row.v)
        if (sum(row_sizes) > self.staging_bytes
                or max(row_sizes) > self.backend.max_kv_bytes_per_row):
            raise VerificationError("returned staged bytes exceed reservation")
        # Validate every returned row, even unused suffix rows, before RNG work.
        rows = batch.logits if request.mode == "greedy" else batch.laws
        _tuple(rows, "target rows")
        if len(rows) != len(request.inputs):
            raise VerificationError("need k+1 target rows")
        for row in rows:
            (_best if request.mode == "greedy" else _law)(row)
            if len(row) != self.backend.vocabulary_size:
                raise VerificationError("target row vocabulary mismatch")
