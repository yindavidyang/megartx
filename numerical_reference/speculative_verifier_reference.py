"""CPU-only specification oracle for short speculative verifier transactions.

No model, native runtime, draft architecture, GPU API or external dependency is
used here. KV payloads are opaque processed bytes supplied by the caller. The
oracle specifies row alignment, exact-rational sampling and cache ownership;
it cannot qualify Gemma arithmetic, native synchronization or performance.
"""

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Optional, Sequence


class SpecContractError(ValueError):
    """A proposal, probability or cache contract is inconsistent."""


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise SpecContractError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class VerifyBlock:
    """The anchor is already emitted, but has no target KV yet.

    cached_length is the next absolute position to materialize. Row j consumes
    inputs[j] and predicts position cached_length+j+1. Proposals contain only
    the scheduled prefix; the draft may have computed a longer discarded block.
    """

    cached_length: int
    anchor: int
    proposals: tuple[int, ...]

    def __post_init__(self):
        _integer(self.cached_length, "cached_length")
        _integer(self.anchor, "anchor")
        if not isinstance(self.proposals, tuple):
            raise SpecContractError("proposals must be an immutable tuple")
        for token in self.proposals:
            _integer(token, "proposal")

    @property
    def inputs(self):
        return (self.anchor,) + self.proposals

    @property
    def input_positions(self):
        return tuple(range(self.cached_length, self.cached_length + len(self.inputs)))

    @property
    def prediction_positions(self):
        return tuple(position + 1 for position in self.input_positions)


@dataclass(frozen=True)
class Decision:
    emitted: tuple[int, ...]
    accepted_draft_tokens: int
    target_token_kind: Optional[str]  # fallback, bonus, or no target token
    stop_reason: Optional[str]        # eos, budget, or continue

    @property
    def commit_count(self):
        # Consume anchor + emitted[:-1]. The final emitted token remains the
        # next pending anchor, including a terminal EOS or budget-stop token.
        return len(self.emitted)

    @property
    def pending_anchor(self):
        return self.emitted[-1] if self.emitted else None


def _limits(block, eos_token_ids, max_new_tokens):
    eos = frozenset(eos_token_ids)
    for token in eos:
        _integer(token, "EOS token")
    if max_new_tokens is not None:
        _integer(max_new_tokens, "max_new_tokens")
    if block.anchor in eos:
        return eos, Decision((), 0, None, "eos")
    if max_new_tokens == 0:
        return eos, Decision((), 0, None, "budget")
    return eos, None


def _stop(token, emitted_count, eos, budget):
    if token in eos:
        return "eos"
    if budget is not None and emitted_count >= budget:
        return "budget"
    return None


def greedy_token(logits):
    """Lowest vocabulary ID wins an exact tie; -inf represents a masked ID."""
    if not logits or any(isinstance(x, bool) or not isinstance(x, (int, float))
                         or math.isnan(x) or x == math.inf for x in logits):
        raise SpecContractError("logits must be nonempty, numeric and contain no NaN/+inf")
    if max(logits) == -math.inf:
        raise SpecContractError("all logits are masked")
    return max(range(len(logits)), key=lambda token: logits[token])


def select_greedy(block, target_logits, *, eos_token_ids=(), max_new_tokens=None):
    """Accept the longest exact target-greedy prefix, then fallback or bonus.

    target_logits[j] must already include the target-only logit processors at
    that prefix. Rows after a rejection are never used to select output.
    """
    eos, early = _limits(block, eos_token_ids, max_new_tokens)
    if early is not None:
        return early
    if len(target_logits) != len(block.inputs):
        raise SpecContractError("verification requires k+1 target rows")
    widths = {len(row) for row in target_logits}
    if len(widths) != 1 or not widths or min(widths) == 0:
        raise SpecContractError("target rows must share a nonempty vocabulary")
    if any(token >= min(widths) for token in block.inputs):
        raise SpecContractError("input token is outside the target vocabulary")
    targets = tuple(greedy_token(row) for row in target_logits)
    emitted = []
    for index, proposal in enumerate(block.proposals):
        if proposal != targets[index]:
            emitted.append(targets[index])
            return Decision(tuple(emitted), index, "fallback",
                            _stop(emitted[-1], len(emitted), eos, max_new_tokens))
        emitted.append(proposal)
        reason = _stop(proposal, len(emitted), eos, max_new_tokens)
        if reason is not None:
            return Decision(tuple(emitted), index + 1, None, reason)
    emitted.append(targets[-1])
    return Decision(tuple(emitted), len(block.proposals), "bonus",
                    _stop(emitted[-1], len(emitted), eos, max_new_tokens))


def distribution(values):
    """Require exact normalized rational mass; float rounding is out of scope."""
    if not values or any(isinstance(x, bool) or not isinstance(x, (int, Fraction))
                         for x in values):
        raise SpecContractError("use nonempty int/Fraction probabilities")
    result = tuple(Fraction(x) for x in values)
    if min(result) < 0 or sum(result) != 1:
        raise SpecContractError("probabilities must be nonnegative and sum exactly to one")
    return result


def _pair(p, q):
    p, q = distribution(p), distribution(q)
    if len(p) != len(q):
        raise SpecContractError("target/draft vocabulary sizes differ")
    return p, q


def acceptance_probability(p, q, token):
    p, q = _pair(p, q)
    _integer(token, "proposal")
    if token >= len(p) or q[token] == 0:
        raise SpecContractError("proposal must have positive mass under its actual draft law")
    return min(Fraction(1), p[token] / q[token])


def residual_distribution(p, q):
    p, q = _pair(p, q)
    positive = tuple(max(Fraction(0), target - draft) for target, draft in zip(p, q))
    total = sum(positive)
    if total == 0:
        raise SpecContractError("equal distributions have no possible rejection/residual")
    return tuple(mass / total for mass in positive)


def _uniform(value):
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise SpecContractError("uniform must be an int/Fraction in [0,1)")
    value = Fraction(value)
    if not 0 <= value < 1:
        raise SpecContractError("uniform must be in [0,1)")
    return value


def sample_distribution(values, uniform):
    values, uniform = distribution(values), _uniform(uniform)
    cumulative = Fraction(0)
    for token, mass in enumerate(values):
        cumulative += mass
        if uniform < cumulative:
            return token
    raise AssertionError("normalized rational distribution must select a token")


def select_stochastic(block, p_rows, q_rows, accept_uniforms, target_uniform,
                      *, eos_token_ids=(), max_new_tokens=None):
    """Exact-rational rejection sampling, with explicit caller-owned uniforms.

    q_rows describe the actual conditional proposal law, including any Markov
    head, draft-vocabulary map and truncation. p_rows are the postprocessed
    target laws at the same proposal-conditioned prefixes. This does not imply
    identical output to target-only decoding under the same random seed.
    """
    eos, early = _limits(block, eos_token_ids, max_new_tokens)
    if early is not None:
        return early
    count = len(block.proposals)
    if len(p_rows) != count + 1 or len(q_rows) != count or len(accept_uniforms) != count:
        raise SpecContractError("need k+1 target laws, k draft laws and k acceptance uniforms")
    p_rows = tuple(distribution(row) for row in p_rows)
    q_rows = tuple(distribution(row) for row in q_rows)
    if len({len(row) for row in p_rows + q_rows}) != 1:
        raise SpecContractError("probability rows must share the target vocabulary")
    if any(token >= len(p_rows[0]) for token in block.inputs):
        raise SpecContractError("input token is outside the target vocabulary")
    uniforms = tuple(_uniform(value) for value in accept_uniforms)
    target_uniform = _uniform(target_uniform)
    emitted = []
    for index, proposal in enumerate(block.proposals):
        alpha = acceptance_probability(p_rows[index], q_rows[index], proposal)
        if uniforms[index] >= alpha:
            token = sample_distribution(residual_distribution(p_rows[index], q_rows[index]),
                                        target_uniform)
            emitted.append(token)
            return Decision(tuple(emitted), index, "fallback",
                            _stop(token, len(emitted), eos, max_new_tokens))
        emitted.append(proposal)
        reason = _stop(proposal, len(emitted), eos, max_new_tokens)
        if reason is not None:
            return Decision(tuple(emitted), index + 1, None, reason)
    token = sample_distribution(p_rows[-1], target_uniform)
    emitted.append(token)
    return Decision(tuple(emitted), count, "bonus",
                    _stop(token, len(emitted), eos, max_new_tokens))


@dataclass(frozen=True)
class KVRecord:
    position: int
    token_id: int
    rope_position: int
    k: bytes
    v: bytes

    def __post_init__(self):
        _integer(self.position, "KV absolute position")
        _integer(self.token_id, "KV token")
        if type(self.rope_position) is not int or self.rope_position != self.position:
            raise SpecContractError("RoPE position must be absolute, never a ring/page index")
        if type(self.k) is not bytes or type(self.v) is not bytes or not self.k or not self.v:
            raise SpecContractError("processed K/V must be separate immutable nonempty byte payloads")


@dataclass(frozen=True)
class CacheGeometry:
    capacity: int
    page_size: int
    window: Optional[int] = None

    def __post_init__(self):
        _integer(self.capacity, "capacity", 1)
        _integer(self.page_size, "page_size", 1)
        if self.window is not None:
            _integer(self.window, "window", 1)
            if self.capacity < self.window:
                raise SpecContractError("ring capacity must retain the whole sliding window")

    def address(self, position):
        _integer(position, "absolute position")
        slot = position % self.capacity if self.window is not None else position
        if slot >= self.capacity:
            raise SpecContractError("global output capacity exhausted")
        return slot // self.page_size, slot % self.page_size


@dataclass(frozen=True)
class CacheSnapshot:
    length: int
    generation: int
    slots: tuple[Optional[KVRecord], ...]


class ReferenceCache:
    """Single-thread CPU state model; physical slots never receive tentative KV."""

    def __init__(self, geometry, committed_rows: Sequence[KVRecord] = ()):
        self.geometry = geometry
        slots = [None] * geometry.capacity
        for position, row in enumerate(committed_rows):
            if not isinstance(row, KVRecord) or row.position != position:
                raise SpecContractError("committed initializer must be contiguous from position zero")
            page, offset = geometry.address(position)
            slots[page * geometry.page_size + offset] = row
        self._state = CacheSnapshot(len(committed_rows), 0, tuple(slots))

    @property
    def snapshot(self):
        return self._state

    def begin(self):
        return CacheTransaction(self)


class CacheTransaction:
    def __init__(self, cache):
        self.cache = cache
        self.base = cache.snapshot
        self._staged = []
        self._closed = False

    def _check(self):
        if self._closed:
            raise SpecContractError("transaction is closed")
        if self.cache.snapshot != self.base:
            raise SpecContractError("stale cache generation or changed committed bytes")

    def stage(self, row):
        self._check()
        if not isinstance(row, KVRecord) or row.position != self.base.length + len(self._staged):
            raise SpecContractError("tentative rows must be contiguous after the cached prefix")
        self._staged.append(row)

    def visible(self, query_position):
        """Only this causal row and its past are visible; future suffix is masked."""
        self._check()
        _integer(query_position, "query position")
        if not self.base.length <= query_position < self.base.length + len(self._staged):
            raise SpecContractError("query must name a staged verifier row")
        window = self.cache.geometry.window
        first = 0 if window is None else max(0, query_position - window + 1)
        visible = []
        for position in range(first, query_position + 1):
            if position >= self.base.length:
                row = self._staged[position - self.base.length]
            else:
                page, offset = self.cache.geometry.address(position)
                row = self.base.slots[page * self.cache.geometry.page_size + offset]
            if row is None or row.position != position:
                raise SpecContractError("needed committed KV was overwritten or is missing")
            visible.append(row)
        return tuple(visible)

    def prepare_commit(self, count):
        """Build a candidate snapshot; any failure occurs before publication."""
        self._check()
        _integer(count, "commit count")
        if count > len(self._staged):
            raise SpecContractError("cannot commit uncomputed rows")
        if count == 0:
            return self.base
        slots = list(self.base.slots)
        for row in self._staged[:count]:
            page, offset = self.cache.geometry.address(row.position)
            slots[page * self.cache.geometry.page_size + offset] = row
        return CacheSnapshot(self.base.length + count, self.base.generation + 1, tuple(slots))

    def abort(self):
        if self._closed:
            raise SpecContractError("transaction is closed")
        self._staged.clear()
        self._closed = True


def commit_group(transactions, counts, *, cancelled=False):
    """Prepare all layers before installing any. No native atomicity is claimed."""
    transactions, counts = tuple(transactions), tuple(counts)
    if not transactions or len(transactions) != len(counts):
        raise SpecContractError("one commit count is required for every cache layer")
    if len({id(tx.cache) for tx in transactions}) != len(transactions):
        raise SpecContractError("duplicate cache ownership in a commit group")
    if len({tx.base.length for tx in transactions}) != 1:
        raise SpecContractError("layer cached prefix lengths must agree before commit")
    if cancelled:
        for tx in transactions:
            tx._check()
        for tx in transactions:
            tx.abort()
        return False
    candidates = tuple(tx.prepare_commit(count) for tx, count in zip(transactions, counts))
    if len({state.length for state in candidates}) != 1:
        raise SpecContractError("layer committed lengths must agree")
    for tx, state in zip(transactions, candidates):
        tx.cache._state = state
        tx._staged.clear()
        tx._closed = True
    return True
