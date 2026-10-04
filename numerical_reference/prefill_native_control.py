"""CPU-only independent contract for one P2048 native cache control.

This module never imports the observer, Torch, vLLM, or an installed runtime.
Its byte inputs must come from separately source-bound native capture sites.
Synthetic success is not evidence that those sites ran, or a numerical oracle.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import struct


PROMPT = 2048
CHUNK = 256
OUTPUT = 256
CAPACITY = 2304
CAPTURE_END = 2049  # prompt plus first actually consumed, previously uncached anchor
LAYERS = tuple(range(30))
GLOBAL = (5, 11, 17, 23, 29)
MIB = 1 << 20
IDENTITY_FIELDS = (
    "checkpoint_identity_sha256", "original_scales_sha256", "quantizer_sha256",
    "tokenizer_sha256", "template_sha256", "prompt_ids_sha256",
    "model_arithmetic_sources_sha256", "environment_sha256", "sampler_sha256",
    "allocation_policy_sha256", "dispatch_sha256", "continuation_ids_sha256",
)


def integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(label + " must be an exact bounded integer")
    return value


def digest(value, label):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(label + " must be a lowercase SHA256")
    return value


def required_positions(layer, start, end):
    """Union for *all* queries [start,end), before any local recycling."""
    integer(layer, 0, 29, "layer")
    integer(start, 0, CAPACITY - 1, "absolute start")
    integer(end, start + 1, CAPACITY, "absolute end")
    return range(0 if layer in GLOBAL else max(0, start - 1023), end)


def row_bytes(layer):
    integer(layer, 0, 29, "layer")
    return 2 * (2 * 512 if layer in GLOBAL else 8 * 256)


def finite_bf16(raw, count):
    if type(raw) is not bytes or len(raw) != 2 * count:
        raise ValueError("Exact little-endian BF16 byte extent required")
    bits = struct.unpack("<" + "H" * count, raw)
    if any(word & 0x7F80 == 0x7F80 for word in bits):
        raise ValueError("Nonfinite BF16 input")
    return bits


def row_digest(layer, position, key, value):
    integer(position, 0, CAPACITY - 1, "absolute position")
    count = row_bytes(layer) // 2
    finite_bf16(key, count)
    finite_bf16(value, count)
    # K/V role, layer, and absolute position are part of the domain, not metadata
    # appended later. Equal raw K/V values are legal, swapped unequal values fail.
    tag = struct.pack("<II", layer, position)
    return hashlib.sha256(b"processed-k\0" + tag + key).digest(), hashlib.sha256(
        b"processed-v\0" + tag + value).digest()


def sample_positions(page_sizes):
    """At most eight rows, including actual first-page/window/handoff edges."""
    if type(page_sizes) is not tuple or len(page_sizes) != 30:
        raise ValueError("All thirty actual kernel page sizes required")
    for page in page_sizes:
        integer(page, 1, CHUNK, "actual kernel page size")
        if CHUNK % page:
            raise ValueError("Unreviewed nondivisor page size")
    if len(set(page_sizes)) > 2:
        raise ValueError("More than two actual page sizes need a new budget")
    return tuple(sorted({1023, 1024, 2047, 2048}
                        | {p - 1 for p in page_sizes} | set(page_sizes)))


def evidence_budget(page_sizes, contexts=2):
    """Combined private evidence, not a per-file or per-context allowance."""
    integer(contexts, 1, 2, "fresh contexts")
    samples = sample_positions(page_sizes)
    kv = contexts * len(samples) * sum(2 * row_bytes(i) for i in LAYERS)
    heads = contexts * 2 * 262144 * 2  # final-prompt and first-decode BF16 rows
    metadata = 2 * MIB  # shared by both contexts, temporary files included
    total = kv + heads + metadata
    if total > 8 * MIB:
        raise ValueError("Combined evidence exceeds 8 MiB")
    return {"sample_positions": samples, "kv_bytes": kv, "head_bytes": heads,
            "metadata_bytes": metadata, "total_bytes": total,
            "limit_bytes": 8 * MIB}


def capture_work_budget():
    """Bound transfer work separately from concurrent scratch and saved evidence."""
    frames = [Frame(p, p + CHUNK) for p in range(0, PROMPT, CHUNK)] + [Frame(PROMPT, CAPTURE_END)]
    transferred = 0
    row_reads = 0
    for frame in frames:
        for layer in LAYERS:
            required = required_positions(layer, frame.start, frame.end)
            # Pre-existing rows before write, actual processed source rows,
            # then the complete post-query union. All include both K and V.
            rows = (frame.start - required.start) + (frame.end - frame.start) + len(required)
            transferred += rows * 2 * row_bytes(layer)
            row_reads += rows
    return {"capture_frames": len(frames), "capture_end": CAPTURE_END,
            "transferred_bytes_per_context": transferred,
            "single_row_equivalent_reads": row_reads,
            "transfer_limit_bytes_per_context": 4 << 30}


def match_native_identity(control, candidate):
    """Require exact same-path identity; this is not a signature/attestation."""
    expected = set(IDENTITY_FIELDS) | {"prompt_tokens", "chunk_tokens", "output_tokens",
                                      "capacity_tokens", "compute_dtype", "kv_dtype"}
    if type(control) is not dict or type(candidate) is not dict:
        raise ValueError("Explicit identity objects required")
    for identity in (control, candidate):
        if set(identity) != expected:
            raise ValueError("Complete exact identity schema required")
        for field in IDENTITY_FIELDS:
            digest(identity[field], field)
        for field, fixed in (("prompt_tokens", PROMPT), ("chunk_tokens", CHUNK),
                             ("output_tokens", OUTPUT), ("capacity_tokens", CAPACITY)):
            integer(identity[field], fixed, fixed, field)
        if identity["compute_dtype"] != "bfloat16" or identity["kv_dtype"] != "bfloat16":
            raise ValueError("Unchanged BF16 numerical lane required")
    if control != candidate:
        raise ValueError("Matched native control identity differs")
    return True


@dataclass(frozen=True)
class Frame:
    start: int
    end: int

    def __post_init__(self):
        integer(self.start, 0, CAPACITY - 2, "frame start")
        expected = CHUNK if self.start < PROMPT else 1
        if self.start < PROMPT and self.start % CHUNK:
            raise ValueError("Unexpected prompt chunk start")
        integer(self.end, self.start + expected, self.start + expected, "frame end")
        if self.end > CAPACITY - 1:
            raise ValueError("Last emitted token must remain uncached")


class ProcessedKVControl:
    """Streaming exact writer-input -> physical-cache control, all thirty layers.

    Inputs are actual post-K-RMS/RoPE K and post-V-RMS V at the writer call.
    Retained expected bytes are SHA256 digests in host memory, never a full cache.
    A native adapter must derive table_slots independently from allocator tables,
    not copy writer slots or call the existing observer's indexing helpers.
    This first contract admits only full-context nonrecycled physical storage
    through the first consumed anchor. Later continuation storage is not checked.
    """

    def __init__(self):
        self.end = 0
        self.active = None
        self.poisoned = False
        self.expected = {layer: {} for layer in LAYERS}
        self.slots = {layer: {} for layer in LAYERS}

    def _fail(self, message):
        self.poisoned = True
        raise ValueError(message)

    def begin(self, frame, table_slots, writer_slots):
        if self.poisoned or self.active is not None or type(frame) is not Frame or frame.start != self.end:
            self._fail("Stale, nested, missing, or poisoned frame")
        if frame.end > CAPTURE_END:
            self._fail("Independent storage capture ends after the first consumed anchor")
        if (type(table_slots) is not dict or type(writer_slots) is not dict
                or set(table_slots) != set(LAYERS) or set(writer_slots) != set(LAYERS)
                or any(type(i) is not int for i in (*table_slots, *writer_slots))):
            self._fail("All thirty exact layer identities required")
        pending = {}
        for layer in LAYERS:
            required = set(required_positions(layer, frame.start, frame.end))
            table, writer = table_slots[layer], writer_slots[layer]
            if (type(table) is not dict or set(table) != required
                    or any(type(p) is not int for p in table)
                    or type(writer) is not dict or set(writer) != set(range(frame.start, frame.end))
                    or any(type(p) is not int for p in writer)):
                self._fail("Missing required local union/global prefix or writer rows")
            if any(type(s) is not int or s < 0 for s in table.values()):
                self._fail("Invalid independently reconstructed table slot")
            if len(set(table.values())) != len(table):
                self._fail("Required rows alias before queries complete")
            for position in range(required_positions(layer, frame.start, frame.end).start, frame.start):
                if (position not in self.expected[layer]
                        or table[position] != self.slots[layer][position]):
                    self._fail("Retained row absent or remapped")
            old_slots = set(self.slots[layer].values())
            for position, slot in writer.items():
                if type(slot) is not int or table[position] != slot or slot in old_slots:
                    self._fail("Writer differs from independent table or reuses full-context slot")
            pending[layer] = dict(table)
        self.active = {"frame": frame, "table": pending,
                       "pre": {i: set() for i in LAYERS},
                       "written": {i: set() for i in LAYERS},
                       "post": {i: set() for i in LAYERS}, "writer_started": False,
                       "post_started": False}

    def _row(self, layer, position, key, value):
        if self.poisoned or self.active is None:
            self._fail("No active unpoisoned frame")
        try:
            return row_digest(layer, position, key, value)
        except (TypeError, ValueError, struct.error) as exc:
            self._fail(str(exc))

    def retained(self, layer, position, slot, key, value, *, phase):
        observed = self._row(layer, position, key, value)
        active = self.active
        frame = active["frame"]
        if type(phase) is not str or phase not in {"pre", "post"}:
            self._fail("Unknown retained-check phase")
        if (position not in active["table"][layer]
                or type(slot) is not int or slot != active["table"][layer][position]
                or observed != self.expected[layer].get(position)):
            self._fail("Stored processed K/V or physical address differs")
        if phase == "pre" and (position >= frame.start or active["writer_started"]):
            self._fail("Prior union must be read before any writer call")
        if phase == "post" and not active["post_started"]:
            if any(active["written"][i] != set(range(frame.start, frame.end)) for i in LAYERS):
                self._fail("Post-query check before complete all-layer writes")
            active["post_started"] = True
        if position in active[phase][layer]:
            self._fail("Duplicate retained-row evidence")
        active[phase][layer].add(position)

    def processed(self, layer, position, writer_slot, key, value):
        expected = self._row(layer, position, key, value)
        active = self.active
        frame = active["frame"]
        if not active["writer_started"]:
            for i in LAYERS:
                required = required_positions(i, frame.start, frame.end)
                if active["pre"][i] != set(range(required.start, frame.start)):
                    self._fail("A writer ran before complete required-prefix verification")
        if (not frame.start <= position < frame.end or position in active["written"][layer]
                or type(writer_slot) is not int or writer_slot != active["table"][layer][position]):
            self._fail("Duplicate/wrong-position processed input or wrong writer slot")
        active["writer_started"] = True
        self.expected[layer][position] = expected
        self.slots[layer][position] = writer_slot
        active["written"][layer].add(position)

    def finish(self, *, queries_complete):
        if self.poisoned or self.active is None or queries_complete is not True:
            self._fail("No complete synchronized query frontier")
        frame = self.active["frame"]
        for layer in LAYERS:
            if (self.active["written"][layer] != set(range(frame.start, frame.end))
                    or self.active["post"][layer] != set(required_positions(layer, frame.start, frame.end))):
                self._fail("Incomplete all-layer write/required-union verification")
        self.end = frame.end
        self.active = None
        return self.end


def verify_frontier(prompt, heads, emissions, decode_inputs, committed_length):
    """Independently check discarded heads and the emitted/uncached frontier.

    Head: (absolute input position, prediction position, discarded boolean).
    Emission: (head ordinal, token ID). Decode input: (position, token ID).
    Actual private tokens are caller data; reports must retain only digests.
    """
    if type(prompt) is not tuple or len(prompt) != PROMPT:
        raise ValueError("Exact private 2048-token prompt required")
    for token in prompt:
        integer(token, 0, 262143, "prompt token")
    expected = [(p, p + 1, p != PROMPT - 1) for p in range(CHUNK - 1, PROMPT, CHUNK)]
    expected += [(p, p + 1, False) for p in range(PROMPT, PROMPT + OUTPUT - 1)]
    if (type(heads) is not list or len(heads) != len(expected)
            or any(type(h) is not tuple or len(h) != 3 or type(h[0]) is not int
                   or type(h[1]) is not int or type(h[2]) is not bool for h in heads)
            or heads != expected):
        raise ValueError("Actual 7 discarded / 1 final / 255 decode head frontier differs")
    if (type(emissions) is not list or len(emissions) != OUTPUT
            or type(decode_inputs) is not list or len(decode_inputs) != OUTPUT - 1):
        raise ValueError("Exact emitted/input counts required")
    for index, event in enumerate(emissions):
        if type(event) is not tuple or len(event) != 2:
            raise ValueError("Emission must bind an actual head ordinal and token")
        integer(event[0], 7 + index, 7 + index, "emission head")
        integer(event[1], 0, 262143, "emitted token")
    for index, event in enumerate(decode_inputs):
        if (type(event) is not tuple or len(event) != 2 or type(event[0]) is not int
                or type(event[1]) is not int
                or event != (PROMPT + index, emissions[index][1])):
            raise ValueError("Uncached anchor/next-input token or absolute position differs")
    integer(committed_length, CAPACITY - 1, CAPACITY - 1, "committed length")
    return {"checked_heads": len(heads), "emitted_tokens": OUTPUT,
            "decode_inputs": OUTPUT - 1, "committed_length": committed_length,
            "uncached_output_position": CAPACITY - 1,
            "token_ids_encoding": "canonical-compact-json-integer-array-utf8",
            "token_ids_sha256": hashlib.sha256(json.dumps(
                [event[1] for event in emissions], sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()}


def compare_bf16_samples(left, right, *, changed_arithmetic_path):
    """Descriptive deltas only: never infer or fit a numerical tolerance."""
    if type(changed_arithmetic_path) is not bool or type(left) is not bytes or not left:
        raise ValueError("Explicit path classification and nonempty BF16 bytes required")
    a, b = finite_bf16(left, len(left) // 2), finite_bf16(right, len(left) // 2)
    def value(word):
        return struct.unpack("<f", struct.pack("<I", word << 16))[0]
    diffs = [abs(value(x) - value(y)) for x, y in zip(a, b)]
    equal = left == right
    return {"values": len(a), "different_words": sum(x != y for x, y in zip(a, b)),
            "max_abs": max(diffs), "rmse": math.sqrt(math.fsum(d*d for d in diffs) / len(diffs)),
            "exact_sample_agreement": equal,
            "same_path_repeatability_passed": equal and not changed_arithmetic_path,
            "numerical_tolerance": None, "independent_arithmetic_qualified": False,
            "whole_model_quality_qualified": False, "performance_qualified": False}
