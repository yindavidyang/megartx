"""Independent byte oracle for a proposed M1 NVFP4 preparation boundary.

Only stdlib CPU operations are used. This is a *reference-source* semantic ABI,
not a serialization of installed CUDA/CuTe descriptors or a GPU implementation.
Owners below are symbolic, address-free fixture names, with no scratch aliases.
"""

from dataclasses import dataclass, fields
import struct


H, E, TOP_K, F = 2816, 128, 8, 704
REFERENCE_ABI = "flashinfer-8bc3b578-nvfp4-128x4-semantic-v1"


class UnsupportedReference(ValueError):
    pass


class PreparationMismatch(AssertionError):
    pass


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")


def _bytes(value, length, name):
    if type(value) is not bytes or len(value) != length:
        raise ValueError(f"{name} must be exactly {length} immutable bytes")


def _f32_bits(value, name, *, positive=False):
    _bytes(value, 4, name)
    bits = int.from_bytes(value, "little")
    if bits & 0x7F800000 == 0x7F800000:
        raise ValueError(f"{name} must encode finite FP32")
    if positive and (bits & 0x80000000 or bits & 0x7FFFFFFF == 0):
        raise ValueError(f"{name} must encode positive FP32")


def sf_coordinate(row, block, blocks):
    """128x4 coordinate; cross-checked against the committed numeric oracle."""
    _integer(row, "row", 0, 2**31 - 1)
    _integer(blocks, "blocks", 4, 2**31 - 1)
    _integer(block, "block", 0, blocks - 1)
    if blocks % 4:
        raise ValueError("scale block width must be divisible by four")
    tile, lane, quadrant = row // 128, row % 32, row % 128 // 32
    return (tile * (blocks // 4) + block // 4) * 512 + lane * 16 + quadrant * 4 + block % 4


def grouped_sf_base(expert, prefix, k):
    """Reference-source expert base, distinct from the within-expert swizzle."""
    _integer(expert, "expert", 0, E)  # E is the capacity endpoint, never a route ID.
    _integer(prefix, "prefix", 0, TOP_K)
    if type(k) is not int or k not in (H, F):
        raise UnsupportedReference("only H=2816 and F=704 SF geometries are supported")
    padded_rows = ((prefix + 127 * expert + 127) // 128) * 128
    return padded_rows * k // 16  # Both K values already satisfy the 64 alignment.


@dataclass(frozen=True)
class InputRow:
    selected_ids: tuple
    route_weight_bits: bytes
    packed_fp4: bytes
    swizzled_sf: bytes

    def __post_init__(self):
        if type(self.selected_ids) is not tuple or len(self.selected_ids) != TOP_K:
            raise ValueError("selected_ids must be eight immutable route slots")
        for expert in self.selected_ids:
            _integer(expert, "selected expert", 0, E - 1)
        if len(set(self.selected_ids)) != TOP_K:
            raise ValueError("duplicate expert IDs are invalid M1 input")
        _bytes(self.route_weight_bits, TOP_K * 4, "route_weight_bits")
        for slot in range(TOP_K):
            _f32_bits(self.route_weight_bits[slot * 4:slot * 4 + 4], "route weight")
        _bytes(self.packed_fp4, H // 2, "packed_fp4")
        _bytes(self.swizzled_sf, 128 * (H // 16), "swizzled_sf")
        for block in range(H // 16):
            code = self.swizzled_sf[sf_coordinate(0, block, H // 16)]
            # Finite, nonnegative E4M3fn, including the numerical -0 encoding.
            if code > 126 and code != 128:
                raise ValueError("valid SF coordinates must encode finite nonnegative E4M3fn")


@dataclass(frozen=True)
class BufferRef:
    owner: str
    byte_offset: int
    extent: int
    capacity: int

    def __post_init__(self):
        if type(self.owner) is not str or not self.owner:
            raise ValueError("symbolic buffer owner required")
        for key in ("byte_offset", "extent", "capacity"):
            _integer(getattr(self, key), key, 0, 2**63 - 1)
        if self.byte_offset + self.extent > self.capacity:
            raise ValueError("symbolic buffer view exceeds its owner capacity")


@dataclass(frozen=True)
class ScalarBinding:
    ref: BufferRef
    bits: bytes
    provenance: str


@dataclass(frozen=True)
class StageContext:
    stage: str
    swap_ab: bool
    fusion: str
    alpha_bits: bytes
    activation_global_bits: bytes
    weight_global_bits: tuple
    scalar_provenance: str

    def __post_init__(self):
        if self.stage not in ("fc1", "fc2") or type(self.swap_ab) is not bool:
            raise UnsupportedReference("stage and both explicit boolean swap_ab flags required")
        allowed = ("none",) if self.stage == "fc1" else ("none", "finalize")
        if self.fusion not in allowed:
            raise UnsupportedReference("unsupported epilogue; do not infer output layout")
        if type(self.scalar_provenance) is not str or not self.scalar_provenance:
            raise ValueError("scalar provenance must be declared")
        _bytes(self.alpha_bits, E * 4, "alpha_bits")
        _f32_bits(self.activation_global_bits, "activation global", positive=True)
        count = 2 if self.stage == "fc1" else 1
        if type(self.weight_global_bits) is not tuple or len(self.weight_global_bits) != count:
            raise ValueError("FC1 needs separate gate/up globals; FC2 needs down globals")
        for table in (self.alpha_bits,) + self.weight_global_bits:
            _bytes(table, E * 4, "per-expert scalar table")
            for expert in range(E):
                _f32_bits(table[expert * 4:expert * 4 + 4], "expert scalar", positive=True)


@dataclass(frozen=True)
class Descriptor:
    expert: int
    logical_mnk: tuple
    problem_mnk: tuple
    activation: BufferRef
    weight: BufferRef
    activation_sf: BufferRef
    weight_sf: BufferRef
    activation_mk_strides: tuple  # Logical E2M1 element units, never byte strides.
    weight_kn_strides: tuple
    alpha: ScalarBinding
    activation_global: ScalarBinding
    weight_globals: tuple
    output: BufferRef | None
    finalize_map: BufferRef | None
    finalize_weights: BufferRef | None


@dataclass(frozen=True)
class StageReference:
    context: StageContext
    problem_shapes: bytes  # 128 triples of little-endian int64, fixture encoding only.
    active_descriptors: tuple  # Inactive pointer/stride fields have no assumed value.


@dataclass(frozen=True)
class SfWrite:
    ref: BufferRef
    data: bytes


@dataclass(frozen=True)
class Preparation:
    reference_abi: str
    slot_to_sorted: bytes  # int32[8], little-endian fixture encoding
    sorted_to_slot: bytes
    expert_offsets: bytes  # int64[129]
    expanded_aq: bytes
    permuted_weight_bits: bytes
    expanded_sf_writes: tuple  # Only 1,408 valid bytes; padding is preserved.
    fc1: StageReference
    fc2: StageReference


def _stage(context, offsets):
    stage = context.stage
    n, k = (2 * F, H) if stage == "fc1" else (H, F)
    blocks, shapes, descriptors = k // 16, [], []
    payload = n * k // 2
    sf_weight = n * blocks  # N already divisible by 128; K divisible by 64.
    names = ("gate", "up") if stage == "fc1" else ("down",)
    for expert in range(E):
        rank, count = offsets[expert], offsets[expert + 1] - offsets[expert]
        logical = (count, n, k)
        problem = (n, count, k) if context.swap_ab else logical
        shapes.extend(problem)  # Freshly include all zero-token problems on every call.
        if not count:
            continue
        def scalar(name, bits, offset=expert * 4, capacity=E * 4):
            return ScalarBinding(BufferRef(f"{stage}.{name}", offset, 4, capacity),
                                 bits, context.scalar_provenance)
        final = context.fusion == "finalize"
        descriptors.append(Descriptor(
            expert, logical, problem,
            BufferRef(f"{stage}.activation", rank * k // 2, k // 2, TOP_K * k // 2),
            BufferRef(f"{stage}.weight", expert * payload, payload, E * payload),
            BufferRef(f"{stage}.activation_sf", grouped_sf_base(expert, rank, k),
                      128 * blocks, grouped_sf_base(E, TOP_K, k)),
            BufferRef(f"{stage}.weight_sf", expert * sf_weight, sf_weight, E * sf_weight),
            (k, 1), (1, k),
            scalar("alpha", context.alpha_bits[expert * 4:expert * 4 + 4]),
            scalar("activation_global", context.activation_global_bits, 0, 4),
            tuple(scalar(f"{name}_global", table[expert * 4:expert * 4 + 4])
                  for name, table in zip(names, context.weight_global_bits)),
            None if final else BufferRef(f"{stage}.output", rank * n * 2, n * 2, TOP_K * n * 2),
            BufferRef("sorted_to_slot", rank * 4, 4, TOP_K * 4) if final else None,
            BufferRef("permuted_weight_bits", rank * 4, 4, TOP_K * 4) if final else None,
        ))
    return StageReference(context, struct.pack("<" + "q" * len(shapes), *shapes), tuple(descriptors))


def prepare(row, fc1, fc2, *, reference_abi):
    """Return exact maps/bytes and canonical symbolic fields; never use real pointers."""
    if reference_abi != REFERENCE_ABI:
        raise UnsupportedReference("unknown or installed ABI is not this reference-source contract")
    if not isinstance(row, InputRow) or not isinstance(fc1, StageContext) or not isinstance(fc2, StageContext):
        raise ValueError("validated input row and two explicit stage contexts required")
    if fc1.stage != "fc1" or fc2.stage != "fc2":
        raise ValueError("stage contexts must be ordered FC1 then FC2")
    sorted_slots = sorted(range(TOP_K), key=row.selected_ids.__getitem__)
    ranks = [0] * TOP_K
    for rank, slot in enumerate(sorted_slots):
        ranks[slot] = rank
    # Histogram/prefix calculation is independent of the sort/rank construction.
    histogram, offsets = [0] * E, [0]
    for expert in row.selected_ids:
        histogram[expert] += 1
    for count in histogram:
        offsets.append(offsets[-1] + count)
    writes, blocks = [], H // 16
    for slot in sorted_slots:
        expert = row.selected_ids[slot]
        base = grouped_sf_base(expert, offsets[expert], H)
        for block in range(0, blocks, 4):
            coordinate = sf_coordinate(0, block, blocks)
            writes.append(SfWrite(BufferRef("fc1.activation_sf", base + coordinate, 4,
                                           grouped_sf_base(E, TOP_K, H)),
                                  row.swizzled_sf[coordinate:coordinate + 4]))
    return Preparation(
        reference_abi, struct.pack("<8i", *ranks), struct.pack("<8i", *sorted_slots),
        struct.pack("<129q", *offsets), row.packed_fp4 * TOP_K,
        b"".join(row.route_weight_bits[slot * 4:slot * 4 + 4] for slot in sorted_slots),
        tuple(writes), _stage(fc1, offsets), _stage(fc2, offsets),
    )


def materialize_sf(expected, before, *, origin=0):
    """CPU-only replay of valid SF writes, preserving declared padding and guards."""
    if type(expected) is not Preparation or expected.reference_abi != REFERENCE_ABI:
        raise UnsupportedReference("typed reference-source preparation required for SF replay")
    _integer(origin, "origin", 0, 2**63 - 1)
    capacity = grouped_sf_base(E, TOP_K, H)
    if type(before) is not bytes or len(before) < origin + capacity:
        raise ValueError("initial SF storage must contain the full reference extent")
    out = bytearray(before)
    for write in expected.expanded_sf_writes:
        ref = write.ref
        if ref.owner != "fc1.activation_sf" or ref.capacity != capacity or len(write.data) != ref.extent:
            raise ValueError("invalid SF write contract")
        start = origin + ref.byte_offset
        out[start:start + ref.extent] = write.data
    return bytes(out)


def require_equal(expected, observed):
    """Zero-tolerance comparison of this semantic contract, not opaque struct bytes."""
    if type(expected) is not Preparation or type(observed) is not Preparation:
        raise PreparationMismatch("typed semantic preparation records required")
    mismatches = [field.name for field in fields(expected)
                  if getattr(expected, field.name) != getattr(observed, field.name)]
    if mismatches:
        raise PreparationMismatch("preparation differs: " + ", ".join(mismatches))


def require_sf_storage(expected, before, observed, *, origin=0):
    """Check valid bytes plus every untouched padding/guard byte, without mutation."""
    if type(observed) is not bytes or observed != materialize_sf(expected, before, origin=origin):
        raise PreparationMismatch("expanded SF storage or untouched padding/guards differ")
