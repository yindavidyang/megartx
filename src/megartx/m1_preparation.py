"""Fail-closed selection interface for future M1 NVFP4 preparation integration.

No candidate loader, registration mechanism, environment switch, CUDA import or
graph hook exists here. Structural CPU eligibility never establishes native ABI,
compiled kernel correctness, corrected numerical-lane or graph qualification.
"""

from dataclasses import dataclass

from .adapter import BackendUnavailable


@dataclass(frozen=True)
class PreparationRequest:
    tokens: int
    hidden: int
    experts: int
    top_k: int
    intermediate: int
    tp_size: int
    ep_size: int
    quantization: str
    input_sf_present: bool
    swizzled_input_sf: bool
    use_per_expert_act_scale: bool
    min_latency_mode: bool
    lora: bool
    groupwise: bool
    all_to_all: bool
    correction_runner_active: bool
    execution_mode: str
    fc1_swap_ab: bool | None
    fc2_swap_ab: bool | None


@dataclass(frozen=True)
class PreparationDecision:
    backend: str
    cpu_contract_eligible: bool
    reasons: tuple


PENDING = (
    "installed_source_and_binary_abi_unverified",
    "typed_layout_and_consumer_read_masks_missing",
    "workspace_owners_extents_aliases_and_lifetimes_unverified",
    "compiled_candidate_unavailable",
    "gpu_byte_equality_and_sanitizers_pending",
    "corrected_numerical_lane_unqualified",
)


def select_preparation(request, *, opt_in=False, selected_ids=None):
    """Pure pre-mutation decision; always retain incumbent preparation in v1.

Optional IDs are bounded CPU fixtures, never device tensors. Invalid IDs for a
declared M1 route are errors; unsupported shapes do not reinterpret their IDs.
No caller-supplied qualification booleans can unlock the unimplemented backend.
"""
    if not isinstance(request, PreparationRequest) or type(opt_in) is not bool:
        raise ValueError("typed preparation request and boolean opt_in required")
    geometry = (request.tokens, request.hidden, request.experts, request.top_k,
                request.intermediate, request.tp_size, request.ep_size)
    expected = (1, 2816, 128, 8, 704, 1, 1)
    reasons = []
    if any(type(value) is not int for value in geometry) or geometry != expected:
        reasons.append("unsupported_geometry_or_parallelism")
    elif selected_ids is not None:
        if (type(selected_ids) is not tuple or len(selected_ids) != 8
                or any(type(expert) is not int or not 0 <= expert < 128 for expert in selected_ids)
                or len(set(selected_ids)) != 8):
            raise ValueError("declared M1 input requires eight distinct int32 expert IDs in [0,128)")
    flags = {
        "input_sf_present": True, "swizzled_input_sf": True,
        "use_per_expert_act_scale": False, "min_latency_mode": False,
        "lora": False, "groupwise": False, "all_to_all": False,
        "correction_runner_active": True,
    }
    for name, required in flags.items():
        if getattr(request, name) is not required:
            reasons.append("unsupported_" + name)
    if request.quantization != "prequantized_nvfp4_common_row":
        reasons.append("unsupported_quantization_lane")
    if request.execution_mode != "eager":
        reasons.append("corrected_graph_path_unqualified")
    if type(request.fc1_swap_ab) is not bool or type(request.fc2_swap_ab) is not bool:
        reasons.append("unknown_swap_ab")
    eligible = not reasons
    if not opt_in:
        reasons.append("candidate_opt_in_disabled")
    reasons.extend(PENDING)
    return PreparationDecision("incumbent", eligible, tuple(reasons))


def run_candidate(*args, **kwargs):
    """No execution escape hatch, including for fabricated positive receipts."""
    raise BackendUnavailable("M1 preparation candidate is uncompiled and unverified; retain incumbent")
