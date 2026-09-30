"""Offline byte accounting from user-supplied physical tensor metadata.

No checkpoint download, inferred architecture, device allocation, or fit claim.
"""

from .contracts import (ContractError, storage_bytes, tensor_storage_totals,
                        validate, validate_kv_layers)


def tensor_bytes(manifest):
    validate(manifest, "tensors")
    return tensor_storage_totals(manifest["tensors"])


def kv_bytes(layers, cached_tokens, output_tokens):
    """Distinct K/V accounting; a local mask alone does not bound allocation."""
    validate_kv_layers(layers)
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0
           for n in (cached_tokens, output_tokens)):
        raise ContractError("token capacities must be nonnegative integers")
    total = 0
    for layer in layers:
        tokens = cached_tokens + output_tokens
        if layer["storage_policy"] == "bounded_window":
            tokens = min(tokens, layer["window_tokens"])
        if tokens == 0:
            continue
        total += storage_bytes([tokens, layer["kv_heads"], layer["key_head_dim"]], layer["key_dtype"])
        total += storage_bytes([tokens, layer["kv_heads"], layer["value_head_dim"]], layer["value_dtype"])
    return total


def estimate_memory(tensors, budget):
    validate(budget, "memory")
    summary = tensor_bytes(tensors)
    unknown = []
    if not tensors["complete"]:
        unknown.append("tensor manifest coverage")
    if not budget["complete"]:
        unknown.append("memory budget coverage")
    if not budget["kv_layers"]:
        unknown.append("per-layer KV geometry and physical allocation")
    cache = kv_bytes(budget["kv_layers"], budget["cached_context_tokens"], budget["output_capacity_tokens"])
    known = summary["unique_storage_bytes"] + cache
    for component, value in budget["other_bytes"].items():
        if value is None:
            unknown.append(component)
        else:
            known += value
    available = budget["device_total_bytes"]
    if available is None:
        unknown.append("device capacity")
    estimate = None if unknown else known + budget["reserve_bytes"] <= available
    return {**summary, "known_kv_bytes": cache, "known_budget_bytes": known,
            "reserve_bytes": budget["reserve_bytes"], "unknown_components": unknown,
            "estimated_within_capacity": estimate, "measured_fit": "pending",
            "note": "Offline accounting only; peak phases, allocator behavior and load/decode require GPU evidence."}
