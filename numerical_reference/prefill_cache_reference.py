"""Independent CPU index/mask/cache oracle for prefill preparation.

Tiny synthetic vectors and tagged processed K/V test semantic ownership only.
No GPU, BF16 arithmetic, model logits, native dispatch, or quality is qualified.
"""

import math


GLOBAL_LAYERS = (5, 11, 17, 23, 29)


def _natural(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(label + " must be a nonnegative integer")


def visible_positions(query_position, sliding):
    _natural(query_position, "absolute query position")
    if type(sliding) is not bool:
        raise ValueError("Explicit local/global visibility required")
    start = max(0, query_position - 1023) if sliding else 0
    return range(start, query_position + 1)


def chunk_required_positions(start, end, sliding):
    """Union needed until every row finishes, before any window recycling."""
    _natural(start, "chunk start")
    _natural(end, "chunk end")
    if end <= start:
        raise ValueError("Nonempty chunk required")
    first = visible_positions(start, sliding).start
    return range(first, end)


def page_address(position, page_table, page_size):
    _natural(position, "position")
    if type(page_size) is not int or page_size <= 0:
        raise ValueError("Positive page size required")
    if type(page_table) is not list or not page_table:
        raise ValueError("Page table required")
    for page in page_table:
        _natural(page, "physical page ID")
    if len(set(page_table)) != len(page_table):
        raise ValueError("Aliased physical pages unsupported in committed fixture")
    logical_page, offset = divmod(position, page_size)
    if logical_page >= len(page_table):
        raise ValueError("Position exceeds page capacity")
    return page_table[logical_page], offset


def rotary_pairs(sliding):
    """NeoX active coordinate pairs; inactive global pairs are identity."""
    if type(sliding) is not bool:
        raise ValueError("Explicit layer type required")
    width, active = (256, 128) if sliding else (512, 64)
    return [(i, i + width // 2) for i in range(active)]


def rotary_angles(position, sliding):
    _natural(position, "absolute RoPE position")
    pairs = rotary_pairs(sliding)
    width, theta = (256, 10000) if sliding else (512, 1000000)
    return [position * theta ** (-2 * i / width) for i in range(len(pairs))]


def ideal_attention(query, position, cache, sliding):
    """Unit-scale high-precision Python toy attention, no native cast model."""
    positions = list(visible_positions(position, sliding))
    if any(p not in cache for p in positions):
        raise ValueError("A query's required prior key/value was recycled or absent")
    if not query or any(not math.isfinite(x) for x in query):
        raise ValueError("Finite nonempty query required")
    entries = [cache[p] for p in positions]
    if any(len(k) != len(query) or len(v) != len(entries[0][1]) or not v
           or any(not math.isfinite(x) for x in (*k, *v)) for k, v in entries):
        raise ValueError("Finite matching K/V vectors required")
    scores = [math.fsum(a * b for a, b in zip(query, k)) for k, _ in entries]
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    denominator = math.fsum(weights)
    return tuple(math.fsum(w * entry[1][d] for w, entry in zip(weights, entries)) / denominator
                 for d in range(len(entries[0][1])))


def tagged_handoff(prompt_tokens, output_reserve=256):
    """Processed K/V tags, not model tensors; final cache semantics only."""
    if type(prompt_tokens) is not int or not 1 <= prompt_tokens <= 32768:
        raise ValueError("Bounded prompt required")
    if type(output_reserve) is not int or output_reserve != 256:
        raise ValueError("Fixture requires separately reserved 256 output positions")
    layers = []
    for layer in range(30):
        sliding = layer not in GLOBAL_LAYERS
        positions = visible_positions(prompt_tokens - 1, sliding)
        layers.append({"layer": layer, "kind": "local" if sliding else "global",
                       "entries": {p: ((layer, p, "processed_k"), (layer, p, "processed_v"))
                                   for p in positions}})
    return {"committed_length": prompt_tokens, "next_position": prompt_tokens,
            "capacity": prompt_tokens + output_reserve, "kv_dtype": "bfloat16",
            "complete": True, "poisoned": False, "layers": layers}


def require_handoff(cache, prompt_tokens):
    """Reject partial/stale/aliased state before a modeled continuation."""
    if (type(prompt_tokens) is not int or not 1 <= prompt_tokens <= 32768
            or cache.get("complete") is not True or cache.get("poisoned") is not False
            or type(cache.get("committed_length")) is not int
            or cache["committed_length"] != prompt_tokens
            or type(cache.get("next_position")) is not int
            or cache["next_position"] != prompt_tokens
            or type(cache.get("capacity")) is not int
            or cache["capacity"] < prompt_tokens + 256
            or cache.get("kv_dtype") != "bfloat16"):
        raise ValueError("Invalid committed length/absolute position/capacity/dtype/state")
    layers = cache.get("layers")
    if type(layers) is not list or len(layers) != 30:
        raise ValueError("Complete all-layer handoff required")
    for index, layer in enumerate(layers):
        local = index not in GLOBAL_LAYERS
        if type(layer.get("layer")) is not int or layer["layer"] != index or layer.get("kind") != ("local" if local else "global"):
            raise ValueError("Layer identity/type changed")
        entries = layer.get("entries")
        expected = set(visible_positions(prompt_tokens - 1, local))
        if type(entries) is not dict or any(type(p) is not int for p in entries) or set(entries) != expected:
            raise ValueError("Committed local/global cache coverage changed")
        for p, (k, v) in entries.items():
            if k != (index, p, "processed_k") or v != (index, p, "processed_v"):
                raise ValueError("Processed K/V identity or absolute position changed")
    return True
