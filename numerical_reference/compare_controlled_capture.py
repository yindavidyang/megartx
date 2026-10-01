"""Bounded NumPy-only comparison of private controlled MegaRTX captures.

Run after the owner has stopped its server. All evidence/checkpoint access is
read-only; a new report must be outside every input directory. Passing observed
operator gates never grants native-MMA, natural quality or whole-model gates.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np

import controlled_comparison as comparison
import controlled_reference as reference
import nvfp4_reference as formats


MAX_JSON = 8 << 20
MAX_NPZ = 16 << 20
MAX_TRACE = 64 << 20
REVISION = "a19cfe00be84568a6867111c9a68c9c44fdcffe6"
ROUTE_FIELDS = {"positions", "tokens", "ids", "weight_bits"}
LOGIT_FIELDS = {"input_positions", "input_token_ids", "logits"}
STAGE_FIELDS = {"input_bits", "q1", "sf1", "gate_bits", "up_bits", "activation_bits", "q2", "sf2", "down_bits",
                "gate_alpha_bits", "up_alpha_bits", "down_alpha_bits", "quant1_global_bits", "quant2_global_bits",
                "a1_bits", "a2_bits", "route_weight_bits", "ordinary_bits", "routed_bits", "weighted_bits", "positions", "token_ids"}
KV_FIELDS = {"logical_positions", "key_bits", "value_bits"}
KV_SOURCE_HASHES = {
    "vllm.model_executor.models.gemma4": "16ac0a67dcf5dd695a59edb177f20e1e69d9c3ec45c5883744470c7c06c516a3",
    "vllm.model_executor.layers.attention.attention": "bc897e452f3aea603e353ca42a365d3be8836e9485d78cbf8cc4677baca2fb42",
    "vllm.v1.attention.backends.flashinfer": "8ee541fde43ed92a417b3f01ea1e6fc64af8ab2eb54cbfc28308a49aaca4ed7f",
    "vllm.forward_context": "cc1473477dcc7b762c3901b25cc7bf35c06a2c42f193136fed553abf6fe33f0e",
}


def read_bytes(path, budget):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > budget:
        raise ValueError("Evidence file is missing or exceeds its bounded read size: " + path.name)
    payload = path.read_bytes()
    if len(payload) > budget:
        raise ValueError("Evidence file grew beyond its bounded read size")
    return payload


def read_json(path):
    return json.loads(read_bytes(path, MAX_JSON))


def digest_file(path, budget):
    return hashlib.sha256(read_bytes(path, budget)).hexdigest()


def local_file(directory, name):
    directory = Path(directory).resolve()
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Captured evidence must use a local basename")
    path = directory / name
    if path.is_symlink() or path.resolve().parent != directory:
        raise ValueError("Captured evidence must not escape through a symlink")
    return path


def read_npz(directory, name, sha256, fields):
    """Inspect declared NPY extents before NumPy can allocate a forged shape."""
    path = local_file(directory, name)
    payload = read_bytes(path, MAX_NPZ)
    if sha256 is not None and hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError("Captured NPZ SHA256 differs: " + name)
    import io
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        expected = {field + ".npy" for field in fields}
        if len(members) != len(expected) or {m.filename for m in members} != expected:
            raise ValueError("Captured NPZ has unexpected or duplicate members")
        if sum(member.file_size for member in members) > MAX_NPZ:
            raise ValueError("Expanded NPZ exceeds the bounded array budget")
        for member in members:
            with archive.open(member) as source:
                version = np.lib.format.read_magic(source)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(source, max_header_size=8192)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(source, max_header_size=8192)
                else:
                    raise ValueError("Unsupported captured NPY header version")
                if dtype.hasobject or dtype.fields is not None or dtype.kind not in "iuf" or dtype.itemsize not in {1, 2, 4, 8}:
                    raise ValueError("Captured NPZ must contain plain numeric arrays, never pickle objects")
                if not isinstance(shape, tuple) or len(shape) > 4 or any(type(n) is not int or n < 0 for n in shape):
                    raise ValueError("Invalid captured NPY array shape")
                extent = math.prod(shape) * dtype.itemsize
                if extent > MAX_NPZ or source.tell() + extent != member.file_size:
                    raise ValueError("Declared NPY extent does not match bounded stored payload")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {field: archive[field].copy() for field in fields}


def load_plan(directory):
    directory = Path(directory)
    metadata = read_json(directory / "manifest.json")
    reference._controlled_origin(metadata)
    if metadata["checkpoint_revision"] != REVISION or metadata["config_sha256"] != reference.CONFIG_SHA256:
        raise ValueError("Controlled plan differs from the immutable checkpoint/config")
    arrays = read_npz(directory, "route-table.npz", None, {"tokens", "ids", "weight_bits"})
    plan = {**reference.CONTROLLED_ORIGIN, **arrays,
            "token_sha256": metadata["token_sha256"], "schedule_sha256": metadata["schedule_sha256"]}
    reference.validate_schedule(plan)
    if metadata["canonical_input_tokens"] != 33 or metadata["cached_prompt_tokens"] != 32 or metadata["expected_cached_emitted_token_ids"] != [int(plan["tokens"][32])] * 2:
        raise ValueError("Controlled client prompt/singleton scope differs")
    if metadata["allowed_singleton_not_eos"] is not True:
        raise ValueError("Controlled singleton was not established as non-EOS")
    return plan


def trace_summary(directory, manifest):
    trace = local_file(directory, "controlled-trace.json.gz")
    if digest_file(trace, MAX_TRACE) != manifest["cuda_trace_sha256"]:
        raise ValueError("Owned controlled trace SHA256 differs")
    with gzip.open(trace, "rb") as source:
        payload = source.read(MAX_TRACE + 1)
    if len(payload) > MAX_TRACE:
        raise ValueError("Expanded owned trace exceeds the bounded read size")
    events = json.loads(payload)["traceEvents"]
    if not isinstance(events, list) or len(events) > 150000:
        raise ValueError("Owned trace exceeds the bounded event count")
    complete = [event for event in events if isinstance(event, dict) and event.get("ph") == "X"]
    spans = [event for event in complete if event.get("cat") == "user_annotation"
             and event.get("name") == "megartx::controlled_expert_" + manifest["mode"]]
    outer = [event for event in complete if event.get("cat") == "user_annotation"
             and event.get("name") == "megartx::corrected_expert_" + manifest["mode"]]
    kernels = [event for event in complete if event.get("cat") == "kernel"]
    if len(spans) != 6 or len(outer) != 6 or any("marlin" in event.get("name", "").lower() for event in kernels):
        raise ValueError("Owned trace lacks six distinct controlled correction spans or contains Marlin")
    for span in spans:
        if any(type(span.get(field)) not in {int, float} or not math.isfinite(span[field]) for field in ("ts", "dur")) or span["dur"] <= 0 or "pid" not in span or "tid" not in span:
            raise ValueError("Controlled trace span has invalid thread/time coordinates")
        if sum(parent.get("pid") == span["pid"] and parent.get("tid") == span["tid"] and parent["ts"] <= span["ts"] and span["ts"] + span["dur"] <= parent["ts"] + parent["dur"] for parent in outer) != 1:
            raise ValueError("Dedicated controlled span lacks its unique corrected parent")
    # CUDA kernel time need not overlap CPU enqueue time. Bind actual launch
    # correlation IDs to the CPU thread/span, never count global MoE kernels.
    launch_owners = {}
    for event in complete:
        if event.get("cat") not in {"cuda_runtime", "cuda_driver"}:
            continue
        correlation = event.get("args", {}).get("correlation")
        if type(correlation) is not int or type(event.get("ts")) not in {int, float}:
            continue
        owners = [i for i, span in enumerate(spans) if event.get("pid") == span["pid"] and event.get("tid") == span["tid"] and span["ts"] <= event["ts"] < span["ts"] + span["dur"]]
        if len(owners) > 1 or (owners and correlation in launch_owners):
            raise ValueError("CUDA launch correlation maps ambiguously to controlled spans")
        if owners:
            launch_owners[correlation] = owners[0]
    per_span = [Counter() for _ in spans]
    for event in kernels:
        correlation = event.get("args", {}).get("correlation")
        if type(correlation) is int and correlation in launch_owners:
            per_span[launch_owners[correlation]][event["name"]] += 1
    dense = [sum(count for name, count in counts.items() if "MainloopSm120TmaWarpSpecializedBlockScaled" in name) for counts in per_span]
    gelu = [counts["_gelu_product"] for counts in per_span]
    required_dense = 0 if manifest["mode"] == "paired_reference" else 3
    if dense != [required_dense] * 6 or gelu != [1] * 6:
        raise ValueError("Dedicated controlled spans lack the exact per-expert dense/GELU launches")
    return {"sha256": manifest["cuda_trace_sha256"], "controlled_adapter_spans": len(spans),
            "adapter_gelu_launches": sum(gelu), "dense_sm120_launches": sum(dense),
            "per_span_dense_launches": dense, "per_span_gelu_launches": gelu,
            "cpu_launch_to_controlled_span_correlation_verified": True,
            "kernel_to_gate_up_down_stage_identity_qualified": False}


def check_proof(run, expected_sha):
    proof_path = Path(run) / "activation-proof.json"
    if digest_file(proof_path, MAX_JSON) != expected_sha:
        raise ValueError("Registered integration proof SHA256 differs")
    proof = read_json(proof_path)
    if proof["forced_all_six_executed"] is not True or proof["natural_model_forward_verified"] is not True or len(proof["registered_layers"]) != 30:
        raise ValueError("Registered integration/guarded model dispatch proof is incomplete")
    layer_names = [item["layer_name"] for item in proof["registered_layers"]]
    ordinals = [int(re.search(r"layers\.(\d+)\.", name)[1]) for name in layer_names]
    if sorted(ordinals) != list(range(30)):
        raise ValueError("Registered proof layer identities differ")
    captures = proof["original_tensor_captures"]
    if len(captures) != 54 or len({item["tensor"] for item in captures}) != 54 or any(item["original_bytes_equal"] is not True for item in captures):
        raise ValueError("Original fifty-four tensor capture proof differs")
    return {item["tensor"]: item for item in captures}


def load_case(run, path, mode, plan):
    run = Path(run)
    if path not in {"full", "cached", "chunked"}:
        raise ValueError("Unknown bounded controlled execution path")
    if (run / "QUALIFICATION-INVALIDATED.json").exists():
        raise ValueError("Controlled run was explicitly invalidated")
    directory = run / "controlled" / path
    if (directory / "INVALIDATED.json").exists():
        raise ValueError("Controlled case was explicitly invalidated")
    manifest = read_json(directory / "controlled-manifest.json")
    reference._controlled_origin(manifest)
    if manifest["routing_unchanged"] is not False or manifest["mode"] != mode or manifest["path"] != path or manifest["input_tokens"] != 33:
        raise ValueError("Controlled case mode/path/scope differs")
    for field in ("token_sha256", "schedule_sha256"):
        if manifest[field] != plan[field]:
            raise ValueError("Controlled case prefix/table identity differs")
    originals = check_proof(run, manifest["execution_proof_sha256"])
    trace = trace_summary(directory, manifest)
    calls = manifest["forward_calls"]
    if type(calls) is not int or not 1 <= calls <= 4 or len(manifest["routes"]) != 30 * calls:
        raise ValueError("Controlled route forward count exceeds or misses the bounded scope")
    files, by_layer, by_forward = set(), {i: [] for i in range(30)}, {}
    for record in manifest["routes"]:
        layer, forward = record["layer"], record["forward"]
        if type(layer) is not int or layer not in by_layer or type(forward) is not int or not 0 <= forward < calls:
            raise ValueError("Unknown controlled route layer/forward identity")
        if record["file"] in files or (layer, forward) in by_forward:
            raise ValueError("Duplicate controlled route file/layer/forward")
        files.add(record["file"])
        arrays = read_npz(directory, record["file"], record["sha256"], ROUTE_FIELDS)
        n = len(arrays["positions"])
        reference._array(arrays["positions"], np.int64, (n,), "route positions")
        reference._array(arrays["tokens"], np.int64, (n,), "route tokens")
        reference._array(arrays["ids"], np.int32, (n, 8), "route IDs")
        reference._array(arrays["weight_bits"], np.uint32, (n, 8), "route weight bits")
        if not 1 <= n <= 33:
            raise ValueError("Controlled route row count exceeds the single sequence")
        by_layer[layer].append((forward, arrays))
        by_forward[layer, forward] = arrays
    merged = []
    for layer in range(30):
        records = [arrays for _, arrays in sorted(by_layer[layer])]
        if len(records) != calls:
            raise ValueError("A controlled layer is missing a model forward")
        merged.append({key: np.concatenate([item[key] for item in records]) for key in ROUTE_FIELDS})
        for forward in range(calls):
            if not np.array_equal(by_forward[layer, forward]["positions"], by_forward[0, forward]["positions"]) or not np.array_equal(by_forward[layer, forward]["tokens"], by_forward[0, forward]["tokens"]):
                raise ValueError("Controlled layers did not process the same actual forward rows")
    expected_frames = {"full": [range(33)], "cached": [range(32), range(32, 33)],
                       "chunked": [range(16), range(16, 32), range(32, 33)]}[path]
    if calls != len(expected_frames) or any(not np.array_equal(by_forward[0, forward]["positions"], list(frame)) for forward, frame in enumerate(expected_frames)):
        raise ValueError("Actual forward batches differ from the declared full/cached/chunked path")
    dispatch = {**manifest, "layers": np.arange(30, dtype=np.int32), "positions": merged[0]["positions"],
                "tokens": merged[0]["tokens"], "ids": np.stack([item["ids"] for item in merged]),
                "weight_bits": np.stack([item["weight_bits"] for item in merged])}
    guarded = reference.validate_dispatch(plan, dispatch)
    expected = {(layer, expert, reference.TARGET_POSITIONS[layer, expert]) for layer, expert in reference.TARGETS}
    stages = {}
    for record in manifest["executed_interventions"]:
        key = record["layer"], record["expert"], record["position"]
        if key not in expected or key in stages or record["file"] in files:
            raise ValueError("Stage capture differs from the six distinct planned interventions")
        files.add(record["file"])
        arrays = read_npz(directory, record["file"], record["stage_capture_sha256"], STAGE_FIELDS)
        _validate_stage(arrays, key, plan, record)
        stages[key] = arrays
    if set(stages) != expected:
        raise ValueError("Controlled stage captures missed an executed intervention")
    counters = validate_counters(manifest)
    return {"run": run, "directory": directory, "manifest": manifest, "stages": stages,
            "original_captures": originals, "dispatch_check": guarded, "trace": trace,
            "counter_check": counters}


def validate_counters(manifest):
    for field in ("natural_hits", "controlled_hits"):
        counters = manifest[field]
        if not isinstance(counters, dict) or set(counters) != {str(i) for i in range(30)} or any(not isinstance(value, dict) for value in counters.values()):
            raise ValueError("Controlled counters must retain all thirty layer dictionaries")
        for values in counters.values():
            if any(not isinstance(key, str) or not key.isdecimal() or type(value) is not int or value < 0 for key, value in values.items()):
                raise ValueError("Controlled counters contain invalid expert/count fields")
    if any(value != 0 for counters in manifest["natural_hits"].values() for value in counters.values()):
        raise ValueError("Controlled run counter was mislabeled as natural coverage")
    expected = {}
    for record in manifest["executed_interventions"]:
        before, after = record["controlled_count_before"], record["controlled_count_after"]
        if type(before) is not int or type(after) is not int or before < 0 or after != before + 1:
            raise ValueError("Controlled completion counter did not increment for the actual row")
        expected.setdefault(str(record["layer"]), {})[str(record["expert"])] = after
    if manifest["controlled_hits"] != {str(i): expected.get(str(i), {}) for i in range(30)}:
        raise ValueError("Controlled per-layer counters disagree with the six completed events")
    return {"six_reported_counter_increments_bound_to_stages": True,
            "natural_positive_counts": 0, "counter_only_execution_qualified": False}


def _validate_stage(stages, key, plan, record):
    layer, expert, position = key
    reference._array(stages["positions"], np.int64, (1,), "actual expert position")
    reference._array(stages["token_ids"], np.int64, (1,), "actual expert token")
    if int(stages["positions"][0]) != position or int(stages["token_ids"][0]) != int(plan["tokens"][position]):
        raise ValueError("Actual saved expert position/token differs")
    for name, shape in (("input_bits", (1, 2816)), ("gate_bits", (1, 704)), ("up_bits", (1, 704)),
                        ("activation_bits", (1, 704)), ("down_bits", (1, 2816)),
                        ("ordinary_bits", (1, 2816)), ("routed_bits", (1, 2816))):
        reference.bf16_values(reference._array(stages[name], np.uint16, shape, name))
    for name, shape in (("q1", (1, 1408)), ("q2", (1, 352))):
        reference._array(stages[name], np.uint8, shape, name)
    for name in ("gate_alpha_bits", "up_alpha_bits", "down_alpha_bits", "quant1_global_bits", "quant2_global_bits", "a1_bits", "a2_bits", "route_weight_bits"):
        reference._f32_scalar(stages[name], name)
    if int(stages["route_weight_bits"][0]) != record["weight_bits"]:
        raise ValueError("Saved actual expert route weight differs from its event")
    down = reference.bf16_values(stages["down_bits"])
    if int(np.count_nonzero(down)) != record["nonzero_output_elements"]:
        raise ValueError("Recorded positive output count differs from saved actual down values")
    reference._array(stages["weighted_bits"], np.uint32, (1, 2816), "F32 weighted contribution")


def compare_weighting(stages):
    weight = reference._f32_scalar(stages["route_weight_bits"], "route weight")
    weighted = np.multiply(reference.bf16_values(stages["down_bits"]), weight, dtype=np.float32)
    expected_bits = weighted.view(np.uint32)
    add = np.add(reference.bf16_values(stages["ordinary_bits"]), weighted, dtype=np.float32)
    combined = reference.bf16_bits(add)
    return {"weighted_f32_bits_equal": bool(np.array_equal(expected_bits, stages["weighted_bits"])),
            "combined_bf16": reference._compare_bits(stages["routed_bits"], combined),
            "scope": "one declared corrected expert per row; selected-runtime BF16 ordinary/F32-add/BF16-cast"}


def logit_value_difference(left, right):
    if left.dtype != np.float32 or right.dtype != np.float32 or left.shape != (262144,) or right.shape != left.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Raw logit comparison requires two finite full-vocabulary F32 rows")
    difference = left.astype(np.float64) - right.astype(np.float64)
    return {"raw_f32_bits_equal": bool(np.array_equal(left.view(np.uint32), right.view(np.uint32))),
            "differing_raw_values": int(np.count_nonzero(left.view(np.uint32) != right.view(np.uint32))),
            "max_absolute_difference": float(np.abs(difference).max(initial=0)),
            "rmse": float(np.sqrt(np.mean(difference * difference))),
            "candidate_top1": int(left.argmax()), "reference_top1": int(right.argmax()),
            "numerical_acceptance_tolerance": None, "quality_gate_passed": False}


def load_logits(case, plan):
    lines = read_bytes(case["directory"] / "logits-records.jsonl", MAX_JSON).decode().splitlines()
    if not 1 <= len(lines) <= 8:
        raise ValueError("Controlled raw-logit capture count exceeds or misses the bound")
    rows, files, records = {}, set(), []
    for line in lines:
        record = json.loads(line)
        reference._controlled_origin(record)
        if record["mode"] != case["manifest"]["mode"] or record["path"] != case["manifest"]["path"] or record["row_identity_verified"] is not True:
            raise ValueError("Controlled raw-logit mode/path/identity differs")
        if record["row_correspondence"] not in {"verified_storage_view", "unique_full_hidden_bit_match"}:
            raise ValueError("Controlled raw logits lack verified full hidden-row correspondence")
        if any(record[field] != plan[field] for field in ("token_sha256", "schedule_sha256")) or record["file"] in files:
            raise ValueError("Controlled raw-logit prefix/table/file identity differs")
        files.add(record["file"])
        arrays = read_npz(case["directory"], record["file"], record["sha256"], LOGIT_FIELDS)
        positions, tokens, logits = arrays["input_positions"], arrays["input_token_ids"], arrays["logits"]
        n = len(positions)
        reference._array(positions, np.int64, (n,), "raw logit positions")
        reference._array(tokens, np.int64, (n,), "raw logit tokens")
        reference._array(logits, np.float32, (n, 262144), "raw full-vocabulary logits")
        if not 1 <= n <= 2 or type(record["source_rows"]) is not int or record["source_rows"] < n or not np.isfinite(logits).all():
            raise ValueError("Controlled raw-logit source-row/finiteness scope differs")
        for p, token, values in zip(positions, tokens, logits):
            p = int(p)
            if p not in (31, 32) or int(token) != int(plan["tokens"][p]):
                raise ValueError("Controlled raw-logit row has wrong token/position")
            key = p, record["source_rows"]
            if key in rows and not np.array_equal(rows[key].view(np.uint32), values.view(np.uint32)):
                raise ValueError("Repeated full-logit snapshot disagrees at the same position and head batching")
            rows[key] = values
        records.append({"file": record["file"], "sha256": record["sha256"],
                        "source_rows": record["source_rows"], "input_positions": positions.tolist(),
                        "row_correspondence": record["row_correspondence"]})
    if {p for p, _ in rows} != {31, 32}:
        raise ValueError("Controlled raw logits miss required prefill/handoff positions")
    roles = {"full": {31: 33, 32: 33}, "cached": {31: 32, 32: 1},
             "chunked": {31: 16, 32: 1}}[case["manifest"]["path"]]
    if any((p, size) not in rows for p, size in roles.items()):
        raise ValueError("Raw logits lack the declared prompt/handoff head batch sizes")
    primary = {**reference.CONTROLLED_ORIGIN, "token_sha256": plan["token_sha256"], "schedule_sha256": plan["schedule_sha256"],
               "row_identity_verified": True, "input_positions": np.asarray([31, 32], dtype=np.int64),
               "input_token_ids": plan["tokens"][[31, 32]], "logits": np.stack([rows[p, roles[p]] for p in (31, 32)])}
    alternatives = []
    for position in (31, 32):
        sizes = sorted(size for p, size in rows if p == position)
        for size in sizes:
            if size != roles[position]:
                alternatives.append({"input_position": position, "primary_source_rows": roles[position],
                                     "other_source_rows": size,
                                     "comparison": logit_value_difference(rows[position, size], rows[position, roles[position]]),
                                     "scope": "Same hidden position; different LM-head batch size is an explicit confound"})
    return {"primary": primary, "rows": rows, "primary_source_rows": roles,
            "records": records, "within_position_different_head_batches": alternatives}


def compare_logit_variants(candidate, oracle, plan):
    if set(candidate["rows"]) != set(oracle["rows"]) or candidate["primary_source_rows"] != oracle["primary_source_rows"]:
        raise ValueError("Paired modes have different raw-head call roles/batch sizes")
    return {"primary_prompt_handoff": reference.compare_logits(candidate["primary"], oracle["primary"], plan),
            "all_same_head_shape_variants": [{"input_position": p, "source_rows": size,
                **logit_value_difference(candidate["rows"][p, size], oracle["rows"][p, size])} for p, size in sorted(candidate["rows"])],
            "same_head_shapes_required": True, "quality_gate_passed": False}


def load_cache(case, plan, nominal):
    directory = case["directory"]
    contract = reference.validate_cache_contract(read_json(directory / "cache-contract.json"))
    provenance = contract["provenance"]
    source_sha = hashlib.sha256(json.dumps(KV_SOURCE_HASHES, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if contract["checkpoint_revision"] != REVISION or contract["layers"] != nominal["layers"] or provenance != {
        "scope": "installed_runtime_confirmed", "config_sha256": reference.CONFIG_SHA256,
        "source_sha256": source_sha, "kv_owner_layout_confirmed": True}:
        raise ValueError("Captured K/V descriptor differs from frozen config/source provenance")
    binding = read_json(directory / "kv-binding.json")
    reference._controlled_origin(binding)
    if binding["source_sha256"] != KV_SOURCE_HASHES or binding["forward_calls"] != case["manifest"]["forward_calls"] or binding["independent_cache_correctness_qualified"] is not False or binding["scheduler_block_table_independently_reconstructed"] is not False:
        raise ValueError("K/V binding has changed source/forward or excessive qualification")
    owners = binding["layers"]
    if len(owners) != 30 or {item["layer"] for item in owners} != set(range(30)):
        raise ValueError("K/V binding lacks thirty distinct registered owners")
    owners = {item["layer"]: item for item in owners}
    records = read_json(directory / "kv-records.json")
    if len(records) != 30 or {item["layer"] for item in records} != set(range(30)):
        raise ValueError("K/V snapshots lack thirty distinct logical layers")
    snapshots, files = [], set()
    for record in sorted(records, key=lambda item: item["layer"]):
        reference._controlled_origin(record)
        layer = record["layer"]
        descriptor, owner = contract["layers"][layer], owners[layer]
        if record["mode"] != case["manifest"]["mode"] or any(record[field] != plan[field] for field in ("token_sha256", "schedule_sha256")) or record["cache_contract_sha256"] != contract["sha256"]:
            raise ValueError("K/V record differs from the case mode/prefix/table/descriptor")
        if owner["registry_owner_verified"] is not True or owner["metadata_writer_slots_equal"] is not True or owner["dtype"] != "bf16" or owner["writer"] != "FlashInferImpl.do_kv_cache_update" or not re.search(rf"(?:^|\.)layers\.{layer}\.self_attn\.attn$", owner["layer_name"]):
            raise ValueError("K/V owner/writer identity assertion differs")
        shape, strides = owner["cache_view_shape"], owner["cache_view_strides"]
        if len(shape) != 4 or len(strides) != 4 or any(type(value) is not int or value < 1 for value in shape + strides) or shape[1] != descriptor["kv_heads"] or shape[3] != 2 * descriptor["head_dim"] or strides[-1] != 1 or not isinstance(owner["resolved_layout"], str) or not 1 <= len(owner["resolved_layout"]) <= 32:
            raise ValueError("K/V raw owner shape/strides/layout differs from its logical descriptor")
        slots = record["writer_slots"]
        if not isinstance(slots, list) or len(slots) != 2 or len(set(slots)) != 2 or any(type(slot) is not int or not 0 <= slot < shape[0] * shape[2] for slot in slots) or record["same_cache_owner_across_forwards"] is not True or record["logical_mapping_scope"] != "actual per-layer writer slot maps within one fresh short request":
            raise ValueError("K/V selected writer slots lack the bounded actual association")
        if record["file"] in files:
            raise ValueError("K/V file is reused across logical owners")
        files.add(record["file"])
        arrays = read_npz(directory, record["file"], record["sha256"], KV_FIELDS)
        snapshot = {**record, **arrays}
        # The existing independent logical checker enforces positions31/32,
        # finite BF16 payloads and mask geometry. Self comparison is validation
        # here and is not reported as cross-path numerical evidence.
        reference.compare_cache(snapshot, snapshot, plan, cache_contract=contract)
        snapshots.append(snapshot)
    return {"contract": contract, "snapshots": snapshots,
            "summary": {"logical_layers": 30, "logical_positions": [31, 32],
                "raw_source_hashes_checked": KV_SOURCE_HASHES,
                "recorded_physical_layouts": sorted({item["resolved_layout"] for item in owners.values()}),
                "logical_view_layout": "BHNC; physical allocation enum/strides remain distinct",
                "producer_registered_owner_writer_slot_assertions_bound": True,
                "installed_owner_layout_independently_verified": False,
                "independent_cache_correctness_qualified": False}}


def selected_runtime_globals(reader, layer):
    values, source_hashes = {"a1": [], "a2": []}, []
    for expert in range(128):
        for label, target in (("gate", "a1"), ("up", "a1"), ("down", "a2")):
            key = f"model.language_model.layers.{layer}.experts.{expert}.{label}_proj.input_scale"
            value, sha = reader.tensor(key, max_bytes=4)
            if value.dtype != np.float32 or value.shape != () or not np.isfinite(value) or value <= 0:
                raise ValueError("Original calibration is not a positive finite F32 scalar")
            values[target].append(value.item())
            source_hashes.append({"tensor": key, "sha256": sha})
    return {name: reference._f32_bits(max(samples)) for name, samples in values.items()}, hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()


def check_original_hashes(cases, originals, hashes, layer, expert):
    for label in ("gate", "up", "down"):
        for suffix, hash_key, byte_count in (("weight", "weight_sha256", originals[label][0].nbytes),
                                              ("weight_scale", "weight_scale_sha256", originals[label][1].nbytes),
                                              ("weight_scale_2", "weight_scale_2_sha256", 4)):
            key = f"model.language_model.layers.{layer}.experts.{expert}.{label}_proj.{suffix}"
            for case in cases:
                item = case["original_captures"][key]
                if item["sha256"] != hashes[label][hash_key] or item["bytes"] != byte_count:
                    raise ValueError("Loaded capture and independently read immutable tensor differ: " + key)


def compare_runs(plan_directory, native_run, paired_run, negative_run, checkpoint, paths):
    if (paired_run is None) != (negative_run is None):
        raise ValueError("Paired reference and single-alpha negative control must be supplied together")
    paired_complete = paired_run is not None
    plan = load_plan(plan_directory)
    checkpoint = Path(checkpoint)
    if digest_file(checkpoint / "config.json", MAX_JSON) != reference.CONFIG_SHA256:
        raise ValueError("Independent checkpoint directory has the wrong immutable config")
    reader = formats.CheckpointReader(checkpoint)
    nominal = reference.nominal_cache_contract(checkpoint / "config.json")
    calibration = {}
    report = {**reference.CONTROLLED_ORIGIN, "qualification": "Controlled captured-operand/runtime comparison; never natural quality",
              "checkpoint_revision": REVISION, "token_sha256": plan["token_sha256"], "schedule_sha256": plan["schedule_sha256"],
              "numpy_version": np.__version__, "paths": [], "quality_gate_passed": False,
              "paired_reference_and_negative_matrix_present": paired_complete,
              "native_mma_qualified": False, "whole_model_or_frozen_quantizer_qualified": False}
    if not 1 <= len(paths) <= 3 or len(set(paths)) != len(paths):
        raise ValueError("Controlled comparison paths are duplicate or over the declared bound")
    snapshots, caches = {}, {}
    runs = [(native_run, "native")]
    if paired_complete:
        runs.extend(((paired_run, "paired_reference"), (negative_run, "gate_only_negative_control")))
    for path in paths:
        cases = [load_case(run, path, mode, plan) for run, mode in runs]
        path_report = {"path": path, "dispatch": [case["dispatch_check"] for case in cases],
                       "traces": [case["trace"] for case in cases],
                       "controlled_counters": [case["counter_check"] for case in cases], "experts": []}
        gates = True
        for layer, expert in reference.TARGETS:
            key = layer, expert, reference.TARGET_POSITIONS[layer, expert]
            originals, hashes = reference.original_expert(reader, layer, expert)
            check_original_hashes(cases, originals, hashes, layer, expert)
            if layer not in calibration:
                calibration[layer] = selected_runtime_globals(reader, layer)
            expected_globals, calibration_hash = calibration[layer]
            stages = [case["stages"][key] for case in cases]
            for stage in stages:
                if not np.array_equal(stage["a1_bits"], expected_globals["a1"]) or not np.array_equal(stage["a2_bits"], expected_globals["a2"]):
                    raise ValueError("Captured A1/A2 differs from independent original layer maxima")
            replay = [reference.replay_expert(stage, originals, a1_bits=stage["a1_bits"], a2_bits=stage["a2_bits"], mode=case["manifest"]["mode"])
                      for stage, case in zip(stages, cases)]
            weighting = [compare_weighting(stage) for stage in stages]
            gate = all(
                item[label]["raw_bf16_bits_equal"] for item in replay for label in ("gate", "up", "down")) and all(
                item["weighted_f32_bits_equal"] and item["combined_bf16"]["raw_bf16_bits_equal"] for item in weighting)
            expert_report = {"layer": layer, "expert": expert, "position": key[2],
                             "original_projection_hashes": hashes, "calibration_source_digest": calibration_hash,
                             "independent_replays": replay, "weighting": weighting}
            if paired_complete:
                pair = comparison.compare_positive_stages(stages[0], stages[1])
                gate &= pair["strict_observed_stage_gate_pass"]
                expert_report["positive_paired_stages"] = pair
            expert_report["strict_observed_operator_gate_pass"] = gate
            gates &= gate
            path_report["experts"].append(expert_report)
            if paired_complete and key == (0, 42, 31):
                negative = comparison.compare_negative_common_input(*stages, originals, a1_bits=stages[0]["a1_bits"])
                path_report["negative_common_input"] = negative
                gates &= negative["negative_control_effect_observed"]
        path_report["strict_observed_operator_gates_pass"] = gates
        if gates:
            logit_rows = [load_logits(case, plan) for case in cases]
            logical_cache = [load_cache(case, plan, nominal) for case in cases]
            path_report["raw_logit_capture_roles"] = [{"records": item["records"],
                "primary_source_rows": item["primary_source_rows"],
                "within_position_different_head_batches": item["within_position_different_head_batches"]} for item in logit_rows]
            path_report["logical_cache_structure"] = [item["summary"] for item in logical_cache]
            if paired_complete:
                path_report["positive_logits"] = compare_logit_variants(logit_rows[0], logit_rows[1], plan)
                path_report["negative_vs_native_logits"] = compare_logit_variants(logit_rows[2], logit_rows[0], plan)
                path_report["positive_logical_cache"] = reference.compare_cache_collection(
                    logical_cache[0]["snapshots"], logical_cache[1]["snapshots"], plan, cache_contract=logical_cache[0]["contract"])
                path_report["negative_vs_native_logical_cache"] = reference.compare_cache_collection(
                    logical_cache[2]["snapshots"], logical_cache[0]["snapshots"], plan, cache_contract=logical_cache[0]["contract"])
            snapshots[path] = logit_rows
            caches[path] = logical_cache
        else:
            path_report["downstream_comparison_blocker"] = "Strict observed operator/stage or negative-control gate failed; logits/cache promotion withheld"
        report["paths"].append(path_report)
    report["all_strict_observed_operator_gates_pass"] = all(item["strict_observed_operator_gates_pass"] for item in report["paths"])
    report["within_mode_path_comparisons"] = {}
    if "full" in snapshots:
        for path in ("cached", "chunked"):
            if path in snapshots:
                mode_reports = {}
                for index, (_, mode) in enumerate(runs):
                    candidate, oracle = snapshots[path][index], snapshots["full"][index]
                    item = {"prompt_handoff_logits": reference.compare_logits(candidate["primary"], oracle["primary"], plan),
                            "candidate_source_rows": candidate["primary_source_rows"],
                            "reference_source_rows": oracle["primary_source_rows"],
                            "head_batch_shape_confound_present": candidate["primary_source_rows"] != oracle["primary_source_rows"],
                            "logical_cache": reference.compare_cache_collection(caches[path][index]["snapshots"], caches["full"][index]["snapshots"], plan, cache_contract=caches[path][index]["contract"])}
                    if (32, 1) in candidate["rows"] and (32, 1) in oracle["rows"]:
                        item["same_single_row_sampler_position32"] = logit_value_difference(candidate["rows"][32, 1], oracle["rows"][32, 1])
                    mode_reports[mode] = item
                report["within_mode_path_comparisons"][path] = mode_reports
    report["remaining_scope"] = ["Shared native activation/CUDA quantizer/untouched operators are not independent model semantics",
                                 "MMA accumulation contract remains conditional; exact observed fixtures are bounded",
                                 "Trace correlation binds each controlled CPU span; individual gate/up/down identity remains separate",
                                 "KV writer/source/owner bindings are producer assertions checked against raw files, not an independent scheduler/attention oracle",
                                 "Different LM-head batch sizes remain explicit; duplicates are retained by position and actual head shape",
                                 "Artificial routes/singleton outputs are not natural quality, held-out evidence or a corrected timing baseline"]
    if not paired_complete:
        report["remaining_scope"].insert(0, "Native-only stage gate: paired reference and single-alpha negative control are still required")
    return report


def report_output_path(destination, inputs):
    destination = Path(destination).resolve()
    if any(destination.is_relative_to(Path(path).resolve()) for path in inputs):
        raise ValueError("Report output must be outside every evidence/checkpoint input directory")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--paired", type=Path)
    parser.add_argument("--negative", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--paths", choices=("full", "cached", "chunked"), nargs="+", default=["full"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.paired is None) != (args.negative is None):
        parser.error("--paired and --negative must be supplied together")
    destination = report_output_path(args.output, (path for path in (args.plan, args.native, args.paired, args.negative, args.checkpoint) if path is not None))
    report = compare_runs(args.plan, args.native, args.paired, args.negative, args.checkpoint, args.paths)
    payload = json.dumps(report, indent=2, allow_nan=False)
    with destination.open("x", encoding="utf-8") as output:
        output.write(payload)
    print(json.dumps({"all_strict_observed_operator_gates_pass": report["all_strict_observed_operator_gates_pass"],
                      "quality_gate_passed": False, "paths": args.paths}))
    return 0 if report["all_strict_observed_operator_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
