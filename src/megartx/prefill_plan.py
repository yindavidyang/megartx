"""Read-only CPU intake and accounting for a future prefill profiler.

This module has no device runner, plugin registration, or execution opt-in.
Validation proves plan consistency, never experiment authorization or GPU fit.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


BASE_COMMIT = "09341f2ad3f6e4b03e8f264b9e3c59fe5ddb52ed"
CHECKPOINT_REVISION = "a19cfe00be84568a6867111c9a68c9c44fdcffe6"
CONTROLS = {
    "lane": "original_weights_runtime_layermax_w4a4",
    "compute_dtype": "bfloat16", "kv_dtype": "bfloat16",
    "attention_backend": "FLASHINFER", "moe_backend": "FLASHINFER_CUTLASS",
    "use_fused_finalize": False, "enforce_eager": True, "text_only": True,
    "concurrency": 1, "dynamic_batching": False, "tp": 1, "ep": 1,
    "prefix_cache": False, "speculation": False, "lora": False,
    "routing": "natural_unchanged", "head_policy": "last_prompt_row",
    "output_capacity_tokens": 256,
    "allow_tf32": False, "allow_fp16_reduced_precision_reduction": True,
    "allow_bf16_reduced_precision_reduction": True,
    "allow_bf16_reduced_precision_reduction_split_k": True,
}
FREEZE_KEYS = {
    "gpu_scope_approval_reference", "sole_owner_handoff_reference",
    "decode_control_commit", "target_environment_manifest_sha256",
    "tokenizer_template_manifest_sha256", "prompt_set_manifest_sha256",
    "oracle_contract_sha256", "g0_acceptance_reference", "g1_acceptance_reference",
    "thresholds_reference", "sampling_policy", "eos_policy",
    "resource_bounds_reference", "cleanup_receipt_reference",
}
RESOURCE_KEYS = {
    "max_gpu_jobs", "max_requests", "max_wall_seconds", "max_trace_bytes",
    "max_build_rss_bytes", "max_build_seconds", "max_extra_scratch_bytes",
    "memory_reserve_bytes",
}


def _keys(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{label}: missing or unsupported fields")


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label}: integer outside supported range")


def _sha(value, label, size=64):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{%d}" % size, value) is None:
        raise ValueError(f"{label}: expected lowercase digest")


def _equal(value, expected, label):
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{label}: unsupported contract")


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 131072:
        raise ValueError("Plan/manifest must be bounded regular JSON")

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key: " + key)
            value[key] = item
        return value

    def reject(value):
        raise ValueError("Nonfinite JSON value: " + value)

    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=reject)


def chunk_spans(prompt_tokens, chunk_tokens, absolute_start=0):
    """Half-open absolute spans; chunk M is unrelated to active request count."""
    _integer(prompt_tokens, 1, 32768, "prompt_tokens")
    _integer(chunk_tokens, 1, prompt_tokens, "chunk_tokens")
    _integer(absolute_start, 0, 131072 - prompt_tokens, "absolute_start")
    return [(p, min(p + chunk_tokens, absolute_start + prompt_tokens))
            for p in range(absolute_start, absolute_start + prompt_tokens, chunk_tokens)]


def validate_plan(plan):
    _keys(plan, {"schema", "gpu_enabled", "scope", "binding", "controls",
                 "workloads", "freeze", "resource_bounds"}, "plan")
    _equal(plan["schema"], "megartx-prefill-cpu-plan-v1", "schema")
    _equal(plan["scope"], "cpu_preparation", "scope")
    _equal(plan["gpu_enabled"], False, "gpu_enabled")
    _keys(plan["binding"], {"repository_base_commit", "source_manifest_sha256"}, "binding")
    _equal(plan["binding"]["repository_base_commit"], BASE_COMMIT, "base commit")
    _sha(plan["binding"]["source_manifest_sha256"], "source manifest")
    _keys(plan["controls"], CONTROLS, "controls")
    for key, value in CONTROLS.items():
        _equal(plan["controls"][key], value, key)
    _keys(plan["freeze"], FREEZE_KEYS, "freeze")
    for key, value in plan["freeze"].items():
        if value is not None:
            if type(value) is not str or not value.strip():
                raise ValueError(key + ": nonempty reference required")
            if key.endswith("_sha256"):
                _sha(value, key)
            elif key.endswith("_commit"):
                _sha(value, key, 40)
    _keys(plan["resource_bounds"], RESOURCE_KEYS, "resource_bounds")
    _equal(plan["resource_bounds"]["max_gpu_jobs"], 1, "max_gpu_jobs")
    for key, value in plan["resource_bounds"].items():
        if value is not None:
            _integer(value, 1, 2**63 - 1, key)
    workloads = plan["workloads"]
    if type(workloads) is not list or len(workloads) != 3:
        raise ValueError("Require 2K/8K and separately deferred 32K")
    for cell, length in zip(workloads, (2048, 8192, 32768)):
        _keys(cell, {"id", "prompt_tokens", "chunk_tokens", "status",
                     "prompt_token_ids_sha256", "fit_receipt_sha256"}, "workload")
        _equal(cell["id"], f"p{length}", "workload ID")
        _equal(cell["prompt_tokens"], length, "whole prompt length")
        _equal(cell["status"], "deferred_fit" if length == 32768 else "planned", "status")
        chunks = cell["chunk_tokens"]
        if type(chunks) is not list or not chunks or len(chunks) > 8:
            raise ValueError("Bounded chunk sweep required")
        for chunk in chunks:
            chunk_spans(length, chunk)
        if len(set(chunks)) != len(chunks) or length not in chunks:
            raise ValueError("Unique chunks and unchunked control required")
        for key in ("prompt_token_ids_sha256", "fit_receipt_sha256"):
            if cell[key] is not None:
                _sha(cell[key], key)
    return plan


def verify_source_binding(plan, manifest_path, root):
    """Compare local source bytes with the reviewed manifest; no target probing."""
    validate_plan(plan)
    manifest = read_json(manifest_path)
    if hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest() != plan["binding"]["source_manifest_sha256"]:
        raise ValueError("Source manifest digest changed")
    _equal(manifest.get("schema"), "megartx-prefill-source-binding-v1", "source schema")
    _equal(manifest.get("repository_base_commit"), BASE_COMMIT, "source base")
    _equal(manifest.get("checkpoint_repository"), "nvidia/Gemma-4-26B-A4B-NVFP4", "checkpoint repository")
    _equal(manifest.get("checkpoint_revision"), CHECKPOINT_REVISION, "checkpoint revision")
    _equal(manifest.get("gpu_qualified"), False, "source qualification")
    files = manifest.get("repo_files")
    if type(files) is not dict or not files:
        raise ValueError("Source files absent")
    root = Path(root).resolve()
    actual = {}
    for name, expected in files.items():
        _sha(expected, name)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Source path must be relative and contained")
        source = root / relative
        if any((root / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
            raise ValueError("Source symlinks unsupported")
        if not source.is_file() or source.stat().st_size > 2**20:
            raise ValueError("Source file missing or unbounded: " + name)
        actual[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    candidates = [files]
    overlays = manifest.get("reviewed_source_overlays", [])
    if type(overlays) is not list or len(overlays) > 2:
        raise ValueError("At most two explicit CPU source overlays are supported")
    pair = {"src/megartx/m1_live.py", "src/megartx/vllm_scale_plugin.py"}
    for overlay in overlays:
        _keys(overlay, {"id", "source_head", "review_scope", "repo_files", "evidence_reference"}, "source overlay")
        _sha(overlay["source_head"], "overlay source head", 40)
        _equal(overlay["review_scope"], "cpu_source_compatibility_only", "overlay scope")
        for key in ("id", "evidence_reference"):
            if type(overlay[key]) is not str or not overlay[key].strip():
                raise ValueError("Source overlay needs explicit identity/evidence")
        _keys(overlay["repo_files"], pair, "atomic controller/plugin overlay")
        if not pair <= set(files):
            raise ValueError("Source overlay cannot add absent baseline files")
        for name, expected in overlay["repo_files"].items():
            _sha(expected, name)
        candidates.append({**files, **overlay["repo_files"]})
    native = manifest.get("native_source_overlay")
    composition = manifest.get("native_composition_overlay")
    if native is not None:
        _keys(native, {"source_base", "review_scope", "repo_files", "evidence_reference", "evidence_sha256"}, "native source overlay")
        _equal(native["source_base"], "cf656d6632b9f1b08019a527a9269a9ed0fb0a26", "native source base")
        _equal(native["review_scope"], "exact_cpu_source_compatibility_only", "native overlay scope")
        _equal(native["evidence_reference"], "docs/prefill/native-source-reconciliation.json", "native overlay reference")
        inventory = pair | {"scripts/run_scale_validation.py"}
        _keys(native["repo_files"], inventory, "exact native controller/plugin/launcher vector")
        evidence = root / native["evidence_reference"]
        if evidence.is_symlink() or not evidence.is_file() or hashlib.sha256(evidence.read_bytes()).hexdigest() != native["evidence_sha256"]:
            raise ValueError("Native source reconciliation digest changed")
        review = read_json(evidence)
        _equal(review["base"], native["source_base"], "native review base")
        _equal(review["repo_files"], native["repo_files"], "native exact source vector")
        for flag in ("gpu_executed", "fit_qualified", "numerical_qualified", "performance_qualified"):
            _equal(review[flag], False, "native pending " + flag)
        for name, expected in native["repo_files"].items():
            _sha(expected, name)
            if composition is not None:
                continue  # Preserve the parent vector; admit only the complete composition below.
            source = root / name
            if any((root / Path(*Path(name).parts[:i])).is_symlink() for i in range(1,len(Path(name).parts)+1)) or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError("Source drift: native atomic plugin/launcher vector " + name)
        if composition is None:
            candidates.append({**files, **{name: expected for name, expected in native["repo_files"].items() if name in files}})
    if composition is not None:
        if native is None:
            raise ValueError("Composition requires the unchanged native parent overlay")
        _keys(composition, {"prefill_parent", "dspark_parent", "review_scope", "repo_files",
                            "evidence_reference", "evidence_sha256"}, "native composition overlay")
        for key, expected in (("prefill_parent", "418e1ecf5af0b5974ecf767a50c6d54f5578bee0"),
                ("dspark_parent", "8027588365ed4b23c6ccbc73de2c8f710e2e166f"),
                ("review_scope", "exact_cpu_source_compatibility_only"),
                ("evidence_reference", "docs/prefill/native-composition-lineage.json")):
            _equal(composition[key], expected, "composition " + key)
        inventory = set(native["repo_files"]) | {"src/megartx/native_diagnostic_composition.py"}
        _keys(composition["repo_files"], inventory, "exact composed plugin/launcher/helper vector")
        _sha(composition["evidence_sha256"], "composition evidence")
        evidence = root / composition["evidence_reference"]
        if evidence.is_symlink() or not evidence.is_file() or hashlib.sha256(evidence.read_bytes()).hexdigest() != composition["evidence_sha256"]:
            raise ValueError("Native composition lineage digest changed")
        review = read_json(evidence)
        _equal(review["schema"], "megartx-native-v2-composition-lineage-v1", "composition schema")
        for key in ("prefill_parent", "dspark_parent"):
            _equal(review[key], composition[key], "composition review " + key)
        _equal(review["shared_repo_files"], composition["repo_files"], "composition exact source vector")
        _equal(review["previous_shared_repo_files"], native["repo_files"], "composition parent source vector")
        _equal(review["parent_ledgers"].get(native["evidence_reference"]),
               native["evidence_sha256"], "composition native parent ledger")
        for flag in ("gpu_executed", "fit_qualified", "numerical_qualified", "performance_qualified"):
            _equal(review[flag], False, "composition pending " + flag)
        for name, expected in composition["repo_files"].items():
            _sha(expected, name)
            source = root / name
            if (any((root / Path(*Path(name).parts[:i])).is_symlink() for i in range(1, len(Path(name).parts)+1))
                    or not source.is_file() or source.stat().st_size > 2**20
                    or hashlib.sha256(source.read_bytes()).hexdigest() != expected):
                raise ValueError("Source drift: native atomic composition vector " + name)
        candidates.append({**files, **{name: expected for name, expected in composition["repo_files"].items() if name in files}})
    if actual not in candidates:
        changed = sorted(name for name in files if actual[name] != files[name])
        raise ValueError("Source drift: " + ", ".join(changed))
    return len(files)


def intake(plan):
    validate_plan(plan)
    freeze_blockers = [key for key, value in plan["freeze"].items() if value is None]
    blockers = list(freeze_blockers)
    blockers += [key for key, value in plan["resource_bounds"].items() if value is None]
    cells = []
    for cell in plan["workloads"]:
        if cell["prompt_token_ids_sha256"] is None:
            blockers.append(cell["id"] + ": prompt identity pending")
        if cell["fit_receipt_sha256"] is None or cell["status"] == "deferred_fit":
            blockers.append(cell["id"] + ": resident peak fit including output reserve pending")
        if cell["status"] == "deferred_fit":
            continue
        for chunk in cell["chunk_tokens"]:
            spans = chunk_spans(cell["prompt_tokens"], chunk)
            cells.append({"id": cell["id"], "prompt_tokens": cell["prompt_tokens"],
                          "chunk_tokens": chunk, "forward_count": len(spans),
                          "final_chunk_m": spans[-1][1] - spans[-1][0],
                          "capacity_tokens": cell["prompt_tokens"] + 256,
                          "dispatch_status": "pending_exact_host_validation"})
    return {"plan_consistent": True, "gpu_execution_available": False,
            "gpu_qualified": False, "freeze_references_complete": not freeze_blockers,
            "blockers": sorted(blockers), "planned_cells": cells,
            "runner_blocker": "No GPU runner implemented; separate reviewed opt-in adapter and sole-owner slot required"}


def expert_row_ledger(selected_ids, selected_weights, padding_multiple=1):
    """Logical row/padding accounting only, never observed CUDA work or routing."""
    _integer(padding_multiple, 1, 1024, "padding_multiple")
    if type(selected_ids) is not list or not 1 <= len(selected_ids) <= 32768:
        raise ValueError("Bounded nonempty row list required")
    if type(selected_weights) is not list or len(selected_weights) != len(selected_ids):
        raise ValueError("Weights must match rows")
    counts, positive = [0] * 128, [0] * 128
    for ids, weights in zip(selected_ids, selected_weights):
        if type(ids) is not list or len(ids) != 8 or type(weights) is not list or len(weights) != 8:
            raise ValueError("Exactly eight IDs/weights per row required")
        for expert, weight in zip(ids, weights):
            _integer(expert, 0, 127, "expert ID")
            if type(weight) not in (float, int) or not math.isfinite(weight) or weight < 0:
                raise ValueError("Weights must be finite and nonnegative")
            counts[expert] += 1
            positive[expert] += weight > 0
        if len(set(ids)) != 8:
            raise ValueError("Selected expert IDs must be distinct per row")
    padded = [((n + padding_multiple - 1) // padding_multiple) * padding_multiple for n in counts]
    flop_per_row = 2 * (2 * 704 * 2816 + 2816 * 704)
    return {"input_m": len(selected_ids), "selected_slots": sum(counts),
            "selected_m": counts, "positive_weight_m": positive,
            "projected_padded_m": padded, "empty_experts": counts.count(0),
            "projected_padding_rows": sum(padded) - sum(counts),
            "logical_expert_gemm_flops": sum(counts) * flop_per_row,
            "actual_executed_work_known": False}


def summarize_timing(record):
    """Whole-prompt host wall boundaries; GPU critical-path time is separate."""
    timestamps = ("request_accept_ns", "input_ready_ns", "prompt_begin_ns",
                  "prompt_complete_ns", "kv_ready_ns", "first_token_ns")
    _keys(record, {*timestamps, "prompt_tokens", "chunk_rows", "state",
                   "observer_enabled", "preallocated", "bucket_warmed"}, "timing")
    _equal(record["observer_enabled"], False, "timed observer")
    if type(record["state"]) is not str or record["state"] not in {"cold", "warm"}:
        raise ValueError("Cold/warm state must be explicit")
    for key in ("preallocated", "bucket_warmed"):
        if type(record[key]) is not bool:
            raise ValueError(key + ": boolean required")
        if record["state"] == "warm" and record[key] is not True:
            raise ValueError("Warm timing requires completed preallocation/warmup")
    values = [record[key] for key in timestamps]
    for value in values:
        _integer(value, 0, 2**63 - 1, "timestamp")
    if sorted(values) != values or record["prompt_complete_ns"] == record["prompt_begin_ns"]:
        raise ValueError("Timing boundaries must be ordered with positive prompt duration")
    _integer(record["prompt_tokens"], 1, 32768, "prompt_tokens")
    rows = record["chunk_rows"]
    if type(rows) is not list or not rows:
        raise ValueError("Actual chunk rows required")
    for row in rows:
        _integer(row, 1, record["prompt_tokens"], "chunk rows")
    if sum(rows) != record["prompt_tokens"]:
        raise ValueError("Chunk rows must cover the whole prompt exactly once")
    duration = record["prompt_complete_ns"] - record["prompt_begin_ns"]
    return {"state": record["state"], "prompt_latency_ns": duration,
            "prompt_tokens_per_second": record["prompt_tokens"] * 1e9 / duration,
            "handoff_ns": record["kv_ready_ns"] - record["prompt_complete_ns"],
            "ttft_ns": record["first_token_ns"] - record["request_accept_ns"],
            "gpu_time_measured": False, "performance_qualified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = read_json(args.plan)
        count = verify_source_binding(plan, args.source_manifest, args.root)
        print(json.dumps({**intake(plan), "local_source_files_bound": count}, indent=2))
        return 0
    except (ValueError, TypeError, OSError) as error:
        parser.exit(2, f"prefill-plan: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
