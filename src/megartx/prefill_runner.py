"""Default-off prefill slot compiler and offline profiling client (stdlib only).

No server, process, network, device, plugin or native module is launched here.
Receipt validation checks structure/integrity; independent review owns acceptance.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Protocol

from . import prefill_plan as plan_contract


BASE_COMMIT = "5a32504b5ccb209a779b8a110eb2ebc56d72e373"
LAUNCHER_CONTRACT = "megartx-prefill-launcher-adapter-v1"
RECEIPTS = {
    "source_review", "launcher_review", "gpu_scope", "ownership", "resource",
    "cleanup", "g0", "g1", "numerical", "natural_correction_coverage",
    "dispatch", "decode_control", "fit_p2048", "fit_p8192",
}
PROFILE_COMPONENTS = {
    "attention": {"qk_norm", "rope", "cache", "attention"},
    "dense": {"qkv", "o_projection", "shared_mlp", "norm"},
    "expert": {"router", "quantize", "pack", "dispatch", "expert_fc1",
               "expert_activation", "expert_fc2", "combine", "correction", "suppressed_stock"},
    "head": {"final_norm", "last_row", "head", "softcap", "sampling"},
}
MEMORY_CATEGORIES = {"model_scales", "retained_originals", "kv_local", "kv_global",
                     "activations", "route_packing", "scratch", "conversion", "allocator_other"}
GLOBAL_LAYERS = {5, 11, 17, 23, 29}
RUNNER_FILES = {
    "src/megartx/prefill_runner.py", "tests/test_prefill_runner.py",
    "src/megartx/prefill_plan.py", "numerical_reference/prefill_cache_reference.py",
    "docs/prefill/runner-protocol.md", "docs/prefill/README.md", "docs/action-plans/wp7.md",
}
keys = plan_contract._keys
integer = plan_contract._integer
equal = plan_contract._equal
sha = plan_contract._sha


class LauncherAdapter(Protocol):
    """Future reviewed launcher: serialize jobs, preserve fallback, own cleanup.

    This is a data interface, with no implementation or dynamic-import facility.
    The final PR15 integration must validate the packet before device work and
    return normalized records, including its owned startup/cleanup receipts.
    """

    def describe_contract(self) -> dict: ...
    def execute_serialized(self, protocol: dict, private_prompts: dict) -> dict: ...


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("ascii")).hexdigest()


def _text(value, label):
    if type(value) is not str or not value.strip() or len(value) > 256:
        raise ValueError(label + ": bounded nonempty string required")


def _list(value, low, high, label):
    if type(value) is not list or not low <= len(value) <= high:
        raise ValueError(label + ": bounded list required")


def _json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(value):
        raise ValueError("Nonfinite JSON value: " + value)

    return json.loads(text, object_pairs_hook=unique, parse_constant=reject)


def read_records(path):
    """Normalized private records only; raw profiler traces are not ingested."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 2**20:
        raise ValueError("Records must be regular JSON of at most 8 MiB; select a smaller slot")
    return _json(path.read_text())


def validate_manifest(manifest):
    keys(manifest, {"schema", "gpu_enabled", "repository_base_commit", "plan_sha256",
                    "runner_binding_sha256", "run_namespace", "launcher", "observer_sites", "cache_layout",
                    "receipts"}, "runner manifest")
    equal(manifest["schema"], "megartx-prefill-runner-v1", "runner schema")
    equal(manifest["gpu_enabled"], False, "GPU execution remains disabled")
    equal(manifest["repository_base_commit"], BASE_COMMIT, "implementation base")
    for key in ("plan_sha256", "runner_binding_sha256"):
        sha(manifest[key], key)
    namespace = manifest["run_namespace"]
    if namespace is not None and (type(namespace) is not str
            or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", namespace) is None):
        raise ValueError("run_namespace must be a bounded slot identifier")
    launcher = manifest["launcher"]
    keys(launcher, {"contract", "pr15_final_commit", "source_manifest_sha256",
                    "stream_provider_sha256", "supported_chunk_tokens"}, "launcher")
    equal(launcher["contract"], LAUNCHER_CONTRACT, "launcher contract")
    for key in ("pr15_final_commit", "source_manifest_sha256", "stream_provider_sha256"):
        if launcher[key] is not None:
            sha(launcher[key], key, 40 if key.endswith("commit") else 64)
    _list(launcher["supported_chunk_tokens"], 0, 8, "launcher chunk support")
    for chunk in launcher["supported_chunk_tokens"]:
        integer(chunk, 1, 8192, "launcher supported chunk")
    if len(set(launcher["supported_chunk_tokens"])) != len(launcher["supported_chunk_tokens"]):
        raise ValueError("Duplicate launcher chunk support")
    layout = manifest["cache_layout"]
    keys(layout, {"local_storage_policy", "page_tokens", "layout_sha256"}, "cache layout")
    if layout["local_storage_policy"] not in {None, "full_context", "bounded_window"}:
        raise ValueError("Unsupported physical local KV policy")
    if layout["page_tokens"] is not None:
        integer(layout["page_tokens"], 1, 1024, "KV page tokens")
        if layout["page_tokens"] & (layout["page_tokens"] - 1):
            raise ValueError("Power-of-two KV page size required")
    if layout["layout_sha256"] is not None:
        sha(layout["layout_sha256"], "cache layout")
    keys(manifest["observer_sites"], PROFILE_COMPONENTS, "observer sites")
    for name, value in manifest["observer_sites"].items():
        if value is not None:
            sha(value, name)
    keys(manifest["receipts"], RECEIPTS, "receipts")
    for name, value in manifest["receipts"].items():
        if value is not None:
            sha(value, name)
    return manifest


def load_inputs(manifest_path, plan_path, source_path, runner_binding_path, root):
    manifest = validate_manifest(plan_contract.read_json(manifest_path))
    plan = plan_contract.read_json(plan_path)
    if hashlib.sha256(Path(plan_path).read_bytes()).hexdigest() != manifest["plan_sha256"]:
        raise ValueError("Prefill plan digest changed")
    plan_contract.verify_source_binding(plan, source_path, root)
    binding = plan_contract.read_json(runner_binding_path)
    if hashlib.sha256(Path(runner_binding_path).read_bytes()).hexdigest() != manifest["runner_binding_sha256"]:
        raise ValueError("Runner binding digest changed")
    keys(binding, {"schema", "repository_base_commit", "repo_files"}, "runner binding")
    equal(binding["schema"], "megartx-prefill-runner-source-v1", "runner binding schema")
    equal(binding["repository_base_commit"], BASE_COMMIT, "runner binding base")
    # Extend historical pins with prefill-only source files; never rehash peers.
    combined = plan_contract.read_json(source_path)
    if type(binding["repo_files"]) is not dict or not binding["repo_files"]:
        raise ValueError("Runner source files absent")
    keys(binding["repo_files"], RUNNER_FILES, "runner source inventory")
    root = Path(root).resolve()
    for name, expected in binding["repo_files"].items():
        sha(expected, name)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Runner source path must be relative and contained")
        source = root / relative
        if any((root / Path(*relative.parts[:i])).is_symlink()
               for i in range(1, len(relative.parts) + 1)):
            raise ValueError("Runner source symlinks unsupported")
        if not source.is_file() or source.stat().st_size > 2**20:
            raise ValueError("Runner source absent or unbounded: " + name)
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("Runner source drift: " + name)
    if not set(combined["repo_files"]).isdisjoint(binding["repo_files"]):
        raise ValueError("Runner binding must extend, not replace historical source pins")
    return manifest, plan


def compile_protocol(manifest, plan, workload=None, chunk=None, receipt_root=None):
    """Compile deterministic, serialized jobs; null decisions stay blockers."""
    validate_manifest(manifest)
    plan_contract.validate_plan(plan)
    if workload not in {None, "p2048", "p8192"}:
        raise ValueError("Only full 2K/8K prefill; 32K and short DSpark verification deferred")
    if chunk is not None:
        integer(chunk, 1, 8192, "chunk selection")
        if workload is None:
            raise ValueError("Chunk selection requires a workload")
    cells = [c for c in plan_contract.intake(plan)["planned_cells"]
             if (workload is None or c["id"] == workload)
             and (chunk is None or c["chunk_tokens"] == chunk)]
    if not cells:
        raise ValueError("Chunk outside the reviewed sweep")
    # Receipt digests cannot hash themselves through the plan. Bind all executable
    # inputs; acceptance pointers are independently content-checked below.
    identity_keys = {"decode_control_commit", "target_environment_manifest_sha256",
                     "tokenizer_template_manifest_sha256", "prompt_set_manifest_sha256",
                     "oracle_contract_sha256", "sampling_policy", "eos_policy"}
    scope = digest({"manifest": {k: v for k, v in manifest.items() if k not in {"receipts", "plan_sha256"}},
                    "controls": plan["controls"], "source_binding": plan["binding"],
                    "resource_bounds": plan["resource_bounds"],
                    "identities": {k: plan["freeze"][k] for k in identity_keys},
                    "cells": cells,
                    "prompt_identities": {c["id"]: c["prompt_token_ids_sha256"]
                                          for c in plan["workloads"] if c["id"] in {x["id"] for x in cells}}})
    blockers = [key for key, value in plan["freeze"].items() if value is None]
    blockers += [key for key, value in plan["resource_bounds"].items() if value is None]
    if plan["freeze"]["sampling_policy"] not in {None, "greedy_seed_1234"}:
        raise ValueError("Client requires frozen greedy_seed_1234 policy")
    if plan["freeze"]["eos_policy"] not in {None, "ignore_eos_256"}:
        raise ValueError("Client requires frozen ignore_eos_256 output policy")
    if manifest["run_namespace"] is None:
        blockers.append("run_namespace")
    blockers += ["launcher." + k for k, v in manifest["launcher"].items() if v is None]
    blockers += ["observer_sites." + k for k, v in manifest["observer_sites"].items() if v is None]
    blockers += ["cache_layout." + k for k, v in manifest["cache_layout"].items() if v is None]
    required = RECEIPTS - {"fit_p2048", "fit_p8192"} | {"fit_" + c["id"] for c in cells}
    for name in sorted(required):
        expected = manifest["receipts"][name]
        if expected is None or receipt_root is None:
            blockers.append("receipt." + name)
            continue
        path = Path(receipt_root) / (expected + ".json")
        if not path.exists():
            blockers.append("receipt." + name + ": file missing")
            continue
        receipt = plan_contract.read_json(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Receipt digest changed: " + name)
        keys(receipt, {"schema", "kind", "scope_sha256", "decision", "evidence_sha256",
                       "reviewer_reference"}, "receipt")
        equal(receipt["schema"], "megartx-prefill-receipt-v1", "receipt schema")
        equal(receipt["kind"], name, "receipt kind")
        equal(receipt["scope_sha256"], scope, "receipt scope")
        sha(receipt["evidence_sha256"], "receipt evidence")
        _text(receipt["reviewer_reference"], "receipt reviewer")
        evidence_path = Path(receipt_root) / (receipt["evidence_sha256"] + ".evidence.json")
        if not evidence_path.exists():
            blockers.append("receipt." + name + ": evidence file missing")
        else:
            plan_contract.read_json(evidence_path)
            if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != receipt["evidence_sha256"]:
                raise ValueError("Receipt evidence digest changed: " + name)
        if receipt["decision"] != "accepted":
            blockers.append("receipt." + name + ": unresolved decision")
    jobs = []
    for cell in cells:
        original = next(c for c in plan["workloads"] if c["id"] == cell["id"])
        if cell["chunk_tokens"] not in manifest["launcher"]["supported_chunk_tokens"]:
            blockers.append(cell["id"] + f": launcher/collector chunk {cell['chunk_tokens']} unsupported")
        for key in ("prompt_token_ids_sha256", "fit_receipt_sha256"):
            if original[key] is None:
                blockers.append(cell["id"] + ": " + key)
        if original["fit_receipt_sha256"] != manifest["receipts"]["fit_" + cell["id"]]:
            blockers.append(cell["id"] + ": fit receipt binding mismatch")
        spans = plan_contract.chunk_spans(cell["prompt_tokens"], cell["chunk_tokens"])
        for mode, region, state in [("initialization", None, "cold"), ("correctness", None, "warm"),
                                    *[("profile", r, "warm") for r in PROFILE_COMPONENTS],
                                    ("memory", None, "warm"), ("timing", None, "cold"),
                                    ("timing", None, "warm")]:
            run_id = (f"{manifest['run_namespace'] or 'pending'}-{cell['id']}-c{cell['chunk_tokens']}"
                      f"-{mode}-{region or state}")
            jobs.append({"run_id": run_id, "request_id": None if mode == "initialization" else run_id,
                         "workload": cell["id"], "prompt_tokens": cell["prompt_tokens"],
                         "prompt_token_ids_sha256": original["prompt_token_ids_sha256"],
                         "chunk_tokens": cell["chunk_tokens"], "spans": [list(s) for s in spans],
                         "capacity_tokens": cell["capacity_tokens"], "mode": mode, "region": region,
                         "state": state, "observer_enabled": mode in {"correctness", "profile", "memory"}})
    count = sum(j["request_id"] is not None for j in jobs)
    limit = plan["resource_bounds"]["max_requests"]
    if limit is not None and count > limit:
        blockers.append("max_requests: serialized suite exceeds bound")
    return {"schema": "megartx-prefill-protocol-v1", "scope_sha256": scope,
            "gpu_execution_available": False, "gpu_qualified": False,
            "prerequisites_complete": not blockers, "blockers": sorted(set(blockers)),
            "launcher": manifest["launcher"], "observer_sites": manifest["observer_sites"],
            "cache_layout": manifest["cache_layout"],
            "resource_bounds": plan["resource_bounds"], "controls": plan["controls"],
            "identities": {k: plan["freeze"][k] for k in identity_keys},
            "request_count": count, "jobs": jobs,
            "remaining_integration": "Final reviewed PR15 launcher + prefill adapter/source review + owner slot; GPU path disabled"}


def require_live_execution(protocol):
    """No command, env var, receipt or injected object can turn on device work."""
    if not protocol["prerequisites_complete"]:
        raise ValueError("Unresolved admission: " + "; ".join(protocol["blockers"]))
    raise ValueError("GPU execution disabled: final reviewed PR15 adapter integration is not implemented")


def validate_prompts(prompts, protocol):
    keys(prompts, {"schema", "token_ids"}, "private prompts")
    equal(prompts["schema"], "megartx-prefill-private-prompts-v1", "prompt schema")
    workloads = {j["workload"]: j for j in protocol["jobs"]}
    keys(prompts["token_ids"], workloads, "selected prompt IDs")
    for name, job in workloads.items():
        tokens = prompts["token_ids"][name]
        _list(tokens, job["prompt_tokens"], job["prompt_tokens"], "prompt tokens")
        for token in tokens:
            integer(token, 0, 262143, "token ID")
        equal(digest(tokens), job["prompt_token_ids_sha256"], "prompt token identity")
    return prompts["token_ids"]


def request_payload(job, tokens):
    if job["request_id"] is None:
        raise ValueError("Initialization is not a model request")
    _list(tokens, job["prompt_tokens"], job["prompt_tokens"], "prompt tokens")
    for token in tokens:
        integer(token, 0, 262143, "token ID")
    equal(digest(tokens), job["prompt_token_ids_sha256"], "prompt token identity")
    return {"model": "gemma4-nvfp4", "prompt": tokens, "request_id": job["request_id"],
            "add_special_tokens": False, "max_tokens": 256, "temperature": 0.0,
            "top_p": 1.0, "top_k": 0, "seed": 1234, "n": 1, "ignore_eos": True,
            "stream": True, "stream_interval": 1, "stream_options": {"include_usage": True},
            "return_token_ids": True, "skip_special_tokens": False}


def parse_client_stream(events, job, response_id):
    """Parse timestamped SSE data payloads from the future local client boundary."""
    _text(response_id, "server response ID")
    _list(events, 2, 1024, "stream events")
    ids, first, usage, finish, done, previous, multi = [], None, None, None, False, -1, False
    for event in events:
        keys(event, {"received_ns", "data"}, "client event")
        integer(event["received_ns"], 0, 2**63 - 1, "client clock")
        if event["received_ns"] < previous or done:
            raise ValueError("Unordered stream or data after DONE")
        previous = event["received_ns"]
        data = event["data"]
        if data == "[DONE]":
            done = True
            continue
        if type(data) is not str or len(data) > 65536:
            raise ValueError("Bounded SSE data string required")
        data = _json(data)
        if type(data) is not dict:
            raise ValueError("SSE object required")
        equal(data.get("id"), response_id, "response stream correlation")
        choices = data.get("choices")
        _list(choices, 0, 1, "single-request choices")
        if choices:
            choice = choices[0]
            if type(choice) is not dict:
                raise ValueError("Choice object required")
            equal(choice.get("index"), 0, "choice index")
            tokens = choice.get("token_ids", [])
            _list(tokens, 0, 256, "stream token IDs")
            if tokens and finish is not None:
                raise ValueError("Tokens after finish")
            for token in tokens:
                integer(token, 0, 262143, "output token ID")
            if tokens and first is None:
                first = previous
            multi |= len(tokens) > 1
            ids.extend(tokens)
            if choice.get("finish_reason") is not None:
                if finish is not None:
                    raise ValueError("Duplicate finish")
                finish = choice["finish_reason"]
        if data.get("usage") is not None:
            if usage is not None:
                raise ValueError("Duplicate usage")
            if finish is None:
                raise ValueError("Final usage reported before finish")
            usage = data["usage"]
    if not done or first is None or finish != "length" or len(ids) != 256:
        raise ValueError("Incomplete fixed-output stream")
    keys(usage, {"prompt_tokens", "completion_tokens", "total_tokens"}, "usage")
    for key, expected in {"prompt_tokens": job["prompt_tokens"], "completion_tokens": len(ids),
                          "total_tokens": job["prompt_tokens"] + len(ids)}.items():
        equal(usage[key], expected, "usage." + key)
    return {"first_token_ns": first, "done_ns": previous, "completion_tokens": len(ids),
            "multi_token_chunks": multi, "itl_per_token_available": not multi}


def _handoff(value, job, layout):
    keys(value, {"complete", "poisoned", "committed_length", "next_position", "capacity",
                 "kv_dtype", "layers", "comparison_receipt_sha256"}, "handoff")
    for key, expected in {"complete": True, "poisoned": False, "committed_length": job["prompt_tokens"],
                          "next_position": job["prompt_tokens"], "capacity": job["capacity_tokens"],
                          "kv_dtype": "bfloat16"}.items():
        equal(value[key], expected, "handoff." + key)
    sha(value["comparison_receipt_sha256"], "native cache comparison receipt")
    _list(value["layers"], 30, 30, "all-layer handoff")
    for index, layer in enumerate(value["layers"]):
        keys(layer, {"layer", "kind", "positions_start", "positions_end", "processed_k_sha256",
                     "processed_v_sha256", "layout_sha256"}, "layer handoff")
        local = index not in GLOBAL_LAYERS
        for key, expected in {"layer": index, "kind": "local" if local else "global",
                              "positions_start": max(0, job["prompt_tokens"] - 1024) if local else 0,
                              "positions_end": job["prompt_tokens"]}.items():
            equal(layer[key], expected, "handoff." + key)
        for key in ("processed_k_sha256", "processed_v_sha256", "layout_sha256"):
            sha(layer[key], key)
        equal(layer["layout_sha256"], layout["layout_sha256"], "reviewed cache layout identity")


def _profile(value, job, protocol):
    keys(value, {"stream_bindings", "launches", "kernels", "expert_rows"}, "profile")
    streams = {}
    _list(value["stream_bindings"], 1, 64, "stream mappings")
    for mapping in value["stream_bindings"]:
        keys(mapping, {"runtime_stream_key", "cupti_stream_id", "provider_sha256"}, "stream mapping")
        _text(mapping["runtime_stream_key"], "runtime stream key")
        integer(mapping["cupti_stream_id"], 0, 2**63 - 1, "CUPTI stream ID")
        equal(mapping["provider_sha256"], protocol["launcher"]["stream_provider_sha256"], "stream provider binding")
        sha(mapping["provider_sha256"], "stream provider")
        if mapping["runtime_stream_key"] in streams or mapping["cupti_stream_id"] in streams.values():
            raise ValueError("Ambiguous stream mapping")
        streams[mapping["runtime_stream_key"]] = mapping["cupti_stream_id"]
    launches = {}
    _list(value["launches"], 1, 100000, "launches")
    for launch in value["launches"]:
        keys(launch, {"correlation_id", "forward_index", "layer", "component", "source_site",
                      "source_sha256", "runtime_stream_key", "m", "n", "k"}, "launch")
        _text(launch["correlation_id"], "launch correlation")
        if launch["correlation_id"] in launches or launch["runtime_stream_key"] not in streams:
            raise ValueError("Duplicate launch ID or unknown runtime stream")
        integer(launch["forward_index"], 0, len(job["spans"]) - 1, "forward index")
        integer(launch["layer"], 0, 29, "layer")
        if launch["component"] not in PROFILE_COMPONENTS[job["region"]]:
            raise ValueError("Attribution crossed separate profiling run")
        _text(launch["source_site"], "source site")
        sha(launch["source_sha256"], "source site digest")
        equal(launch["source_sha256"], protocol["observer_sites"][job["region"]], "observer source binding")
        for key in ("m", "n", "k"):
            integer(launch[key], 1, 2**31 - 1, "actual shape " + key)
        if job["region"] == "head" and launch["m"] != 1:
            raise ValueError("First-token head requires last-prompt-row M=1")
        if launch["component"] in {"attention", "qkv", "o_projection", "shared_mlp"}:
            start, end = job["spans"][launch["forward_index"]]
            equal(launch["m"], end - start, "actual chunk query/projection M")
        launches[launch["correlation_id"]] = launch
    seen, sums, byte_sums, intervals = set(), {}, {}, {}
    _list(value["kernels"], 1, 100000, "kernels")
    kernel_ids = set()
    for kernel in value["kernels"]:
        keys(kernel, {"kernel_id", "correlation_id", "cupti_stream_id", "name", "device_begin_ns",
                      "device_end_ns", "bytes", "grid", "block"}, "kernel")
        _text(kernel["kernel_id"], "kernel ID")
        if kernel["kernel_id"] in kernel_ids or kernel["correlation_id"] not in launches:
            raise ValueError("Duplicate kernel or orphan GPU launch")
        kernel_ids.add(kernel["kernel_id"])
        launch = launches[kernel["correlation_id"]]
        equal(kernel["cupti_stream_id"], streams[launch["runtime_stream_key"]], "GPU/CPU stream correlation")
        _text(kernel["name"], "kernel name")
        for key in ("device_begin_ns", "device_end_ns", "bytes"):
            integer(kernel[key], 0, 2**63 - 1, key)
        if kernel["device_end_ns"] <= kernel["device_begin_ns"]:
            raise ValueError("Positive device duration required")
        for key in ("grid", "block"):
            _list(kernel[key], 3, 3, key)
            for axis in kernel[key]:
                integer(axis, 1, 2**31 - 1, key)
        component = launch["component"]
        sums[component] = sums.get(component, 0) + kernel["device_end_ns"] - kernel["device_begin_ns"]
        byte_sums[component] = byte_sums.get(component, 0) + kernel["bytes"]
        intervals.setdefault(kernel["cupti_stream_id"], []).append((kernel["device_begin_ns"], kernel["device_end_ns"]))
        seen.add(kernel["correlation_id"])
    if seen != set(launches):
        raise ValueError("Uncorrelated CPU launch")
    expected_sites = ({(len(job["spans"]) - 1, 29)} if job["region"] == "head"
                      else {(i, layer) for i in range(len(job["spans"])) for layer in range(30)})
    if {(launch["forward_index"], launch["layer"]) for launch in launches.values()} != expected_sites:
        raise ValueError("Incomplete layer/chunk profiling coverage or repeated first-token head")
    rows = value["expert_rows"]
    expected = {(i, layer) for i in range(len(job["spans"])) for layer in range(30)} if job["region"] == "expert" else set()
    _list(rows, len(expected), len(expected), "expert shape ledger")
    covered = set()
    for row in rows:
        keys(row, {"forward_index", "layer", "selected_m", "positive_m", "scheduled_m",
                   "route_sha256", "correction_hits", "suppressed_stock_rows"}, "expert row ledger")
        pair = (row["forward_index"], row["layer"])
        if any(type(x) is not int for x in pair) or pair not in expected or pair in covered:
            raise ValueError("Expert row identity/coverage changed")
        covered.add(pair)
        m = job["spans"][pair[0]][1] - job["spans"][pair[0]][0]
        for field in ("selected_m", "positive_m", "scheduled_m"):
            _list(row[field], 128, 128, field)
            for count in row[field]:
                integer(count, 0, 2**31 - 1, field)
        if sum(row["selected_m"]) != m * 8 or any(p > s or a < p for s, p, a in
                zip(row["selected_m"], row["positive_m"], row["scheduled_m"])):
            raise ValueError("Expert selected/positive/scheduled counts inconsistent")
        sha(row["route_sha256"], "route digest")
        for field in ("correction_hits", "suppressed_stock_rows"):
            integer(row[field], 0, m * 8, field)
    # Union by stream measures observed busy spans; cross-stream sums may overlap.
    busy = {}
    for stream, spans in intervals.items():
        total, end = 0, -1
        for begin, finish in sorted(spans):
            total += max(0, finish - max(begin, end))
            end = max(end, finish)
        busy[str(stream)] = total
    return {"kernel_sum_ns_by_component": sums, "reported_bytes_by_component": byte_sums,
            "stream_busy_union_ns": busy,
            "critical_path_ns": None, "critical_path_status": "dependency timeline still required",
            "kernel_count": len(kernel_ids), "expert_layer_chunk_records": len(rows)}


def _memory(value, job, bounds, layout):
    phases = ["load", "import_repack", "compile_warmup", *[f"chunk_{i}" for i in range(len(job["spans"]))],
              "handoff", "first_head", "decode_continuation"]
    keys(value, {"device_total_bytes", "samples"}, "memory")
    integer(value["device_total_bytes"], 1, 2**63 - 1, "device total")
    _list(value["samples"], len(phases), len(phases), "memory phases")
    if any(v is None for v in layout.values()):
        raise ValueError("Unresolved physical KV storage/layout")
    local_positions = (job["capacity_tokens"] if layout["local_storage_policy"] == "full_context"
                       else max(1024, max(end - max(0, start - 1023) for start, end in job["spans"])))
    page = layout["page_tokens"]
    local_positions = ((local_positions + page - 1) // page) * page
    global_positions = ((job["capacity_tokens"] + page - 1) // page) * page
    unattributed = []
    for index, (sample, phase) in enumerate(zip(value["samples"], phases)):
        keys(sample, {"phase", "allocated_bytes", "reserved_bytes", "device_used_bytes", "allocations"}, "memory sample")
        equal(sample["phase"], phase, "memory phase")
        for field in ("allocated_bytes", "reserved_bytes", "device_used_bytes"):
            integer(sample[field], 0, value["device_total_bytes"], field)
        if not sample["allocated_bytes"] <= sample["reserved_bytes"] <= sample["device_used_bytes"]:
            raise ValueError("Allocated/reserved/device accounting order changed")
        _list(sample["allocations"], 0, 10000, "physical allocations")
        categories, ids = {k: 0 for k in MEMORY_CATEGORIES}, set()
        for allocation in sample["allocations"]:
            keys(allocation, {"allocation_id", "category", "bytes"}, "allocation")
            _text(allocation["allocation_id"], "physical allocation ID")
            if allocation["allocation_id"] in ids or allocation["category"] not in categories:
                raise ValueError("Aliased allocation or unknown memory category")
            ids.add(allocation["allocation_id"])
            integer(allocation["bytes"], 0, value["device_total_bytes"], "allocation bytes")
            categories[allocation["category"]] += allocation["bytes"]
        if sum(categories.values()) > sample["allocated_bytes"]:
            raise ValueError("Memory ledger double-counts allocated bytes")
        unattributed.append(sample["allocated_bytes"] - sum(categories.values()))
        if index >= 3 and (categories["kv_local"] < 25 * 8192 * local_positions
                or categories["kv_global"] < 5 * 4096 * global_positions):
            raise ValueError("BF16 KV ledger omits chunk lifetime or output capacity")
        if bounds["memory_reserve_bytes"] is None or bounds["max_extra_scratch_bytes"] is None:
            raise ValueError("Unresolved memory/scratch bounds")
        if (value["device_total_bytes"] - sample["device_used_bytes"] < bounds["memory_reserve_bytes"]
                or categories["scratch"] > bounds["max_extra_scratch_bytes"]):
            raise ValueError("Memory reserve/scratch bound breached")
    return {**{field: max(s[field] for s in value["samples"])
               for field in ("allocated_bytes", "reserved_bytes", "device_used_bytes")},
            "peak_unattributed_allocated_bytes": max(unattributed)}


def validate_records(protocol, records, prompts):
    """Fail on incomplete records; never turn a replay into a numerical/GPU gate."""
    tokens = validate_prompts(prompts, protocol)
    keys(records, {"schema", "scope_sha256", "provenance", "host_clock_id", "runs"}, "records")
    equal(records["schema"], "megartx-prefill-records-v1", "record schema")
    equal(records["scope_sha256"], protocol["scope_sha256"], "record scope")
    provenance = records["provenance"]
    keys(provenance, {"kind", "origin_reference", "artifact_sha256", "sanitized"}, "provenance")
    if provenance["kind"] not in {"synthetic_cpu", "recorded_target"}:
        raise ValueError("Explicit fixture/recorded provenance required")
    _text(provenance["origin_reference"], "record origin")
    sha(provenance["artifact_sha256"], "source artifact")
    equal(provenance["sanitized"], True, "sanitized normalized records")
    _text(records["host_clock_id"], "single host monotonic clock identity")
    _list(records["runs"], len(protocol["jobs"]), len(protocol["jobs"]), "serialized run count")
    summaries, previous, trace_bytes, begin_suite = [], -1, 0, None
    for record, job in zip(records["runs"], protocol["jobs"]):
        keys(record, {"run_id", "request_id", "mode", "region", "state", "observer_enabled", "status", "stop_reason", "checks", "lifecycle_receipts",
                      "run_begin_ns", "run_end_ns", "trace_bytes", "initialization", "forwards",
                      "handoff", "server_response_id", "client_events", "timing", "profile", "memory"}, "run")
        for key in ("run_id", "request_id", "mode", "region", "state", "observer_enabled"):
            equal(record[key], job[key], "run." + key)
        equal(record["status"], "complete", "run completion status")
        equal(record["stop_reason"], None, "run stop reason")
        lifecycle = record["lifecycle_receipts"]
        keys(lifecycle, {"startup_sha256", "cleanup_sha256", "launcher_source_manifest_sha256",
                         "ownership_scope_sha256", "owned_only", "cleanup_complete"}, "launcher lifecycle receipts")
        for field in ("startup_sha256", "cleanup_sha256", "launcher_source_manifest_sha256", "ownership_scope_sha256"):
            sha(lifecycle[field], field)
        equal(lifecycle["launcher_source_manifest_sha256"], protocol["launcher"]["source_manifest_sha256"], "launcher lifecycle source")
        equal(lifecycle["ownership_scope_sha256"], protocol["scope_sha256"], "launcher lifecycle scope")
        for field in ("owned_only", "cleanup_complete"):
            equal(lifecycle[field], True, "launcher lifecycle." + field)
        for key in ("run_begin_ns", "run_end_ns", "trace_bytes"):
            integer(record[key], 0, 2**63 - 1, key)
        if record["run_begin_ns"] < previous or record["run_end_ns"] <= record["run_begin_ns"]:
            raise ValueError("Runs must be serialized on one host clock")
        previous = record["run_end_ns"]
        if begin_suite is None:
            begin_suite = record["run_begin_ns"]
        trace_bytes += record["trace_bytes"]
        summary = {"run_id": job["run_id"], "mode": job["mode"], "state": job["state"]}
        if job["mode"] == "initialization":
            keys(record["initialization"], {"load_ns", "import_repack_ns", "compile_ns", "allocation_ns",
                                             "warmup_ns", "build_rss_peak_bytes", "build_wall_ns"}, "initialization")
            for key, value in record["initialization"].items():
                integer(value, 0, 2**63 - 1, key)
            init, bounds = record["initialization"], protocol["resource_bounds"]
            if bounds["max_build_rss_bytes"] is None or bounds["max_build_seconds"] is None:
                raise ValueError("Unresolved build bounds")
            if (init["build_rss_peak_bytes"] > bounds["max_build_rss_bytes"]
                    or init["build_wall_ns"] > bounds["max_build_seconds"] * 10**9):
                raise ValueError("Build resource bound breached")
            if sum(v for k, v in init.items() if k.endswith("_ns") and k != "build_wall_ns") > record["run_end_ns"] - record["run_begin_ns"]:
                raise ValueError("Initialization stage durations exceed lifecycle interval")
            for key in ("handoff", "server_response_id", "timing", "profile", "memory"):
                equal(record[key], None, "initialization." + key)
            equal(record["forwards"], [], "initialization forwards")
            equal(record["client_events"], [], "initialization client")
            equal(record["checks"], None, "initialization correctness")
            summary["initialization"] = init
        else:
            equal(record["initialization"], None, "request initialization isolation")
            forwards = record["forwards"]
            _list(forwards, len(job["spans"]), len(job["spans"]), "full prompt forward count")
            for index, (forward, span) in enumerate(zip(forwards, job["spans"])):
                keys(forward, {"forward_index", "start", "end", "rows", "output_rows", "layer_count",
                               "token_ids_sha256", "complete"}, "forward")
                for key, expected in {"forward_index": index, "start": span[0], "end": span[1],
                                      "rows": span[1] - span[0], "output_rows": 0, "layer_count": 30,
                                      "token_ids_sha256": digest(tokens[job["workload"]][span[0]:span[1]]),
                                      "complete": True}.items():
                    equal(forward[key], expected, "forward." + key)
            _handoff(record["handoff"], job, protocol["cache_layout"])
            if job["mode"] == "correctness":
                checks = record["checks"]
                keys(checks, {"oracle_contract_sha256", "decode_control_commit", "finite_intermediates",
                              "natural_routing", "cache_oracle_match", "logit_oracle_match",
                              "comparison_receipt_sha256"}, "correctness checks")
                for field in ("oracle_contract_sha256", "decode_control_commit"):
                    equal(checks[field], protocol["identities"][field], "correctness identity")
                    sha(checks[field], field, 40 if field.endswith("commit") else 64)
                for field in ("finite_intermediates", "natural_routing", "cache_oracle_match", "logit_oracle_match"):
                    equal(checks[field], True, "correctness." + field)
                equal(checks["comparison_receipt_sha256"], record["handoff"]["comparison_receipt_sha256"], "cache comparison identity")
            else:
                equal(record["checks"], None, "correctness capture stays separate")
            client = parse_client_stream(record["client_events"], job, record["server_response_id"])
            if not record["run_begin_ns"] <= client["first_token_ns"] <= client["done_ns"] <= record["run_end_ns"]:
                raise ValueError("Client clock outside its run")
            if any(not record["run_begin_ns"] <= e["received_ns"] <= record["run_end_ns"]
                   for e in record["client_events"]):
                raise ValueError("Client event clock outside its run")
            summary["client"] = client
            if job["mode"] == "timing":
                timing = record["timing"]
                summary["timing"] = plan_contract.summarize_timing(timing)
                for key, expected in {"prompt_tokens": job["prompt_tokens"], "state": job["state"],
                                      "chunk_rows": [end - start for start, end in job["spans"]],
                                      "first_token_ns": client["first_token_ns"]}.items():
                    equal(timing[key], expected, "timing." + key)
                if timing["request_accept_ns"] < record["run_begin_ns"] or record["trace_bytes"]:
                    raise ValueError("Timing clock/trace isolation changed")
            else:
                equal(record["timing"], None, "instrumented run cannot produce timing evidence")
            for mode in ("profile", "memory"):
                if job["mode"] == mode:
                    summary[mode] = (_profile(record[mode], job, protocol) if mode == "profile"
                                     else _memory(record[mode], job, protocol["resource_bounds"], protocol["cache_layout"]))
                else:
                    equal(record[mode], None, "separate " + mode + " run")
        summaries.append(summary)
    bounds = protocol["resource_bounds"]
    if any(bounds[k] is None for k in ("max_wall_seconds", "max_trace_bytes", "max_requests")):
        raise ValueError("Unresolved suite resource bounds")
    if (previous - begin_suite > bounds["max_wall_seconds"] * 10**9
            or trace_bytes > bounds["max_trace_bytes"] or protocol["request_count"] > bounds["max_requests"]):
        raise ValueError("Suite request/wall/trace bound breached")
    return {"records_consistent": True, "provenance_kind": provenance["kind"],
            "run_count": len(summaries), "request_count": protocol["request_count"],
            "trace_bytes": trace_bytes, "gpu_execution_available": False, "gpu_qualified": False,
            "numerical_qualified": False, "performance_qualified": False,
            "admission_blockers": protocol["blockers"], "runs": summaries}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--runner-binding", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path)
    parser.add_argument("--workload")
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute-gpu", action="store_true", help="always rejected in this CPU implementation")
    args = parser.parse_args(argv)
    try:
        manifest, plan = load_inputs(args.manifest, args.plan, args.source_manifest, args.runner_binding, args.root)
        protocol = compile_protocol(manifest, plan, args.workload, args.chunk, args.receipt_root)
        if args.execute_gpu:
            require_live_execution(protocol)
        if (args.records is None) != (args.prompts is None):
            raise ValueError("Offline record validation requires both --records and --prompts")
        result = (validate_records(protocol, read_records(args.records), read_records(args.prompts))
                  if args.records else protocol)
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x") as output:
                output.write(serialized)
        else:
            print(serialized, end="")
        return 0
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.exit(2, f"prefill-runner: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
