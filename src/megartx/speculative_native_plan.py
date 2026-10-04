"""CPU-only source freeze and admission for the first owned layout receipt.

Reading this module never imports Torch/vLLM or queries devices. A frozen plan
is a proposal, not GPU permission or a scratch-fit certificate.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

from .speculative_native_probe import ADAPTER_FILES, REVISION, ProbeError
from .speculative_native_lifecycle import WORKER_EXTENSION

REVIEWED_LIFECYCLE = "95f31095f94d254b79c6620a046ccdffce712941"
COMPOSED_COLLECTOR_CORRECTION = "b9f8e95e9483daa7be263e3a02a1b6a63ab09844"
COLLECTOR_GUARD = {
    "correction_commit": COMPOSED_COLLECTOR_CORRECTION,
    "receipt_source_sha256": "bcea2a8201529004220b46518df3e5d5e88f7b526e337a7e5ebcd251c897b308",
    "targeted_python_records_bounded": True, "canonical_serialization_bounded": True,
    "native_counter_query_preallocation_bound_bytes": None,
    "review_status": "independent_CPU_review_clear_parent_scope_accepted_GPU_closed",
    "independent_review_reference": "task-5/review-bounded-summary.json; exact b9f8 correction",
    "parent_scope_acceptance_reference": "parent-thread:01a0f166-1f28-7799-9a66-c30c5c758028; prospective monitored native counter query; no hard heap bound or GPU authorization"}
PREFLIGHT_BLOCKERS = ["shared_plugin_registration_and_evidence_hook_not_integrated"]
BASE = Path("/home/yyang/projects/megartx-baseline-20260930")
PYTHON = str(BASE / ".venv/bin/python")
SITE = BASE / ".venv/lib/python3.12/site-packages"
MODEL = str(BASE / "models/gemma4-nvfp4")
PURPOSE = "first_zero_forward_layout_workspace_measurement"
SCHEMA = "megartx-native-receipt-client-plan-v1"
LIMITS = {"incremental_verifier_gpu_bytes": 8 << 20,
          "minimum_gpu_free_bytes": 2 << 30, "minimum_host_free_bytes": 8 << 30,
          "compiler_shared_host_bytes": 2 << 30, "compiler_seconds": 300,
          "wall_seconds_including_cleanup": 1800, "startup_seconds": 900,
          "cleanup_seconds": 60, "receipt_host_metadata_bytes": 8 << 20,
          "plan_bytes": 256 << 10, "authorization_bytes": 64 << 10,
          "private_log_bytes": 8 << 20, "private_run_evidence_bytes": 64 << 20,
          "max_gpu_jobs": 1, "max_fresh_engine_leases": 1,
          "diagnostic_target_forwards": 0, "startup_target_forwards": 11,
          "forced_expert_fixture_pairs": 6}
CLIENT_SOURCES = {
    "vllm/v1/engine/core_client.py": "fd6ca0cb7499b3086eef2fc57674b7ace56ef1c648f09e92e144b255b79018f8",
    "vllm/engine/arg_utils.py": "ead004ca266419cd25f54a7aefc0e20e1eb046f363b7f5cc977e14f52ee44bb1",
    "vllm/v1/executor/abstract.py": "97791426b549e4447c967f8b8e668c2e376c6a75fa7091fd9df11bc9dafb65f7"}
OWNED_FILES = (
    "src/megartx/speculative_native_plan.py", "src/megartx/speculative_native_client.py",
    "src/megartx/speculative_native_evidence.py", "src/megartx/speculative_native_compare.py",
    "scripts/speculative_native_receipt_client.py", "scripts/speculative_native_receipt_preflight.py",
    "scripts/speculative_native_receipt_compare.py", "scripts/m1_owned_processes.py",
    "src/megartx/speculative_native_lifecycle.py", "src/megartx/speculative_native_receipt.py",
    "src/megartx/speculative_native_probe.py", "pyproject.toml",
    "docs/design/speculative-native-zero-forward-protocol.json",
    "docs/design/speculative-native-protocol.json", "docs/evidence/speculative-native-source-binding.json",
    "schemas/speculative-native-receipt-client-plan.schema.json",
    "schemas/speculative-native-receipt-authorization.schema.json",
) + tuple("src/megartx/" + name for name in ADAPTER_FILES)
SHA = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ProbeError("Invalid bounded integer")
    return value


def hash_file(path, cap=None):
    path = Path(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not path.is_file() or (cap is not None and before.st_size > cap):
            raise ProbeError("File is missing, nonregular or oversized")
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
        after = os.fstat(stream.fileno())
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if fields(before) != fields(after) or fields(after) != fields(path.stat()):
        raise ProbeError("Source/evidence changed during hashing")
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def object_digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProbeError("Duplicate evidence key")
        result[key] = value
    return result


def read_json(path, cap):
    # Bound before reading, and cap the read independently against growth races.
    with Path(path).open("rb") as stream:
        if os.fstat(stream.fileno()).st_size > cap:
            raise ProbeError("Evidence exceeds its frozen bound")
        data = stream.read(cap + 1)
    if len(data) > cap:
        raise ProbeError("Evidence grew beyond its frozen bound")
    try:
        return json.loads(data, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ProbeError("Nonfinite evidence")))
    except (ValueError, RecursionError) as error:
        raise ProbeError("Malformed JSON evidence") from error


def engine_kwargs():
    # Exact EngineArgs fields read from the installed pinned source. No serving
    # frontend, renderer, prompt, sampler or target-probe entry point is made.
    return {"model": MODEL, "revision": REVISION, "tokenizer_revision": REVISION,
            "dtype": "bfloat16", "max_model_len": 8448, "max_num_seqs": 1,
            "max_num_batched_tokens": 256, "gpu_memory_utilization": 0.84,
            "kv_cache_memory_bytes": 2 << 30, "kv_cache_dtype": "bfloat16",
            "moe_backend": "flashinfer_cutlass", "attention_backend": "FLASHINFER",
            "enable_prefix_caching": False, "language_model_only": True,
            "generation_config": "vllm", "seed": 1234, "enforce_eager": True,
            "worker_extension_cls": WORKER_EXTENSION, "async_scheduling": False,
            "enable_dbo": False, "distributed_executor_backend": "uni",
            "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
            "data_parallel_size": 1, "trust_remote_code": False}


def engine_argv():
    result = []
    for key, value in engine_kwargs().items():
        flag = "--" + key.replace("_", "-")
        if type(value) is bool:
            # Only BooleanOptionalAction fields have a --no-* spelling.
            # False store_true defaults are omitted; exact kwargs are also frozen.
            if value:
                result.append(flag)
            elif key in ("enable_prefix_caching", "async_scheduling"):
                result.append("--no-" + key.replace("_", "-"))
        else:
            result.extend((flag, str(value)))
    return result


def environment(project, private):
    project, private = Path(project).resolve(), Path(private).resolve()
    cache = BASE / "cache"
    return {"MEGARTX_NATIVE_DIAGNOSTIC": "1", "MEGARTX_NATIVE_RECEIPT_EVIDENCE": "1",
            "MEGARTX_SCALE_MODE": "native", "VLLM_PLUGINS": "megartx_scale_adapter",
            "MEGARTX_CHECKPOINT_PATH": MODEL,
            "MEGARTX_SCALE_MANIFEST": str(private / "adapter-manifest.jsonl"),
            "MEGARTX_ACTIVATION_PROOF_PATH": str(private / "activation-proof.json"),
            "MEGARTX_ACTIVATION_TRACE_PATH": str(private / "activation-forced.json.gz"),
            "PYTHONPATH": str(project / "src") + ":" + str(project / "numerical_reference"),
            "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": str(cache / "huggingface"), "XDG_CACHE_HOME": str(cache),
            "TMPDIR": str(BASE / "tmp"), "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
            "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "MAX_JOBS": "2", "FLASHINFER_NVCC_THREADS": "1",
            "CUDA_HOME": "/usr/local/cuda", "CPATH": str(BASE / "toolchains/python-headers/usr/include/python3.12")
            + ":" + str(BASE / "toolchains/python-headers/usr/include") + ":"
            + str(BASE / "toolchains/python-headers/usr/include/x86_64-linux-gnu/python3.12"),
            "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer-workspace"),
            "TRITON_CACHE_DIR": str(cache / "triton"), "CUDA_CACHE_PATH": str(cache / "cuda"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"), "TORCH_EXTENSIONS_DIR": str(cache / "torch-extensions")}


def _git(project, *args):
    return subprocess.check_output(["git", "-C", str(project), *args], text=True).strip()


def freeze(project, *, client_mode="async"):
    project = Path(project).resolve()
    if client_mode not in ("sync", "async"):
        raise ProbeError("Client mode must be sync or async")
    if _git(project, "status", "--porcelain", "--untracked-files=normal"):
        raise ProbeError("Freeze requires a clean committed source tree")
    head = _git(project, "rev-parse", "HEAD")
    subprocess.run(["git", "-C", str(project), "merge-base", "--is-ancestor", REVIEWED_LIFECYCLE, head], check=True)
    subprocess.run(["git", "-C", str(project), "merge-base", "--is-ancestor", COMPOSED_COLLECTOR_CORRECTION, head], check=True)
    if hash_file(project / "src/megartx/speculative_native_receipt.py", 1 << 20) != COLLECTOR_GUARD["receipt_source_sha256"]:
        raise ProbeError("Composed exact collector correction source differs")
    result = {"schema": SCHEMA, "purpose": PURPOSE, "reviewed_lifecycle_commit": REVIEWED_LIFECYCLE,
              "source_head": head, "source_sha256": {p: hash_file(project / p, 1 << 20) for p in OWNED_FILES},
              "installed_client_sources": CLIENT_SOURCES, "checkpoint_revision": REVISION,
              "python": PYTHON, "installed_root": str(SITE), "model": MODEL,
              "client_mode": client_mode, "engine_kwargs": engine_kwargs(), "engine_argv": engine_argv(),
              "limits": LIMITS, "gpu_authorized": False, "target_probe_authorized": False,
              "collector_materialization_guard": COLLECTOR_GUARD,
              "preflight_blockers": PREFLIGHT_BLOCKERS}
    result["plan_sha256"] = object_digest(result)
    return result


def validate_plan(plan, project):
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA or plan.get("purpose") != PURPOSE:
        raise ProbeError("Wrong plan schema or purpose")
    if set(plan) != {"schema", "purpose", "reviewed_lifecycle_commit", "source_head", "source_sha256",
                    "installed_client_sources", "checkpoint_revision", "python", "installed_root", "model",
                    "client_mode", "engine_kwargs", "engine_argv", "limits", "gpu_authorized",
                    "target_probe_authorized", "collector_materialization_guard", "preflight_blockers", "plan_sha256"}:
        raise ProbeError("Unexpected plan field")
    expected = freeze(project, client_mode=plan["client_mode"])
    if plan != expected:
        raise ProbeError("Plan differs from exact committed source/configuration")
    return plan


def installed_preflight(root):
    from .speculative_native_probe import inspect_sources
    from .speculative_native_receipt import FFI_SOURCES, TORCH_MEMORY_SOURCE_SHA256
    result = inspect_sources(root)
    for path, expected in {**CLIENT_SOURCES, **FFI_SOURCES, "torch/cuda/memory.py": TORCH_MEMORY_SOURCE_SHA256}.items():
        actual = hash_file(Path(root) / path, 1 << 20)
        if actual != expected:
            raise ProbeError("Installed source changed: " + path)
        result[path] = actual
    # Metadata distribution inspection reads text only, no imported packages.
    from importlib.metadata import distributions
    versions = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in distributions(path=[str(root)])}
    for name, expected in (("vllm", "0.30.0"), ("torch", "2.13.0+cu130"), ("flashinfer-python", "0.6.18.post1")):
        if versions.get(name) != expected:
            raise ProbeError("Installed package metadata differs: " + name)
    return {"source_sha256": result, "packages": {k: versions[k] for k in ("vllm", "torch", "flashinfer-python")},
            "runtime_imports": False, "device_queries": False}


def checkpoint_preflight(manifest_path, model=MODEL):
    manifest = read_json(manifest_path, 64 << 10)
    if manifest.get("repo_id") != "nvidia/Gemma-4-26B-A4B-NVFP4" or manifest.get("revision") != REVISION:
        raise ProbeError("Wrong immutable checkpoint manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 12:
        raise ProbeError("Checkpoint manifest topology differs")
    names = set()
    identities = {}
    for row in files:
        name = row.get("name")
        if not isinstance(name, str) or Path(name).name != name or name in names or not SHA.fullmatch(str(row.get("sha256", ""))):
            raise ProbeError("Malformed checkpoint source")
        names.add(name)
        path = Path(model) / name
        if not path.is_file() or path.stat().st_size != integer(row.get("bytes"), 1):
            raise ProbeError("Checkpoint file size differs")
        if name.endswith(".safetensors") and row.get("verified_upstream_lfs_sha256") is not True:
            raise ProbeError("Unverified checkpoint shard")
        # Explicit full streaming rehash: no model parse or package import.
        if hash_file(path) != row["sha256"]:
            raise ProbeError("Checkpoint file hash differs")
        stat = path.stat()
        identities[name] = {"sha256": row["sha256"], "bytes": stat.st_size,
                            "stat": [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]}
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
    if not required <= names:
        raise ProbeError("Checkpoint/tokenizer/template evidence missing")
    return {"checkpoint_revision": REVISION, "manifest_sha256": hash_file(manifest_path, 64 << 10),
            "files": identities, "full_shards_rehashed": True, "tokenizer_template_verified": True}


def validate_authorization(auth, plan):
    required = {"schema": "megartx-native-receipt-authorization-v1", "purpose": PURPOSE,
                "plan_sha256": plan["plan_sha256"], "source_head": plan["source_head"],
                "independent_review_clear": True, "exact_head_ci_green": True,
                "parent_source_protocol_accepted": True, "gpu_slot_assigned": True,
                "owned_lifecycle_verified": True}
    if not isinstance(auth, dict) or any(type(auth.get(k)) is not type(v) or auth.get(k) != v for k, v in required.items()):
        raise ProbeError("Missing exact-source review/CI/parent GPU-slot authorization")
    refs = ("parent_slot", "parent_acceptance_reference", "independent_review_reference", "ci_reference")
    if set(auth) != set(required) | set(refs):
        raise ProbeError("Unexpected authorization field")
    for key in refs:
        if not isinstance(auth.get(key), str) or not 1 <= len(auth[key]) <= 512:
            raise ProbeError("Missing bounded authorization evidence reference")
    if plan["preflight_blockers"] or plan["collector_materialization_guard"] is None:
        raise ProbeError("Source preflight remains blocked; authorization cannot override unresolved collection bounds")
    return auth


def receipt_admission(plan, auth, checkpoint, *, deadline):
    validate_authorization(auth, plan)
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ProbeError("Nonfinite monotonic deadline")
    return {"phase": "zero_forward_receipt", "independent_review_clear": True,
            "parent_source_protocol_accepted": True, "owned_lifecycle_verified": True,
            "max_concurrent_gpu_jobs": 1, "max_extra_gpu_bytes": 8 << 20,
            "compiler_memory_bytes": 2 << 30, "compiler_timeout_seconds": 300, "compiler_scope": "shared_host",
            "checkpoint_revision": REVISION, "checkpoint_manifest_verified": checkpoint["full_shards_rehashed"] is True,
            "tokenizer_template_verified": checkpoint["tokenizer_template_verified"] is True,
            "implementation_binding": {Path(p).name: plan["source_sha256"][p] for p in OWNED_FILES
                                       if Path(p).name in ("speculative_native_lifecycle.py", "speculative_native_receipt.py", "speculative_native_probe.py")},
            "zero_protocol_sha256": plan["source_sha256"]["docs/design/speculative-native-zero-forward-protocol.json"],
            "adapter_source_sha256": {p: plan["source_sha256"]["src/megartx/" + p] for p in ADAPTER_FILES},
            "deadline_monotonic": deadline, "parent_slot": auth["parent_slot"],
            "client_purpose": PURPOSE, "client_plan_sha256": plan["plan_sha256"], "source_head": plan["source_head"]}
