"""Bounded ownership-aware server lifecycle and host-local baseline runner."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import threading
import time

import requests
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from megartx.nvfp4_qualification import activation, natural_coverage
from megartx.m1_execution import synchronous_scheduler_args

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="flashinfer_cutlass")
parser.add_argument("--kv", default="bfloat16")
parser.add_argument("--label", required=True)
parser.add_argument("--trials", type=int, default=30)
parser.add_argument("--prefill-chunk", type=int, default=256)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--m1-decode-profile", action="store_true",
                    help="Untimed four-step CPU/CUDA attribution per 2K lane in the exact one-pair eager pilot")
parser.add_argument("--mode", choices=("native", "reference", "control", "paired_reference", "gate_only_negative_control"), required=True)
parser.add_argument("--client", choices=("quality", "benchmark", "controlled", "normal", "m1-eager-benchmark"), default="quality")
parser.add_argument("--m1-eager-benchmark-plan", type=pathlib.Path)
parser.add_argument("--m1-private-aot", type=pathlib.Path)
parser.add_argument("--m1-timing-metadata-help", action="store_true",
                    help="Enable only the reviewed hash/identity/argv-bound tileiras --help timing distinction")
parser.add_argument("--controlled-plan", type=pathlib.Path)
parser.add_argument("--controlled-path", choices=("full", "cached", "chunked"))
parser.add_argument("--layer0-boundaries", action="store_true")
parser.add_argument("--layer0-capture-policy", choices=("synchronous", "deferred_downstream"), default="synchronous")
parser.add_argument("--activation-only", action="store_true")
parser.add_argument("--routing-diagnostic", action="store_true")
parser.add_argument("--router-score-only", action="store_true")
parser.add_argument("--router-prefix-manifest", type=pathlib.Path)
parser.add_argument("--m1-preparation", choices=("stock", "fused"))
parser.add_argument("--m1-execution", choices=("captured", "capture-free"), default="captured")
parser.add_argument("--m1-bridge", type=pathlib.Path)
parser.add_argument("--m1-build-receipt", type=pathlib.Path)
parser.add_argument("--m1-route-controls", action="store_true")
parser.add_argument("--m1-normal-plan", type=pathlib.Path)
parser.add_argument("--m1-external-observer", action="store_true",
                    help="Enable the perturbing capture-free launch/output sidecar")
args = parser.parse_args()
eager_benchmark = args.client == "m1-eager-benchmark"
benchmark_plan = None
metadata_timing = None
if eager_benchmark:
    if any(os.environ.get(key) for key in ("FLASHINFER_DISABLE_JIT", "FLASHINFER_DISABLE_VERSION_CHECK")):
        parser.error("private AOT requires ordinary FlashInfer version/JIT policy")
    from megartx.m1_eager_benchmark import load_plan
    if (args.m1_eager_benchmark_plan is None or args.m1_private_aot is None or args.m1_preparation not in {"stock", "fused"}
            or args.mode != "native" or args.m1_execution != "capture-free"
            or args.m1_bridge is None or args.m1_build_receipt is None
            or args.profile or args.m1_external_observer or args.m1_route_controls
            or args.layer0_boundaries or args.activation_only or args.routing_diagnostic
            or args.router_score_only or args.router_prefix_manifest is not None
            or args.m1_normal_plan is not None or args.controlled_plan is not None
            or args.controlled_path is not None or args.prefill_chunk != 256
            or args.kv != "bfloat16" or args.backend != "flashinfer_cutlass"):
        parser.error("eager benchmark requires its bounded native capture-free observer-off plan and bridge")
    benchmark_plan = load_plan(args.m1_eager_benchmark_plan)
    if args.m1_decode_profile and (benchmark_plan["trials"] != 1 or benchmark_plan["warmups"] != 1):
        parser.error("decode attribution requires the exact one-pair eager pilot")
    if args.m1_timing_metadata_help != benchmark_plan["metadata_help_timing"]:
        parser.error("metadata timing opt-in differs from the source-bound eager plan")
    from m1_private_aot import MODULE_PINS, sha, validate_cache
    aot = args.m1_private_aot.resolve()
    aot_manifest = validate_cache(aot, benchmark_plan["source_head"])
    aot_cpu = json.loads((aot / "cpu-dry-run.json").read_text())
    if (not aot_cpu.get("passed") or aot_cpu.get("source_head") != benchmark_plan["source_head"]
            or aot_cpu.get("manifest_sha256") != sha(aot / "manifest.json")
            or aot_cpu.get("build_calls") != 0 or aot_cpu.get("cuda_initialized") is not False
            or aot_cpu.get("module_hashes") != MODULE_PINS):
        parser.error("eager benchmark requires an exact-source CUDA-hidden private AOT dry-run")
    project_root = pathlib.Path(__file__).resolve().parents[1]
    from megartx.m1_execution import CONTROLLER_SOURCES
    expected_sources = {n: hashlib.sha256((project_root / "src/megartx" / n).read_bytes()).hexdigest()
                        for n in CONTROLLER_SOURCES}
    if benchmark_plan["controller_source_hashes"] != expected_sources:
        parser.error("eager benchmark controller source differs")
    if any(hashlib.sha256((project_root / p).read_bytes()).hexdigest() != h
           for p, h in benchmark_plan["driver_source_hashes"].items()):
        parser.error("eager benchmark driver source differs")
    build = json.loads(args.m1_build_receipt.read_text())
    if (build.get("base_head") != benchmark_plan["source_head"]
            or build.get("live_contract", {}).get("controller_source_hashes") != expected_sources
            or any(build.get("source_hashes", {}).get(p) != h
                   for p, h in benchmark_plan["driver_source_hashes"].items())):
        parser.error("eager benchmark build/source identity differs")
    if args.m1_timing_metadata_help:
        from m1_owned_processes import MetadataHelpTiming
        metadata_timing = MetadataHelpTiming(aot_manifest["flashinfer_root"], benchmark_plan["source_head"])
elif args.m1_eager_benchmark_plan is not None or args.m1_private_aot is not None or args.m1_timing_metadata_help or args.m1_decode_profile:
    parser.error("eager benchmark plan requires --client m1-eager-benchmark")
normal_plan = None
if args.client == "normal":
    from megartx.m1_normal_plan import load_plan
    if (args.m1_normal_plan is None or args.m1_preparation is None or args.mode != "native"
            or args.profile or args.m1_route_controls or args.layer0_boundaries
            or args.activation_only or args.routing_diagnostic or args.router_score_only
            or args.prefill_chunk != 256 or args.kv != "bfloat16" or args.backend != "flashinfer_cutlass"):
        parser.error("normal M1 requires only its exact approved plan and native eager preparation")
    if args.m1_execution != "captured":
        parser.error("normal M1 plan validation requires captured diagnostics; capture-free is controlled-only")
    normal_plan = load_plan(args.m1_normal_plan)
elif args.m1_normal_plan is not None:
    parser.error("normal M1 plan requires --client normal")
if args.m1_preparation:
    if ((args.client != "controlled" or args.controlled_path != "cached") and args.client != "normal" and not eager_benchmark
            or args.mode != "native"
            or args.m1_bridge is None or args.m1_build_receipt is None or args.layer0_boundaries):
        parser.error("M1 preparation requires the bounded native cached controlled request and built bridge")
    if os.environ.get("LD_PRELOAD"):
        parser.error("owned M1 lifecycle requires no inherited interposition library")
    if args.m1_execution == "capture-free" and args.m1_route_controls:
        parser.error("Capture-free M1 excludes separately artificial diagnostic route controls")
elif args.m1_bridge is not None or args.m1_build_receipt is not None or args.m1_route_controls:
    parser.error("M1 bridge paths require explicit --m1-preparation")
if args.m1_execution == "capture-free" and ((args.client != "controlled" and not eager_benchmark) or args.m1_preparation is None):
    parser.error("capture-free execution is limited to an explicit native controlled stock/fused request")
if args.m1_external_observer and (args.client != "controlled" or args.controlled_path != "cached"
        or args.mode != "native" or args.m1_preparation not in {"stock", "fused"}
        or args.m1_execution != "capture-free" or args.m1_route_controls or args.profile
        or args.layer0_boundaries):
    parser.error("external observer requires one native capture-free controlled cached run without profiling or route controls")
if args.m1_execution != "captured" and not args.m1_preparation:
    parser.error("Capture-free execution requires explicit --m1-preparation stock/fused")
if args.layer0_capture_policy != "synchronous" and not args.layer0_boundaries:
    parser.error("Layer-0 capture policy requires --layer0-boundaries")
if args.layer0_boundaries and (args.client != "controlled" or args.mode != "native" or args.controlled_path not in {"full", "cached"}):
    parser.error("Layer-0 boundaries require only the bounded native full/cached pair")
if args.client == "controlled":
    if args.mode not in {"native", "paired_reference", "gate_only_negative_control"} or args.controlled_plan is None or args.controlled_path is None or args.profile or args.activation_only or args.routing_diagnostic or args.router_score_only or args.router_prefix_manifest is not None:
        parser.error("Controlled capture requires its plan/path and paired lane without other corpus/profile flags")
    if args.prefill_chunk != (16 if args.controlled_path == "chunked" else 256) or args.kv != "bfloat16" or args.backend != "flashinfer_cutlass":
        parser.error("Controlled capture requires the declared chunk, BF16 KV and FlashInfer CUTLASS lane")
elif args.controlled_plan is not None or args.controlled_path is not None or args.mode in {"paired_reference", "gate_only_negative_control"}:
    parser.error("Controlled plan/path/arithmetic lanes require --client controlled")
if args.client == "benchmark":
    parser.error("Timing is fail-closed until a paired quality report and positive natural correction coverage are explicitly validated")
if args.client == "quality" and args.profile:
    parser.error("Quality capture and profiling are separate phases; --profile requires --client benchmark")
if args.activation_only and args.client != "quality":
    parser.error("--activation-only requires the bounded quality client")
if args.router_score_only and (args.mode != "native" or args.activation_only or args.routing_diagnostic or args.prefill_chunk != 256 or args.router_prefix_manifest is None):
    parser.error("--router-score-only requires native mode, a recorded prefix manifest, chunk256, and no other corpus flags")
if args.router_prefix_manifest is not None and not args.router_score_only:
    parser.error("--router-prefix-manifest requires --router-score-only")
base = pathlib.Path(os.environ["MEGARTX_BASE"])
project = pathlib.Path(__file__).resolve().parents[1]
work = pathlib.Path(os.environ["MEGARTX_WORK"])
output = work / "results" / args.label
output.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
env.update({"XDG_CACHE_HOME": str(base / "cache"), "TMPDIR": str(base / "tmp"), "HF_HOME": str(base / "cache/huggingface"), "HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "0", "CUDA_HOME": "/usr/local/cuda", "FLASHINFER_WORKSPACE_BASE": str(base / "cache/flashinfer-workspace"), "TRITON_CACHE_DIR": str(base / "cache/triton"), "CUDA_CACHE_PATH": str(base / "cache/cuda"), "TORCHINDUCTOR_CACHE_DIR": str(base / "cache/torchinductor"), "TORCH_EXTENSIONS_DIR": str(base / "cache/torch-extensions")})
env["PATH"] = "/usr/local/cuda/bin:" + str(base / ".venv/bin") + ":" + env["PATH"]
headers = base / "toolchains/python-headers/usr/include"
env["CPATH"] = str(headers / "python3.12") + ":" + str(headers) + ":" + str(headers / "x86_64-linux-gnu/python3.12")
env["PYTHONPATH"] = os.environ.get("MEGARTX_ADAPTER_SITE", str(work / "adapter-site")) + ":" + str(project / "numerical_reference")
env["VLLM_PLUGINS"] = "megartx_scale_adapter"
env["MEGARTX_SCALE_MODE"] = args.mode
env["MEGARTX_SCALE_MANIFEST"] = str(output / "adapter-manifest.jsonl")
env["MEGARTX_ACTIVATION_PROOF_PATH"] = str(output / "activation-proof.json")
env["MEGARTX_ACTIVATION_TRACE_PATH"] = str(output / "activation-forced.json.gz")
env["MEGARTX_CHECKPOINT_PATH"] = str(base / "models/gemma4-nvfp4")
for inherited in ("MEGARTX_LOGITS_DIR", "MEGARTX_ROUTE_AUDIT_PATH", "MEGARTX_ROUTING_COVERAGE_PATH", "MEGARTX_ROUTER_SCORE_DIR", "MEGARTX_CONTROLLED_DIR", "MEGARTX_CONTROLLED_PLAN", "MEGARTX_LAYER0_BOUNDARIES", "MEGARTX_LAYER0_CAPTURE_POLICY"):
    env.pop(inherited, None)
probe_sha256 = None
env.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
for inherited in ("MEGARTX_M1_PREPARATION", "MEGARTX_M1_EXECUTION", "MEGARTX_M1_BRIDGE",
                  "MEGARTX_M1_BUILD_RECEIPT", "MEGARTX_M1_CAPTURE_DIR", "MEGARTX_M1_STOCK_MODULE",
                  "MEGARTX_M1_ROUTE_CONTROLS", "MEGARTX_M1_NORMAL_PLAN", "MEGARTX_M1_NORMAL_DIR",
                  "MEGARTX_M1_EXTERNAL_OBSERVER_DIR", "MEGARTX_M1_PROCESS_EVIDENCE_DIR",
                  "MEGARTX_M1_PROCESS_EXPECTED_METHOD", "MEGARTX_M1_PROCESS_PROBE_SHA256",
                  "MEGARTX_M1_API_PID_FILE", "MEGARTX_M1_PROCESS_PROBE_ACTIVE"):
    env.pop(inherited, None)
for inherited in ("MEGARTX_M1_EAGER_BENCHMARK_PLAN", "MEGARTX_M1_EAGER_BENCHMARK_DIR", "MEGARTX_M1_DECODE_PROFILE_DIR"):
    env.pop(inherited, None)
env.pop("MEGARTX_M1_PRIVATE_AOT", None)
if args.m1_preparation:
    env.update({"MEGARTX_M1_ROUTE_CONTROLS": "1" if args.m1_route_controls else "0",
                "MEGARTX_M1_PREPARATION": args.m1_preparation,
                "MEGARTX_M1_EXECUTION": args.m1_execution,
                "MEGARTX_M1_BRIDGE": str(args.m1_bridge.resolve()),
                "MEGARTX_M1_BUILD_RECEIPT": str(args.m1_build_receipt.resolve()),
                "MEGARTX_M1_STOCK_MODULE": str(base / "cache/flashinfer-workspace/.cache/flashinfer/0.6.18.post1/120f/cached_ops/fused_moe_120/fused_moe_120.so")})
    if args.m1_execution == "captured":
        env["MEGARTX_M1_CAPTURE_DIR"] = str(output / "preparation")
    if args.m1_external_observer:
        env["MEGARTX_M1_EXTERNAL_OBSERVER_DIR"] = str(output / "m1-external-observer")
        probe = project / "scripts/m1_process_probe/sitecustomize.py"
        process_evidence = output / "m1-process-evidence"
        process_evidence.mkdir(mode=0o700, exist_ok=False)
        probe_sha256 = hashlib.sha256(probe.read_bytes()).hexdigest()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["MEGARTX_M1_PROCESS_EXPECTED_METHOD"] = "spawn"
        env["MEGARTX_M1_PROCESS_EVIDENCE_DIR"] = str(process_evidence)
        env["MEGARTX_M1_PROCESS_PROBE_SHA256"] = probe_sha256
        env["MEGARTX_M1_API_PID_FILE"] = str(output / "owned-server.pid")
        env["PYTHONPATH"] = str(probe.parent) + ":" + env["PYTHONPATH"]
if eager_benchmark:
    env.pop("FLASHINFER_CUDA_ARCH_LIST", None)
    env.pop("FLASHINFER_CUBIN_DIR", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    expected_shims = {"m1_private_aot.py": sha(project / "scripts/m1_private_aot.py"),
                      "sitecustomize.py": sha(project / "scripts/m1_aot_cache/sitecustomize.py"),
                      "flashinfer_jit_cache/__init__.py": sha(project / "scripts/m1_aot_cache/flashinfer_jit_cache/__init__.py")}
    if aot_manifest["shim_hashes"] != expected_shims:
        parser.error("private AOT bootstrap copies differ from source-bound drivers")
    env["MEGARTX_M1_PRIVATE_AOT"] = str(aot)
    env["PYTHONPATH"] = str(aot / "python") + ":" + env["PYTHONPATH"]
    env["MEGARTX_M1_EAGER_BENCHMARK_PLAN"] = str(args.m1_eager_benchmark_plan.resolve())
    env["MEGARTX_M1_EAGER_BENCHMARK_DIR"] = str(output / "eager-benchmark")
    if args.m1_decode_profile:
        env["MEGARTX_M1_DECODE_PROFILE_DIR"] = str(output / "decode-profile")
    (output / "eager-benchmark-plan.json").write_text(json.dumps(benchmark_plan, indent=2))
elif args.client == "normal":
    env["MEGARTX_LOGITS_DIR"] = str(output / "logits")
    env["MEGARTX_M1_NORMAL_DIR"] = str(output / "normal")
    env["MEGARTX_M1_NORMAL_PLAN"] = str(args.m1_normal_plan.resolve())
    (output / "normal-plan.json").write_text(json.dumps(normal_plan, indent=2))
elif args.client == "quality":
    env["MEGARTX_LOGITS_DIR"] = str(output / "logits")
    env["MEGARTX_ROUTE_AUDIT_PATH"] = str(output / "route-hits.jsonl")
    env["MEGARTX_ROUTING_COVERAGE_PATH"] = str(output / "route-histograms.jsonl")
    pathlib.Path(env["MEGARTX_ROUTE_AUDIT_PATH"]).touch(exist_ok=False)
    if args.router_score_only:
        env["MEGARTX_ROUTER_SCORE_DIR"] = str(output / "router-scores")
elif args.client == "controlled":
    env["MEGARTX_LOGITS_DIR"] = str(output / "logits")
    env["MEGARTX_CONTROLLED_DIR"] = str(output / "controlled")
    env["MEGARTX_CONTROLLED_PLAN"] = str(args.controlled_plan.resolve())
    if args.layer0_boundaries:
        env["MEGARTX_LAYER0_BOUNDARIES"] = "1"
        env["MEGARTX_LAYER0_CAPTURE_POLICY"] = args.layer0_capture_policy
    (output / "CONTROLLED-ROUTING.json").write_text(json.dumps({"route_origin": "controlled", "scope": "controlled_routing_fixture", "routing_intervention": True, "routing_unchanged": False, "quality_gate_passed": False, "timing_qualified": False}, indent=2))
else:
    env.pop("MEGARTX_LOGITS_DIR", None)
    env.pop("MEGARTX_ROUTE_AUDIT_PATH", None)
env["MAX_JOBS"] = "2"
env["FLASHINFER_NVCC_THREADS"] = "1"
phases = (output / "server-phases.jsonl").open("a", buffering=1)


def phase(name, **fields):
    row = {"phase": name, "monotonic_ns": time.perf_counter_ns(), "unix_ns": time.time_ns(), "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields}
    phases.write(json.dumps(row) + "\n")
    (output / "status.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)


def gpu_jobs():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise RuntimeError("GPU process query failed: " + result.stderr)
    return [int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit()]


if gpu_jobs():
    phase("blocked_other_gpu_jobs")
    raise SystemExit("Existing GPU compute job present; no server launched")

idle = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.used,memory.free,utilization.gpu,driver_version,power.limit", "--format=csv"], capture_output=True, text=True, timeout=5)
(output / "gpu-before.csv").write_text(idle.stdout)
phase("idle_checked")
stop_sample = threading.Event()
server = None
guard_failure = None
ownership = None
stop_guard = threading.Event()


def require_resources():
    # observe()/cleanup() can latch a failure before the watchdog copies it.
    # The retained ownership state is authoritative at every admission gate.
    failure = ownership.failure if ownership is not None else None
    if failure or guard_failure:
        raise RuntimeError(failure or guard_failure)


def fail_guard(message):
    global guard_failure
    if ownership is not None:
        ownership.fail(message)
        guard_failure = ownership.failure
        ownership.stop()
    else:
        if guard_failure is None:
            guard_failure = message
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)


def compiler_guard():
    from m1_owned_processes import snapshot
    while not stop_guard.is_set():
        try:
            ownership.observe(snapshot(), time.monotonic())
            if ownership.failure:
                fail_guard(ownership.failure)
                return
        except Exception as error:
            fail_guard("Owned compiler telemetry failed: " + str(error))
            return
        stop_guard.wait(.05)


def interrupted(signum, frame):
    raise InterruptedError(f"Owned experiment interrupted by signal {signum}")


for termination_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(termination_signal, interrupted)


def sampler():
    global guard_failure
    fields = "memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem,pstate"
    with (output / "gpu-telemetry.jsonl").open("a", buffering=1) as logfile:
        while not stop_sample.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                values = result.stdout.strip().split(", ")
                logfile.write(json.dumps({"monotonic_ns": time.perf_counter_ns(), "unix_ns": time.time_ns(), "fields": fields.split(","), "values": values, "exit": result.returncode}) + "\n")
                if eager_benchmark and (result.returncode != 0 or len(values) != len(fields.split(","))):
                    raise RuntimeError("eager benchmark GPU headroom telemetry unavailable")
                if result.returncode == 0 and len(values) >= 2 and float(values[1]) < 2048 and server is not None and server.poll() is None:
                    fail_guard("GPU free memory fell below the 2 GiB headroom guard")
                available_kib = int(next(l.split()[1] for l in pathlib.Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:")))
                if available_kib < 8 * 1024 * 1024 and server is not None and server.poll() is None:
                    fail_guard("Host available RAM fell below the 8 GiB headroom guard")
            except Exception as error:
                logfile.write(json.dumps({"error": str(error)}) + "\n")
                if eager_benchmark:
                    try:
                        fail_guard("Eager benchmark resource telemetry failed: " + str(error))
                    except ProcessLookupError:
                        pass
            stop_sample.wait(.2)


sample_thread = threading.Thread(target=sampler, daemon=True)
guard_thread = None
command = [str(base / ".venv/bin/vllm"), "serve", str(base / "models/gemma4-nvfp4"), "--host", "127.0.0.1", "--port", "18000", "--served-model-name", "gemma4-nvfp4", "--dtype", "bfloat16", "--max-model-len", "8448", "--max-num-seqs", "1", "--max-num-batched-tokens", str(args.prefill_chunk), "--gpu-memory-utilization", "0.84", "--kv-cache-memory-bytes", "2147483648", "--kv-cache-dtype", args.kv, "--moe-backend", args.backend, "--attention-backend", "FLASHINFER", "--no-enable-prefix-caching", "--language-model-only", "--generation-config", "vllm", "--seed", "1234", "--stream-interval", "1", "--enforce-eager", "--max-logprobs", "5"]
# vLLM 0.30 enables async scheduling by default for this executor. Every
# admitted M1 lane requires synchronous request/frame identity.
command.extend(synchronous_scheduler_args(args))
if args.profile:
    command += ["--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_dir": str(output / "traces"), "torch_profiler_with_stack": False, "torch_profiler_with_flops": False, "torch_profiler_with_memory": True})]
launch_env_keys = [
    "XDG_CACHE_HOME", "TMPDIR", "HF_HOME", "HF_HUB_OFFLINE", "VLLM_NO_USAGE_STATS",
    "DO_NOT_TRACK", "TOKENIZERS_PARALLELISM", "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CPATH",
    "FLASHINFER_WORKSPACE_BASE", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR", "MAX_JOBS", "FLASHINFER_NVCC_THREADS", "PYTHONPATH", "VLLM_PLUGINS",
    "MEGARTX_SCALE_MODE", "MEGARTX_SCALE_MANIFEST", "MEGARTX_LOGITS_DIR", "MEGARTX_ROUTE_AUDIT_PATH",
    "MEGARTX_ACTIVATION_PROOF_PATH", "MEGARTX_ACTIVATION_TRACE_PATH", "MEGARTX_CHECKPOINT_PATH",
    "MEGARTX_ROUTING_COVERAGE_PATH", "MEGARTX_ROUTER_SCORE_DIR", "MEGARTX_CONTROLLED_DIR",
    "MEGARTX_CONTROLLED_PLAN", "MEGARTX_LAYER0_BOUNDARIES", "MEGARTX_LAYER0_CAPTURE_POLICY",
    "MEGARTX_M1_PREPARATION", "MEGARTX_M1_EXECUTION", "MEGARTX_M1_BRIDGE",
    "MEGARTX_M1_BUILD_RECEIPT", "MEGARTX_M1_CAPTURE_DIR", "MEGARTX_M1_STOCK_MODULE",
    "MEGARTX_M1_ROUTE_CONTROLS", "MEGARTX_M1_NORMAL_PLAN", "MEGARTX_M1_NORMAL_DIR",
    "MEGARTX_M1_EXTERNAL_OBSERVER_DIR",
    "MEGARTX_M1_EAGER_BENCHMARK_PLAN", "MEGARTX_M1_EAGER_BENCHMARK_DIR",
    "MEGARTX_M1_DECODE_PROFILE_DIR",
    "MEGARTX_M1_PRIVATE_AOT",
]
if args.m1_external_observer:
    launch_env_keys.extend(("VLLM_WORKER_MULTIPROC_METHOD",
                            "MEGARTX_M1_PROCESS_EXPECTED_METHOD",
                            "MEGARTX_M1_PROCESS_EVIDENCE_DIR",
                            "MEGARTX_M1_PROCESS_PROBE_SHA256",
                            "MEGARTX_M1_API_PID_FILE"))
launch_manifest = {
    "command": command,
    "environment_overrides": {key: env[key] for key in launch_env_keys if key in env},
    "path_prefixes": ["/usr/local/cuda/bin", str(base / ".venv/bin")],
    "backend_requested": args.backend,
    "kv_requested": args.kv,
    "trust_remote_code": False,
    "trials_per_context": benchmark_plan["trials"] if benchmark_plan else (0 if args.client in {"controlled", "normal"} else args.trials),
    "eager_benchmark_plan_sha256": benchmark_plan["plan_sha256"] if benchmark_plan else None,
    "m1_decode_profile_requested": args.m1_decode_profile,
    "controlled_request_count": 1 if args.client == "controlled" else None,
    "normal_request_count": 2 if normal_plan else None,
    "normal_plan_sha256": normal_plan["plan_sha256"] if normal_plan else None,
    "continuation_constrained": bool(normal_plan),
    "router_score_only": args.router_score_only,
    "controlled_path": args.controlled_path,
    "controlled_plan_sha256": json.loads((args.controlled_plan / "manifest.json").read_text())["schedule_sha256"] if args.controlled_plan else None,
    "minimum_free_memory_mib": 2048,
    "minimum_host_available_ram_gib": 8,
    "owned_startup_containment": "subreaper_pid_start_time_pidfd" if eager_benchmark else None,
    "compiler_aggregate_rss_limit_bytes": 2 << 30 if eager_benchmark else None,
    "shared_compiler_budget_seconds": 300 if eager_benchmark else None,
    "timing_metadata_preflight": metadata_timing.report() if metadata_timing is not None else None,
    "private_aot_manifest_sha256": sha(aot / "manifest.json") if eager_benchmark else None,
    "private_aot_cpu_dry_run_sha256": sha(aot / "cpu-dry-run.json") if eager_benchmark else None,
    "qualification": "Experimental separate-projection original-weight correction; numerical qualification evaluated in separate reports. Eager execution and deterministic finalization are distinct from the original exploratory graph lane.",
    "adapter_mode": args.mode,
    "m1_preparation_requested": args.m1_preparation,
    "m1_execution_requested": args.m1_execution,
    "m1_route_controls": args.m1_route_controls,
    "m1_external_observer_requested": args.m1_external_observer,
    "m1_process_lifecycle_probe_sha256": probe_sha256,
    "m1_process_lifecycle_expected_method": "spawn" if args.m1_external_observer else None,
    "m1_process_lifecycle_policy": "actual_spawn_owner_pre_dispatch" if args.m1_external_observer else None,
    "m1_loader": "after_torch_rtld_global" if args.m1_preparation else None,
}
(output / "launch-manifest.json").write_text(json.dumps(launch_manifest, indent=2))
client = requests.Session()
client.trust_env = False
try:
    if eager_benchmark:
        from m1_owned_processes import OwnedProcesses, enable_subreaper, read_process
        ownership = OwnedProcesses(enable_subreaper(), metadata_timing=metadata_timing)
        available_kib = int(next(l.split()[1] for l in pathlib.Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:")))
        if available_kib < 8 * 1024 * 1024:
            raise RuntimeError("Host available RAM below 8 GiB before launch")
        if idle.returncode != 0 or float(idle.stdout.splitlines()[-1].split(",")[3].strip().split()[0]) < 2048:
            raise RuntimeError("GPU free memory below 2 GiB or preflight unavailable")
    logfile = (output / "server.log").open("w")
    phase("server_launch")
    server = subprocess.Popen(command, env=env, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
    if ownership is not None:
        ownership.register(read_process(server.pid))
        guard_thread = threading.Thread(target=compiler_guard, daemon=True)
        guard_thread.start()
    sample_thread.start()
    (output / "owned-server.pid").write_text(str(server.pid) + "\n")
    deadline = time.monotonic() + 1200
    ready = False
    while time.monotonic() < deadline:
        require_resources()
        if server.poll() is not None:
            raise RuntimeError(f"Server exited before ready (code {server.returncode}); see server.log")
        try:
            response = client.get("http://127.0.0.1:18000/health", timeout=1)
            ready = response.status_code == 200
        except requests.RequestException:
            pass
        if ready:
            break
        time.sleep(2)
    if not ready:
        raise RuntimeError("Server startup exceeded the bounded 20 minute compile/load limit")
    phase("server_ready")
    if args.m1_external_observer:
        from megartx.m1_process_lifecycle import validate_engine_core_registration_before_dispatch
        evidence_dir = output / "m1-process-evidence"
        api_pid_path = output / "owned-server.pid"
        build_report = json.loads(args.m1_build_receipt.read_text())
        bridge_identity = {"path": str(args.m1_bridge.resolve()),
                           "sha256": build_report.get("binary_sha256")}
        if not bridge_identity["sha256"]:
            raise RuntimeError("observer pre-dispatch gate has no source-bound bridge identity")
        lifecycle = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("owned server exited before observer process ownership was verified")
            event_path = evidence_dir / "process-events.jsonl"
            if event_path.is_file() and not event_path.is_symlink():
                lifecycle = validate_engine_core_registration_before_dispatch(
                    evidence_dir, api_pid_path, probe_sha256, bridge_identity)
                if lifecycle is not None:
                    break
            time.sleep(.25)
        if lifecycle is None:
            raise RuntimeError("observer EngineCore ownership was not established before request dispatch")
        lifecycle_record = {"schema": "megartx-m1-pre-dispatch-owner-v1",
                            "passed": True, "before_request_dispatch": True,
                            **lifecycle}
        (evidence_dir / "pre-dispatch.json").write_text(
            json.dumps(lifecycle_record, indent=2) + "\n")
        phase("observer_process_gate_passed", **lifecycle)
    activation(output)
    phase("integration_activation_verified")
    # The process group belongs solely to the server started above. Refuse
    # benchmarks if a different compute job appears; never stop that job.
    other = []
    if ownership is not None:
        from m1_owned_processes import snapshot
        owned_snapshot = snapshot()
        ownership.observe(owned_snapshot, time.monotonic())
    for pid in gpu_jobs():
        try:
            if ((ownership is not None and (pid not in owned_snapshot or owned_snapshot[pid].identity not in ownership.remembered))
                    or ownership is None and os.getpgid(pid) != server.pid):
                other.append(pid)
        except ProcessLookupError:
            pass
    if other:
        raise RuntimeError("Another GPU compute job appeared; benchmark not started")
    require_resources()
    if eager_benchmark:
        bench_command = [str(base / ".venv/bin/python"), str(project / "scripts/m1_eager_benchmark_client.py"),
                         "--plan", str(args.m1_eager_benchmark_plan), "--output", str(output)]
    elif args.client == "normal":
        bench_command = [str(base / ".venv/bin/python"), str(project / "scripts/m1_normal_client.py"), "--plan", str(args.m1_normal_plan), "--output", str(output)]
    elif args.client == "controlled":
        bench_command = [str(base / ".venv/bin/python"), str(project / "scripts/controlled_client.py"), "--plan", str(args.controlled_plan), "--path", args.controlled_path, "--output", str(output)]
    elif args.router_score_only:
        bench_command = [str(base / ".venv/bin/python"), str(project / "scripts/router_score_client.py"), "--prefix-manifest", str(args.router_prefix_manifest), "--output", str(output)]
    else:
        bench_command = [str(base / ".venv/bin/python"), str(project / "scripts/quality_client.py" if args.client == "quality" else project / "scripts/host_benchmark.py"), "--model-path", str(base / "models/gemma4-nvfp4"), "--output", str(output), "--trials", str(args.trials)]
    if args.profile:
        bench_command.append("--profile")
    if args.activation_only:
        bench_command.append("--activation-only")
    if args.routing_diagnostic:
        bench_command.append("--routing-diagnostic")
    phase("client_launch", command=bench_command)
    with (output / "client.log").open("w") as bench_log:
        result = subprocess.run(bench_command, env=env, stdout=bench_log, stderr=subprocess.STDOUT, timeout=3600)
    (output / "benchmark.exit").write_text(str(result.returncode) + "\n")
    require_resources()
    if result.returncode:
        raise RuntimeError("Host-local client failed; see client.log")
    if eager_benchmark:
        phase("bounded_eager_benchmark_complete", plan_sha256=benchmark_plan["plan_sha256"],
              qualified_quality_baseline=False, qualified_performance_baseline=False)
    elif args.client == "normal":
        phase("bounded_normal_m1_capture_complete", request_count=2, expected_live_calls=210, expected_fallback_calls=150, continuation_constrained=True, qualified_quality_baseline=False, qualified_performance_baseline=False)
    elif args.client == "controlled":
        phase("bounded_controlled_execution_complete" if args.m1_execution == "capture-free" else "bounded_controlled_capture_complete",
              m1_execution=args.m1_execution, qualified_quality_baseline=False, qualified_performance_baseline=False)
    elif args.router_score_only:
        try:
            report = natural_coverage(output, args.mode)
            scope = {"natural_coverage": report, "qualified_quality_baseline": False, "qualified_performance_baseline": False}
        except RuntimeError as error:
            scope = {"natural_coverage_blocker": str(error), "qualified_quality_baseline": False, "qualified_performance_baseline": False}
            (output / "QUALIFICATION-BLOCKED.json").write_text(json.dumps(scope, indent=2))
        (output / "router-score-scope.json").write_text(json.dumps(scope, indent=2))
        phase("bounded_router_score_capture_complete", **scope)
    elif args.client == "quality":
        natural_coverage(output, args.mode)
        phase("natural_correction_coverage_verified")
    phase("diagnostic_complete" if args.router_score_only or args.client in {"controlled","normal"} else "benchmark_complete")
    (output / "run.exit").write_text("0\n")
except Exception as error:
    phase("failed", error=str(error))
    (output / "run.exit").write_text("1\n")
    raise
finally:
    for termination_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(termination_signal, signal.SIG_IGN)
    stop_guard.set()
    if guard_thread is not None:
        guard_thread.join(timeout=10)
    stop_sample.set()
    if sample_thread.ident is not None:
        sample_thread.join(timeout=10)
    if ownership is not None:
        ownership_report = ownership.cleanup(gpu_jobs, server.poll if server is not None else lambda: None)
        metadata_final = ownership.finalize_metadata()
        if metadata_final is not None:
            (output / "timing-metadata-final.json").write_text(json.dumps(metadata_final, indent=2))
            ownership_report.update(ownership.report())
        (output / "owned-processes.json").write_text(json.dumps(ownership_report, indent=2))
    if server is not None and ownership is None:
        phase("owned_server_stop")
        # The session/group was created solely for this Popen. Workers can
        # outlive its CLI leader, so cleanup does not depend on leader liveness.
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            server.wait(timeout=10)
        deadline = time.monotonic() + 10
        group_alive = True
        while group_alive and time.monotonic() < deadline:
            try:
                os.killpg(server.pid, 0)
                time.sleep(.2)
            except ProcessLookupError:
                group_alive = False
        if group_alive:
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 10
            while group_alive and time.monotonic() < deadline:
                try:
                    os.killpg(server.pid, 0)
                    time.sleep(.2)
                except ProcessLookupError:
                    group_alive = False
    cleanup_error = None
    cleanup_complete = ownership_report["cleanup_complete"] if ownership is not None else True
    if (args.m1_external_observer or eager_benchmark) and server is not None:
        owned_gpu_pids = []
        if ownership is not None:
            owned_gpu_pids = ownership_report["owned_gpu_pids_remaining"]
            group_alive = bool(ownership_report["owned_identities_remaining"])
            cleanup_error = ownership_report["cleanup_errors"] or None
        else:
            try:
                for pid in gpu_jobs():
                    try:
                        if os.getpgid(pid) == server.pid:
                            owned_gpu_pids.append(pid)
                    except ProcessLookupError:
                        pass
            except BaseException as error:
                cleanup_error = type(error).__name__
            try:
                os.killpg(server.pid, 0)
                group_alive = True
            except ProcessLookupError:
                group_alive = False
        cleanup_complete = not group_alive and not owned_gpu_pids and cleanup_error is None
        cleanup = {"schema": "megartx-m1-process-cleanup-v1",
                   "server_pid": server.pid, "server_returncode": server.returncode,
                   "owned_process_group_alive": group_alive,
                   "owned_gpu_pids_remaining": owned_gpu_pids,
                   "cleanup_query_error": cleanup_error,
                   "cleanup_complete": cleanup_complete}
        if ownership is not None:
            cleanup.update(ownership_report)
        cleanup_path = output / "eager-benchmark-cleanup.json" if eager_benchmark else output / "m1-process-evidence" / "cleanup.json"
        cleanup_path.write_text(
            json.dumps(cleanup, indent=2))
        if not cleanup_complete:
            (output / "run.exit").write_text("1\n")
            phase("cleanup_incomplete", **cleanup)
    phase("cleanup_complete" if cleanup_complete else "cleanup_incomplete")
    phases.close()
    if not cleanup_complete:
        from m1_owned_processes import preserve_primary
        preserve_primary(sys.exc_info()[1], "Owned M1 server cleanup did not complete")
    if eager_benchmark and cleanup_complete and sys.exc_info()[1] is None:
        try:
            require_resources()
            ownership.require_compiler_quiescence()
            if args.m1_decode_profile:
                for lane in ("stock", "fused"):
                    record = json.loads((output / "decode-profile" / (lane + "-scalars.json")).read_text())
                    if (record["lane"] != lane or record["decode_steps"] != 4
                            or record["source_head"] != benchmark_plan["source_head"]
                            or record["plan_sha256"] != benchmark_plan["plan_sha256"]
                            or record["timing_qualified"] is not False
                            or not (output / "decode-profile" / (lane + ".json")).is_file()):
                        raise RuntimeError("decode attribution window/source evidence differs")
            else:
                from m1_eager_benchmark_client import summarize_run
                summarize_run(output)
        except BaseException:
            (output / "run.exit").write_text("1\n")
            raise
