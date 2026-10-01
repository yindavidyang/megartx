"""Bounded ownership-aware server lifecycle and host-local baseline runner."""
import argparse
import datetime
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

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="flashinfer_cutlass")
parser.add_argument("--kv", default="bfloat16")
parser.add_argument("--label", required=True)
parser.add_argument("--trials", type=int, default=30)
parser.add_argument("--prefill-chunk", type=int, default=256)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--mode", choices=("native", "reference", "control"), required=True)
parser.add_argument("--client", choices=("quality", "benchmark"), default="quality")
parser.add_argument("--activation-only", action="store_true")
parser.add_argument("--routing-diagnostic", action="store_true")
args = parser.parse_args()
if args.client == "benchmark":
    parser.error("Timing is fail-closed until a paired quality report and positive natural correction coverage are explicitly validated")
if args.client == "quality" and args.profile:
    parser.error("Quality capture and profiling are separate phases; --profile requires --client benchmark")
if args.activation_only and args.client != "quality":
    parser.error("--activation-only requires the bounded quality client")
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
env["PYTHONPATH"] = os.environ.get("MEGARTX_ADAPTER_SITE", str(work / "adapter-site"))
env["VLLM_PLUGINS"] = "megartx_scale_adapter"
env["MEGARTX_SCALE_MODE"] = args.mode
env["MEGARTX_SCALE_MANIFEST"] = str(output / "adapter-manifest.jsonl")
env["MEGARTX_ACTIVATION_PROOF_PATH"] = str(output / "activation-proof.json")
env["MEGARTX_ACTIVATION_TRACE_PATH"] = str(output / "activation-forced.json.gz")
env["MEGARTX_CHECKPOINT_PATH"] = str(base / "models/gemma4-nvfp4")
if args.client == "quality":
    env["MEGARTX_LOGITS_DIR"] = str(output / "logits")
    env["MEGARTX_ROUTE_AUDIT_PATH"] = str(output / "route-hits.jsonl")
    env["MEGARTX_ROUTING_COVERAGE_PATH"] = str(output / "route-histograms.jsonl")
    pathlib.Path(env["MEGARTX_ROUTE_AUDIT_PATH"]).touch(exist_ok=False)
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
                if result.returncode == 0 and len(values) >= 2 and float(values[1]) < 2048 and server is not None and server.poll() is None:
                    guard_failure = "GPU free memory fell below the 2 GiB headroom guard; stopping only the owned test server"
                    os.killpg(server.pid, signal.SIGTERM)
                available_kib = int(next(l.split()[1] for l in pathlib.Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:")))
                if available_kib < 8 * 1024 * 1024 and server is not None and server.poll() is None:
                    guard_failure = "Host available RAM fell below the 8 GiB headroom guard; stopping only the owned test server"
                    os.killpg(server.pid, signal.SIGTERM)
            except Exception as error:
                logfile.write(json.dumps({"error": str(error)}) + "\n")
            stop_sample.wait(.2)


sample_thread = threading.Thread(target=sampler, daemon=True)
sample_thread.start()
command = [str(base / ".venv/bin/vllm"), "serve", str(base / "models/gemma4-nvfp4"), "--host", "127.0.0.1", "--port", "18000", "--served-model-name", "gemma4-nvfp4", "--dtype", "bfloat16", "--max-model-len", "8448", "--max-num-seqs", "1", "--max-num-batched-tokens", str(args.prefill_chunk), "--gpu-memory-utilization", "0.84", "--kv-cache-memory-bytes", "2147483648", "--kv-cache-dtype", args.kv, "--moe-backend", args.backend, "--attention-backend", "FLASHINFER", "--no-enable-prefix-caching", "--language-model-only", "--generation-config", "vllm", "--seed", "1234", "--stream-interval", "1", "--enforce-eager", "--max-logprobs", "5"]
if args.profile:
    command += ["--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_dir": str(output / "traces"), "torch_profiler_with_stack": False, "torch_profiler_with_flops": False, "torch_profiler_with_memory": True})]
(output / "launch-manifest.json").write_text(json.dumps({"command": command, "environment_overrides": {k: env[k] for k in ["XDG_CACHE_HOME", "TMPDIR", "HF_HOME", "HF_HUB_OFFLINE", "VLLM_NO_USAGE_STATS", "DO_NOT_TRACK", "TOKENIZERS_PARALLELISM", "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CPATH", "FLASHINFER_WORKSPACE_BASE", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "MAX_JOBS", "FLASHINFER_NVCC_THREADS", "PYTHONPATH", "VLLM_PLUGINS", "MEGARTX_SCALE_MODE", "MEGARTX_SCALE_MANIFEST", "MEGARTX_LOGITS_DIR", "MEGARTX_ROUTE_AUDIT_PATH", "MEGARTX_ACTIVATION_PROOF_PATH", "MEGARTX_ACTIVATION_TRACE_PATH", "MEGARTX_CHECKPOINT_PATH", "MEGARTX_ROUTING_COVERAGE_PATH"] if k in env}, "path_prefixes": ["/usr/local/cuda/bin", str(base / ".venv/bin")], "backend_requested": args.backend, "kv_requested": args.kv, "trust_remote_code": False, "trials_per_context": args.trials, "minimum_free_memory_mib": 2048, "minimum_host_available_ram_gib": 8, "qualification": "Experimental separate-projection original-weight correction; numerical qualification evaluated in separate reports. Eager execution and deterministic finalization are distinct from the original exploratory graph lane.", "adapter_mode": args.mode}, indent=2))
client = requests.Session()
client.trust_env = False
try:
    logfile = (output / "server.log").open("w")
    phase("server_launch")
    server = subprocess.Popen(command, env=env, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
    (output / "owned-server.pid").write_text(str(server.pid) + "\n")
    deadline = time.monotonic() + 1200
    ready = False
    while time.monotonic() < deadline:
        if guard_failure:
            raise RuntimeError(guard_failure)
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
    activation(output)
    phase("integration_activation_verified")
    # The process group belongs solely to the server started above. Refuse
    # benchmarks if a different compute job appears; never stop that job.
    other = []
    for pid in gpu_jobs():
        try:
            if os.getpgid(pid) != server.pid:
                other.append(pid)
        except ProcessLookupError:
            pass
    if other:
        raise RuntimeError("Another GPU compute job appeared; benchmark not started")
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
    if result.returncode:
        raise RuntimeError("Host-local client failed; see client.log")
    if args.client == "quality":
        natural_coverage(output, args.mode)
        phase("natural_correction_coverage_verified")
    phase("benchmark_complete")
    (output / "run.exit").write_text("0\n")
except Exception as error:
    phase("failed", error=guard_failure or str(error))
    (output / "run.exit").write_text("1\n")
    raise
finally:
    for termination_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(termination_signal, signal.SIG_IGN)
    if server is not None:
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
    stop_sample.set()
    sample_thread.join(timeout=10)
    phase("cleanup_complete")
    phases.close()
