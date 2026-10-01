"""Bounded manual installed preparation probe; run on the existing target environment.

No installation, cache update, model operation or timing is performed. Sources are
copied to an exclusive work directory; the installed runtime is read-only.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

PINS = {
    "data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh": "fd9e2e976496ab318bda6d133d2b68f45b3451a6482978f452acd7a86e029841",
    "data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h": "eca60a5a7f30b70b7a4f426fb833085ee2ae320445fb217fdd8642c9480b23ab",
    "data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h": "8289e5d92e4a8fd04963d58286db5a550cd6fac0cae94919ac4cb34b883152af",
    "data/csrc/nv_internal/tensorrt_llm/kernels/quantization_utils.cuh": "c2a860da4407f70c281c84981db796ec111520d141a3753d77fe66c6c11f3928",
    "data/cutlass/include/cutlass/detail/sm100_blockscaled_layout.hpp": "598e054bef21edf94b1fd6bb1447cfa9cfcf5a5907ab370128102448dbb6d530",
}
MODULE_SHA = "dc26a85431c946b0f636ddd2e08aa676f6340fd085361b21e8b0fe00617d28f9"
NINJA_SHA = "8e01f5b25d3cd211874278a7756f6694e2fa5bb38c6e8a148610cf16aaa3de6f"
TVM_FFI_SHA = "8cb7bdda84545ad7eb5294198bb02fe16930e47e4c3c6de67ab87cad19f6bb7e"
SOURCE_FILES = (
    "probes/m1_installed_capture.cu", "probes/m1_installed_bridge.cuh",
    "probes/m1_host_abi_probe.cpp", "probes/m1_probe_wire.hpp", "kernels/m1_maps_expand.cuh",
    "kernels/m1_installed_preparation.cuh",
    "probes/m1_preparation_flow_test.cpp", "numerical_reference/test_m1_preparation_control_flow.py",
    "numerical_reference/m1_kernel_fixture.py", "numerical_reference/m1_preparation_reference.py",
    "numerical_reference/m1_installed_compare.py", "numerical_reference/m1_abi_probe.py",
)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_flags(ninja):
    variables = {}
    for line in ninja.read_text().replace("$\n", "").splitlines():
        if line.startswith("rule "):
            break
        if "=" in line:
            key, value = line.split("=", 1)
            variables[key.strip()] = value.strip()

    def expand(name, depth=0):
        if depth > 8 or name not in variables:
            raise ValueError("unsupported Ninja variable")
        return re.sub(r"\$([A-Za-z_][A-Za-z_0-9]*)",
                      lambda match: expand(match[1], depth + 1), variables[name])

    flags = shlex.split(expand("cuda_cflags"))
    required = {"--threads=1", "-DENABLE_FP4", "-DENABLE_BF16",
                "-D_GLIBCXX_USE_CXX11_ABI=1", "-gencode=arch=compute_120f,code=sm_120f"}
    if not required.issubset(flags):
        raise ValueError("unexpected installed CUDA profile")
    return expand("nvcc"), flags


def available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("host availability missing")


def gpu():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.free,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=5)
    name, free, used, util = result.strip().split(", ")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True, timeout=5)
    return {"name": name, "free_bytes": int(free) << 20, "used_bytes": int(used) << 20,
            "utilization_percent": int(util), "processes": processes.strip().splitlines()}


def rss(group):
    total = 0
    for p in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = p.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != group and p.parent.name != str(os.getpid()):
                continue
            for line in (p.parent / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            continue
    return total


def run(args):
    source = Path(__file__).resolve().parents[1]
    work = Path(args.output).resolve()
    prior = None
    if args.resume_compiled:
        prior = json.loads((work / "report.json").read_text())
        if (prior["phases"]["compile"]["returncode"] != 0
                or sha(work / "m1_installed_capture") != prior["binary_sha256"]
                or any(sha(work / name) != prior["sources"][name] or sha(source / name) != prior["sources"][name]
                       for name in SOURCE_FILES)):
            raise ValueError("compiled resume identity changed")
        with (work / "report.pre-resume.json").open("x") as f:
            json.dump(prior, f, indent=2)
    else:
        work.mkdir(mode=0o700)
        (work / "temporary").mkdir()
    for name in SOURCE_FILES:
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / name).read_bytes())
    fi, cache = Path(args.flashinfer_root).resolve(), Path(args.cache).resolve()
    module, ninja = cache / "fused_moe_120.so", cache / "build.ninja"
    tvm_ffi = fi.parent / "tvm_ffi/lib/libtvm_ffi.so"
    pins = {fi / name: value for name, value in PINS.items()} | {
        module: MODULE_SHA, ninja: NINJA_SHA, tvm_ffi: TVM_FFI_SHA}
    if any(sha(path) != expected for path, expected in pins.items()):
        raise ValueError("installed source/build identity changed")
    report = {"scope": "installed M1 maps/AQ/SF preparation only", "base_head": args.base_head,
              "compile_limit_seconds": 300, "aggregate_rss_limit_bytes": 2 << 30,
              "minimum_host_available_bytes": 8 << 30, "minimum_gpu_free_bytes": 2 << 30,
              "scratch_limit_bytes": 8 << 20, "installed_pins": {str(p): sha(p) for p in pins},
              "sources": {name: sha(work / name) for name in SOURCE_FILES}, "phases": {}}
    report["coordinator_sha256"] = sha(Path(__file__))
    if prior:
        report["phases"] = prior["phases"]
        report["resumed_compiled_binary"] = True
        report["original_coordinator_sha256"] = prior["coordinator_sha256"]
    report["before"] = gpu()
    if report["before"]["processes"]:
        raise RuntimeError("GPU is owned by another compute process")
    report["packages"] = {name: importlib.metadata.version(name) for name in
                          ("flashinfer-python", "vllm", "torch", "numpy")}
    report["python"] = sys.version.split()[0]
    # Wheel metadata omits CUDA local tags; check Torch's runtime version separately.
    expected_versions = {"flashinfer-python": "0.6.18.post1", "vllm": "0.30.0", "torch": "2.13.0"}
    if report["python"] != "3.12.3" or any(report["packages"][n] != v for n, v in expected_versions.items()):
        raise RuntimeError("installed runtime version changed")

    def save():
        (work / "report.json").write_text(json.dumps(report, indent=2))

    def bounded(command, name, seconds):
        minimum, free = available(), gpu()["free_bytes"]
        if minimum < 8 << 30 or free < 2 << 30:
            raise RuntimeError("headroom before phase")
        start, peak, reason = time.monotonic(), 0, None
        with (work / (name + ".stdout")).open("xb") as out, (work / (name + ".stderr")).open("xb") as err:
            proc = subprocess.Popen(command, cwd=work, stdout=out, stderr=err,
                                    env=os.environ | {"TMPDIR": str(work / "temporary")}, start_new_session=True)
            try:
                sample_at = 0
                while proc.poll() is None:
                    peak = max(peak, rss(proc.pid))
                    minimum = min(minimum, available())
                    now = time.monotonic()
                    if now >= sample_at:
                        free = min(free, gpu()["free_bytes"])
                        sample_at = now + 1
                    if now - start > seconds:
                        reason = "wall_time_limit"
                    elif peak > 2 << 30:
                        reason = "aggregate_rss_limit"
                    elif minimum < 8 << 30 or free < 2 << 30:
                        reason = "headroom_limit"
                    elif out.tell() > 4 << 20 or err.tell() > 4 << 20:
                        reason = "log_size_limit"
                    if reason:
                        break
                    time.sleep(0.1)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=2)
        result = {"command": command, "returncode": proc.returncode, "wall_seconds": time.monotonic() - start,
                  "peak_group_plus_wrapper_rss_bytes": peak, "minimum_host_available_bytes": minimum,
                  "minimum_sampled_gpu_free_bytes": free, "limit_reason": reason}
        report["phases"][name] = result
        save()
        print(json.dumps({"phase": name, **{k: v for k, v in result.items() if k != "command"}}), flush=True)
        if proc.returncode != 0 or reason:
            raise RuntimeError(name + " stopped: " + (work / (name + ".stderr")).read_text()[:5000])

    compiler, flags = installed_flags(ninja)
    compile_command = [compiler, *flags, "--generate-dependencies-with-compile", "-MF", "probe.d",
                       "probes/m1_installed_capture.cu", str(module), str(tvm_ffi),
                       "-Xlinker", "-rpath", "-Xlinker", str(cache),
                       "-Xlinker", "-rpath", "-Xlinker", str(tvm_ffi.parent),
                       "-ldl", "-lcuda", "-o", "m1_installed_capture"]
    try:
        if not prior:
            bounded([sys.executable, "-B", "-c", "import torch; print(torch.__version__)"], "torch_version", 30)
        report["torch_runtime_version"] = (work / "torch_version.stdout").read_text().strip()
        if report["torch_runtime_version"] != "2.13.0+cu130":
            raise RuntimeError("Torch CUDA runtime version changed")
        if not prior:
            bounded([sys.executable, "-B", "numerical_reference/m1_kernel_fixture.py", "create", "fixtures"], "fixtures", 30)
            bounded([sys.executable, "-B", "-m", "unittest", "discover", "-s", "numerical_reference",
                     "-p", "test_m1_preparation_control_flow.py", "-v"], "prepare_control_flow", 30)
            bounded(compile_command, "compile", 300)
        elif prior["phases"]["compile"]["command"] != compile_command:
            raise ValueError("compiled resume command changed")
        report["binary_sha256"] = sha(work / "m1_installed_capture")
        dependencies = shlex.split((work / "probe.d").read_text().replace("\\\n", "").split(":", 1)[1])
        if len(dependencies) > 3000:
            raise ValueError("include graph exceeds cap")
        dep_paths = sorted({(work / p).resolve() for p in dependencies})
        report["dependencies"] = [{"path": str(p), "sha256": sha(p)} for p in dep_paths]
        bounded([str(work / "m1_installed_capture"), "abi", "abi"], "host_abi", 30)
        base = [str(work / "m1_installed_capture"), "capture", str(module), "fixtures"]
        bounded(base + ["capture"], "capture", 60)
        bounded([sys.executable, "-B", "numerical_reference/m1_installed_compare.py", "fixtures", "capture", "abi"], "compare", 30)
        report["compatibility"] = json.loads((work / "compare.stdout").read_text())
        if args.sanitizers:
            for tool in ("memcheck", "initcheck", "racecheck", "synccheck"):
                bounded(["/usr/local/cuda/bin/compute-sanitizer", "--tool", tool, "--error-exitcode", "3",
                         "--log-file", tool + ".sanitizer.log", *base, tool], tool, 60)
                bounded([sys.executable, "-B", "numerical_reference/m1_installed_compare.py", "fixtures", tool, "abi"], tool + "_compare", 30)
    finally:
        report["after"] = gpu()
        report["installed_pins_unchanged"] = all(sha(p) == value for p, value in pins.items())
        report["files"] = {str(p.relative_to(work)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                           for p in work.rglob("*") if p.is_file() and p.name != "report.json" and "temporary" not in p.parts}
        save()
    print(json.dumps({"complete": True, "after": report["after"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-head", required=True)
    parser.add_argument("--sanitizers", action="store_true")
    parser.add_argument("--resume-compiled", action="store_true",
                        help="Resume an unchanged successful build stopped before ABI/capture; preserve its prior report")
    run(parser.parse_args())
