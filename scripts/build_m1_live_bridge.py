"""Build the task-local live bridge under the existing probe resource bounds."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_m1_installed_probe import PINS, MODULE_SHA, NINJA_SHA, TVM_FFI_SHA
from run_m1_installed_probe import sha, installed_flags, available, gpu, rss


# Source-bound scheduler, physical SF TMA domain and exact typed FFI lane.
LIVE_CONSUMER_PINS = {'data/cutlass/include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp': 'e01bcc4eb6ae05ecbee7251514519c3d7b8e9b37ae4e955f58f7c85591e2df9e', 'data/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp': '8dd4fcdd5706e6c8c6111e71791c113e34f137c106eb5c514685069dc6ebac44', 'data/cutlass/include/cutlass/gemm/kernel/tile_scheduler.hpp': 'acc90548b9e2b19f944764ced57e1459d5c2ed7e118d6a1af476add26c3d5e73', 'data/cutlass/include/cutlass/gemm/kernel/tile_scheduler_params.h': 'ef48a12e8920183e88259d0b685279c2232fc2fb12c4fb4db7e8d0fbfdc019e9', 'data/cutlass/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl': 'c81e6473efc15a07ac5707febd2a8db69edd2949afd2af64fb87cfb020632989', 'data/cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp': '66fcea9bab8db40e22201d4a78f2de9777c616c15d800716d8446172fbcd9824', 'data/csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu': '9588117b6f8d6431dd19935bdffd428f56b8de938f3f4335cf9b6f96d6ef80a5', 'data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu': '2aa95ebe6fb2f4f45c09fba824df18d18fbe95f9950ac6e37507f4432a077a9d'}


def control_commands(python, work, module):
    """Build argv without encoding conversions or shell/path splitting."""
    return ([str(python), "-B", str(work / "scripts/check_m1_live_bridge.py"),
             str(work / "m1_live_bridge.so"), str(work / "lease-controls.json")],
            [str(python), "-B", str(work / "scripts/check_m1_live_bindings.py"),
             str(work / "m1_live_bridge.so"), str(module), str(work / "binding-controls.json")])


def build(args):
    root = Path(__file__).resolve().parents[1]
    fi, cache = args.flashinfer_root.resolve(), args.cache.resolve()
    module, ninja = cache / "fused_moe_120.so", cache / "build.ninja"
    ffi = fi.parent / "tvm_ffi/lib/libtvm_ffi.so"
    cuda_runtime = fi.parent / "nvidia/cu13/lib/libcudart.so.13"
    pins = {fi / p: digest for p, digest in (PINS | LIVE_CONSUMER_PINS).items()} | {
        module: MODULE_SHA, ninja: NINJA_SHA, ffi: TVM_FFI_SHA,
        cuda_runtime: sha(cuda_runtime)}
    if any(sha(p) != digest for p, digest in pins.items()):
        raise RuntimeError("installed identity changed")
    expected = {"vllm": "0.30.0", "flashinfer-python": "0.6.18.post1", "torch": "2.13.0"}
    if sys.version.split()[0] != "3.12.3" or any(importlib.metadata.version(p) != v for p, v in expected.items()):
        raise RuntimeError("installed package/Python pin changed")
    work = args.output.resolve()
    work.mkdir(mode=0o700)
    (work / "temporary").mkdir()
    sources = ("probes/m1_live_bridge.cu", "probes/m1_installed_bridge.cuh",
               "kernels/m1_installed_preparation.cuh", "kernels/m1_maps_expand.cuh",
               "scripts/build_m1_live_bridge.py", "scripts/check_m1_live_bridge.py",
               "scripts/check_m1_live_bindings.py",
               "src/megartx/m1_live.py", "src/megartx/vllm_scale_plugin.py",
               "src/megartx/m1_execution.py", "src/megartx/controlled_capture.py",
               "src/megartx/controlled_kv_capture.py")
    for name in sources:
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    before = gpu()
    if before["processes"] or before["free_bytes"] < 2 << 30 or available() < 8 << 30:
        raise RuntimeError("ownership/headroom preflight failed")
    compiler, flags = installed_flags(ninja)
    names = subprocess.check_output(["nm", "-D", str(module)], text=True)
    prefix = "_ZN12tensorrt_llm7kernels15cutlass_kernels18CutlassMoeFCRunnerI13__nv_fp4_e2m1S3_13__nv_bfloat16S3_S4_Lb0ELNS1_21Sm90Wfp4Afp8ScaleModeE0EvE"
    setup = [line.split()[-1] for line in names.splitlines()
             if line.split() and line.split()[-1].startswith(prefix)
             and "setupTmaWarpSpecializedInputs" in line]
    if len(setup) != 1:
        raise RuntimeError("typed TMA setup symbol identity missing or ambiguous")
    hooks = {setup[0]}
    for marker in ("6runMoe",):
        matches = [line.split()[-1] for line in names.splitlines()
                   if len(line.split()) >= 3 and line.split()[-1].startswith(prefix)
                   and marker in line.split()[-1]]
        if len(matches) != 1:
            raise RuntimeError("typed runner symbol identity missing or ambiguous")
        hooks.update(matches)
    for marker in ("fusedBuildExpertMapsSortFirstTokenEPKi",
                   "expandInputRowsKernelLauncherI13__nv_fp4_e2m1S3_E"):
        matches = [line.split()[-1] for line in names.splitlines()
                   if len(line.split()) >= 3 and line.split()[-1].startswith("_ZN")
                   and marker in line.split()[-1]]
        if len(matches) != 1:
            raise RuntimeError("preparation symbol identity missing or ambiguous")
        hooks.update(matches)
    relocations = []
    for line in subprocess.check_output(["objdump", "-R", str(module)], text=True).splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2].split("@")[0] in hooks:
            if fields[1] not in {"R_X86_64_64", "R_X86_64_JUMP_SLOT"}:
                raise RuntimeError("unqualified native relocation kind")
            relocations.append({"symbol": fields[2].split("@")[0], "offset": int(fields[0], 16),
                                "kind": fields[1]})
    if len(relocations) != 4 or {r["symbol"] for r in relocations} != hooks:
        raise RuntimeError("pinned native hook relocation set changed")
    header = '#define M1_TMA_SETUP_SYMBOL "' + setup[0] + '"\n'
    header += 'struct M1LiveHookRelocation { char const* symbol; uintptr_t offset; };\n'
    header += 'constexpr M1LiveHookRelocation m1_live_hook_relocations[] = {\n'
    header += ''.join('{"' + r["symbol"] + '", ' + hex(r["offset"]) + '},\n' for r in relocations)
    live_contract = {"abi_version": 2, "view_count": 15, "view_bytes": 32,
                     "controller_source_hashes": {name: sha(work / "src/megartx" / name)
                          for name in ("m1_live.py", "vllm_scale_plugin.py", "m1_execution.py",
                                       "controlled_capture.py", "controlled_kv_capture.py")},
                     "native_source_sha256": sha(work / "probes/m1_live_bridge.cu"),
                     "execution_modes": ["captured", "capture-free"],
                     "capture_free_begin": "megartx_m1_begin_capture_free_v2"}
    header += '};\n#define M1_LIVE_CONTRACT_JSON ' + json.dumps(json.dumps(live_contract, sort_keys=True)) + '\n'
    (work / "m1_live_symbols.h").write_text(header)
    command = [compiler, *flags, "--shared", "--cudart=shared", "--generate-dependencies-with-compile",
               "-I" + str(work), "-MF", str(work / "bridge.d"), str(work / "probes/m1_live_bridge.cu"),
               str(module), str(ffi), "-Xlinker", "-rpath", "-Xlinker", str(cache),
               "-Xlinker", "-rpath", "-Xlinker", str(cuda_runtime.parent),
               "-Xlinker", "-rpath", "-Xlinker", str(ffi.parent), "-ldl", "-lcuda",
               "-o", str(work / "m1_live_bridge.so")]
    report = {"base_head": args.base_head, "command": command,
              "limits": {"compiler_rss_bytes": 2 << 30, "compile_seconds": 300,
                         "host_available_bytes": 8 << 30, "gpu_free_bytes": 2 << 30,
                         "additional_device_workspace_bytes": 8 << 20},
              "live_contract": live_contract, "source_hashes": {p: sha(work / p) for p in sources}, "hook_relocations": relocations,
              "installed_pins": {str(p): digest for p, digest in pins.items()}, "before": before}
    start, peak, reason = time.monotonic(), 0, None
    with (work / "compile.stdout").open("xb") as out, (work / "compile.stderr").open("xb") as err:
        proc = subprocess.Popen(command, cwd=work, stdout=out, stderr=err, start_new_session=True,
                                env=os.environ | {"TMPDIR": str(work / "temporary")})
        try:
            while proc.poll() is None:
                peak = max(peak, rss(proc.pid))
                if peak > 2 << 30 or available() < 8 << 30 or time.monotonic() - start > 300:
                    reason = "compiler resource bound";break
                time.sleep(.05)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL);proc.wait(timeout=5)
    report.update(returncode=proc.returncode, reason=reason, seconds=time.monotonic()-start,
                  peak_aggregate_rss_bytes=peak, installed_pins_unchanged=all(sha(p)==v for p,v in pins.items()),
                  after=gpu())
    if proc.returncode == 0 and not reason:
        report["binary_sha256"] = sha(work / "m1_live_bridge.so")
        exported = subprocess.check_output(["nm", "-D", str(work / "m1_live_bridge.so")], text=True)
        required = {"megartx_m1_begin_v2", "megartx_m1_contract_v2", "megartx_m1_end", "megartx_m1_active",
                    "megartx_m1_begin_capture_free_v2", "megartx_m1_error", "megartx_m1_metadata",
                    "megartx_m1_verify_bindings"} | hooks
        actual = {line.split()[-1] for line in exported.splitlines() if len(line.split()) >= 3}
        if (not required.issubset(actual) or "megartx_m1_begin" in actual
                or any(name.startswith("cuda") for name in actual)):
            report["reason"] = reason = "C ABI export missing or private CUDA runtime interposed"
        report["required_exports_present"] = required.issubset(actual)
        if not reason:
            with (work / "lease-controls.stdout").open("xb") as out, (work / "lease-controls.stderr").open("xb") as err:
                controls = subprocess.run(control_commands(sys.executable, work, module)[0],
                                          stdout=out, stderr=err, timeout=30)
            report["compiled_lease_controls_returncode"] = controls.returncode
            if controls.returncode:
                report["reason"] = reason = "compiled lease controls failed"
        if not reason:
            with (work / "binding-controls.stdout").open("xb") as out, (work / "binding-controls.stderr").open("xb") as err:
                controls = subprocess.run(control_commands(sys.executable, work, module)[1],
                                          stdout=out, stderr=err, timeout=30)
            report["compiled_binding_controls_returncode"] = controls.returncode
            if controls.returncode:
                report["reason"] = reason = "compiled late-load binding controls failed"
    (work / "build.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k in (
        "returncode", "reason", "seconds", "peak_aggregate_rss_bytes", "binary_sha256", "installed_pins_unchanged")}),flush=True)
    if proc.returncode or reason:
        raise RuntimeError("bounded bridge build failed: " + (work / "compile.stderr").read_text()[-7000:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-head", required=True)
    build(parser.parse_args())
