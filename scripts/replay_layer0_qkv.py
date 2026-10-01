"""Bounded operator-only QKV replay; no model/server, tuning or timing gate.

Uses the same frozen 33 inputs, original layer-0 BF16 Q/K/V, and retained
boundaries. Three repeats per M1/M33/profile plus exact real-geometry controls
are fixed before execution. Process-local backend flags are restored.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "numerical_reference"))
import compare_controlled_capture as evidence
import layer0_reference as reference


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gpu_guard():
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, timeout=5, check=True)
    if len(result.stdout.splitlines()) != 1 or int(result.stdout.strip()) < 2048:
        raise RuntimeError("The bounded replay requires one GPU and at least 2 GiB free")
    available = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    if available < 8 * 1024 * 1024:
        raise RuntimeError("The bounded replay requires 8 GiB available host memory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "plan", "full", "cached", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.checkpoint.resolve(), args.plan.resolve(), args.full.resolve(), args.cached.resolve()]
    output = args.output.resolve()
    if any(output == path or path in output.parents for path in inputs):
        raise ValueError("New replay evidence must be outside every immutable input")
    jobs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                          text=True, capture_output=True, timeout=5, check=True)
    if any(line.strip().isdigit() for line in jobs.stdout.splitlines()):
        raise RuntimeError("An existing GPU compute process blocks the replay")
    gpu_guard()
    plan = evidence.load_plan(args.plan)
    if sha(args.checkpoint / "config.json") != evidence.reference.CONFIG_SHA256:
        raise ValueError("Original checkpoint config changed")
    cases = [reference.load_case(root, path, plan) for root, path in ((args.full, "full"), (args.cached, "cached"))]
    for stage in ("input_norm_in", "qkv_in"):
        if any(not np.array_equal(cases[0]["rows"][p][stage], cases[1]["rows"][p][stage]) for p in (31, 32)):
            raise ValueError("Retained QKV operands differ before the replay")
    embeddings, input_hashes = reference.original_layer0_inputs(args.checkpoint, plan["tokens"])
    if not np.array_equal(embeddings["input_norm"], cases[0]["constants"]["input_norm_weight_bits"]):
        raise ValueError("Loaded input norm differs from the original checkpoint")
    weights, weight_hashes = [], {}
    for field in ("q", "k", "v"):
        bits, checksum = reference.original_bf16_projection(args.checkpoint, field)
        if field in {"k", "v"} and checksum != cases[0]["binding"]["loaded_" + field + "_sha256"]:
            raise ValueError("Retained loaded projection differs from original bytes")
        weights.append(bits.copy()); weight_hashes[field] = checksum
    import torch
    import vllm._custom_ops  # Registers the unchanged vllm_c RMS CUDA op.
    import vllm.model_executor.models.gemma4 as gemma
    if sha(inspect.getsourcefile(gemma)) != evidence.KV_SOURCE_HASHES[gemma.__name__]:
        raise ValueError("Pinned embedding/decoder source changed")
    if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0" or tuple(torch.cuda.get_device_capability()) != (12, 0):
        raise ValueError("QKV replay requires the recorded Torch/CUDA/SM120 runtime")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"scope": "bounded_operator_replay", "checkpoint_revision": evidence.REVISION,
                "token_sha256": plan["token_sha256"], "schedule_sha256": plan["schedule_sha256"],
                "original_weight_sha256": weight_hashes, "original_input_sha256": input_hashes,
                "source_sha256": sha(__file__), "cpu_oracle_sha256": sha(reference.__file__),
                "historical_binding_sha256": [c["binding_sha256"] for c in cases],
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "shapes": {"m1": [1, 2816, 8192], "m33": [33, 2816, 8192]}, "repetitions": 3,
                "profiles": ["original_reduced_precision_allowed", "reduced_precision_disallowed"],
                "controls": ["all_zero", "one_term_exact_6_or_minus_8"],
                "native_accumulation_qualified": False, "full_cached_handoff_accepted": False,
                "quality_gate_passed": False, "timing_qualified": False}
    (output / "predeclared-contract.json").write_text(json.dumps(manifest, indent=2))

    def cuda_bf16(bits):
        return torch.from_numpy(bits.copy()).view(torch.bfloat16).to("cuda")

    def storage(tensor):
        return tensor.detach().contiguous().view(torch.uint16).cpu().numpy().copy()

    with torch.inference_mode():
        incoming = cuda_bf16(embeddings["embedding"]) * torch.tensor(2816 ** 0.5, dtype=torch.bfloat16, device="cuda")
        norm_weight = cuda_bf16(embeddings["input_norm"])
        normalized = torch.empty_like(incoming)
        torch.ops._C.rms_norm(normalized, incoming, norm_weight, 1e-6)
        actual_in, actual_norm = storage(incoming), storage(normalized)
        if any(not np.array_equal(actual_in[p], cases[0]["rows"][p]["input_norm_in"]) or
               not np.array_equal(actual_norm[p], cases[0]["rows"][p]["qkv_in"]) for p in (31, 32)):
            raise RuntimeError("Reconstructed embedding/RMS operands do not reproduce the retained rows")
        combined = cuda_bf16(np.concatenate(weights, axis=0))
        arrays = {"input_bits": actual_norm}
        old_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        if old_reduction is not True:
            raise RuntimeError("Original process-local BF16 preference changed")
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
                for profile, enabled in (("original", True), ("disallowed", False)):
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = enabled
                    for m in (1, 33):
                        x = normalized[32:33].contiguous() if m == 1 else normalized
                        for repetition in range(3):
                            with torch.profiler.record_function(f"megartx.qkv_replay.{profile}.m{m}.repeat{repetition}"):
                                result = torch.nn.functional.linear(x, combined)
                            arrays[f"{profile}_m{m}_repeat{repetition}"] = storage(result[[-1]] if m == 1 else result[[31, 32]])
                        control_x = torch.zeros_like(x)
                        control_w = torch.zeros_like(combined)
                        with torch.profiler.record_function(f"megartx.qkv_replay.{profile}.m{m}.zero_control"):
                            zero = storage(torch.nn.functional.linear(control_x, control_w))
                        if np.any(zero != 0):
                            raise RuntimeError("Exact all-zero real-geometry control failed")
                        control_x[:, 0] = 2
                        control_w[::2, 0], control_w[1::2, 0] = 3, -4
                        with torch.profiler.record_function(f"megartx.qkv_replay.{profile}.m{m}.one_term_control"):
                            one_term = storage(torch.nn.functional.linear(control_x, control_w))
                        expected = np.tile(np.array([0x40C0, 0xC100], dtype=np.uint16), (m, 4096))
                        if not np.array_equal(one_term, expected):
                            raise RuntimeError("Exact one-term real-geometry control failed")
                        del control_x, control_w
                        gpu_guard()
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(output / "operator-trace.json"))
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduction
        manifest["process_local_flags_restored"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == old_reduction
        manifest["reconstructed_retained_inputs_exact"] = True
        manifest["exact_real_geometry_controls_passed"] = True
        with (output / "operands-and-results.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        manifest["arrays_sha256"] = sha(output / "operands-and-results.npz")
        manifest["trace_sha256"] = sha(output / "operator-trace.json")
        report = {"profiles": {}, "captured_reproduction": {}, "contract": "Unfitted conditional F32-RNE diagnostics and exact controls; native acceptance withheld"}
        for profile in ("original", "disallowed"):
            stable = {str(m): all(np.array_equal(arrays[f"{profile}_m{m}_repeat0"], arrays[f"{profile}_m{m}_repeat{r}"]) for r in (1, 2)) for m in (1, 33)}
            report["profiles"][profile] = {"within_shape_three_repeats_bit_exact": stable,
                "m1_vs_m33_last_row": reference.bit_difference(arrays[f"{profile}_m1_repeat0"], arrays[f"{profile}_m33_repeat0"][1:2]), "projections": {}}
            for field, weight, section in zip(("q", "k", "v"), weights, (slice(0,4096), slice(4096,6144), slice(6144,8192))):
                report["profiles"][profile]["projections"][field] = {
                    str(m): reference.projection_diagnostic(actual_norm[[32]] if m == 1 else actual_norm[[31,32]], weight,
                        arrays[f"{profile}_m{m}_repeat0"][:, section])[0] for m in (1,33)}
        report["captured_reproduction"] = {
            "full_selected_rows": reference.bit_difference(arrays["original_m33_repeat0"], np.stack([cases[0]["rows"][p]["qkv_out"] for p in (31,32)])),
            "cached_decode_row": reference.bit_difference(arrays["original_m1_repeat0"], cases[1]["rows"][32]["qkv_out"][None,:])}
        manifest["report"] = report
        with (output / "replay-report.json").open("x") as stream:
            json.dump(manifest, stream, indent=2)
    print(json.dumps({"completed": True, "repeats": 3, "controls_passed": True,
                      "captured_reproduction": report["captured_reproduction"],
                      "profiles": {p: {k: v for k,v in item.items() if k != "projections"} for p,item in report["profiles"].items()},
                      "native_accumulation_qualified": False, "timing_qualified": False}), flush=True)


if __name__ == "__main__":
    main()
