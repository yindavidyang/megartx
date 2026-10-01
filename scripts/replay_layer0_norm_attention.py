#!/usr/bin/env python3
"""Predeclared operator-only replay of the retained layer-0 norm/attention pair.

No model/server/checkpoint load, prompt sweep, training, or timing claim. Three
repeats per fixed case: original M33/M32/M1 norms and RoPE; both attention
providers crossed with the full/cached position-32 inputs; zero-query/constant-V
controls. Artifacts include private operands/results and a CUDA trace. Source
and BF16 flags are checked before initializing CUDA; subprocess exit frees all
GPU allocations. Prior runtime, captures, and source files are never modified.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "numerical_reference"))
import layer0_reference as layer0
import layer0_norm_attention_reference as contract


def source_guards():
    expected = {
        "vllm.kernels.vllm_c": "ef37e71736807026f0b8ad187155b63f147984c03d287a78790ef8f8ed642ec1",
        "flashinfer.decode": "d82d107a644596a9349780b839b34e690c50169ea4cea3d0ee02b78e9dc88c1a",
        "flashinfer.prefill": "2ad12a8387b3f6bff192e5945769b68d90cfcab00b9eb2a8a524af8dd71a29de",
        "flashinfer.xqa": "6a3f7330980b976201cff0fcdf51e2958e454c4c5951ad2b30c1f2aa13722d33",
    }
    records = {}
    for module, digest in expected.items():
        path = Path(importlib.util.find_spec(module).origin)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest and actual != digest:
            raise RuntimeError("Installed source differs from the declared lane: " + module)
        records[module] = actual
    extension = Path(importlib.util.find_spec("vllm").origin).parent / "_C_stable_libtorch.abi3.so"
    actual = hashlib.sha256(extension.read_bytes()).hexdigest()
    if actual != "7a283666e16f66348351ed0e877425016b09e1986a2a1086b0aaad08dbadd101":
        raise RuntimeError("Pinned vLLM CUDA extension changed")
    records[extension.name] = actual
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--cached", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Replay output must be fresh; never replace retained evidence")
    plan = layer0.evidence.load_plan(args.plan)
    cases = {"full": layer0.load_case(args.full, "full", plan), "cached": layer0.load_case(args.cached, "cached", plan)}
    for case in cases.values():
        contract.validate_binding(case)
    trace_profiles = {path: contract.bind_attention_trace(root, path)
                      for path, root in (("full", args.full), ("cached", args.cached))}
    for key in cases["full"]["constants"]:
        if not np.array_equal(cases["full"]["constants"][key], cases["cached"]["constants"][key]):
            raise ValueError("Paired constants changed")
    sources = source_guards()
    import torch
    if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0":
        raise RuntimeError("Pinned Torch/CUDA lane changed")
    if not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction or torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("Original BF16 reduction/TF32 flags changed")
    import vllm._custom_ops as ops
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    from flashinfer.decode import xqa_batch_decode_with_kv_cache
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 0):
        raise RuntimeError("Only the pinned SM120 device is allowed")
    args.output.mkdir(parents=True)
    arrays, reports = {}, {}
    workspace = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")

    def device(bits):
        return torch.from_numpy(bits.copy()).view(torch.bfloat16).to("cuda")

    def bits(tensor):
        return tensor.detach().contiguous().view(torch.uint16).cpu().numpy().copy()

    def fixed_repeats(label, call):
        outputs = []
        for repetition in range(3):
            with torch.profiler.record_function("megartx.norm_attention." + label + ".r" + str(repetition)):
                outputs.append(bits(call()))
        if any(not np.array_equal(outputs[0], output) for output in outputs[1:]):
            raise RuntimeError("Fixed operator repeat changed: " + label)
        for i, output in enumerate(outputs):
            arrays[label + "_r" + str(i)] = output
        return outputs[0]

    def paged_cache(case, position, control=False):
        prefix = case["prefixes"][-1]
        count = position + 1
        slots = prefix["cache_slots"][:count]
        if not np.array_equal(slots % 16, np.arange(count) % 16):
            raise RuntimeError("Retained prefix is not the declared page-aligned causal sequence")
        original_pages = [int(slots[i] // 16) for i in range(0, count, 16)]
        if len(set(original_pages)) != len(original_pages):
            raise RuntimeError("Prefix pages alias")
        # Compact only physical page IDs; logical positions and within-page
        # offsets stay exact. Preserve the model's interleaved K/V strides.
        cache = torch.zeros((len(original_pages), 8, 16, 512), dtype=torch.bfloat16, device="cuda")
        k = device(prefix["stored_prefix_k"][:count].reshape(count, 8, 256))
        v = device(prefix["stored_prefix_v"][:count].reshape(count, 8, 256))
        for i in range(count):
            cache[i // 16, :, i % 16, :256] = k[i]
            cache[i // 16, :, i % 16, 256:] = v[i]
        if control:
            cache[..., :256].zero_()
            cache[..., 256:].fill_(1)
        return cache[..., :256], cache[..., 256:]

    def prefill(case, count, positions, label, control=False):
        q = torch.zeros((count, 16, 256), dtype=torch.bfloat16, device="cuda")
        for p in positions:
            q[p] = device(case["rows"][p]["attention_q"].reshape(16, 256))
        if control:
            q.zero_()
        kv = paged_cache(case, count - 1, control)
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="HND", backend="fa2")
        pages = (count + 15) // 16
        wrapper.plan(torch.tensor([0, count], dtype=torch.int32), torch.tensor([0, pages], dtype=torch.int32),
                     torch.arange(pages, dtype=torch.int32), torch.tensor([(count - 1) % 16 + 1], dtype=torch.int32),
                     16, 8, 256, 16, causal=True, pos_encoding_mode="NONE", sm_scale=1.0,
                     window_left=1023, logits_soft_cap=0.0, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        return fixed_repeats(label, lambda: wrapper.run(q, kv)[positions])

    def decode(case, label, control=False):
        q = device(case["rows"][32]["attention_q"].reshape(1, 16, 256))
        if control:
            q.zero_()
        kv = paged_cache(case, 32, control)
        # XQA derives scheduling capacity from page_table.shape[-1], not the
        # nominal max_seq_len argument. Preserve the model's 8448-token table
        # capacity so the original 21-subsequence multi-block branch is used.
        # Only its first three page entries belong to the actual 33-key prefix.
        table = torch.zeros((1, 8448 // 16), dtype=torch.int32, device="cuda")
        table[0, :3] = torch.arange(3, dtype=torch.int32, device="cuda")
        lengths = torch.tensor([33], dtype=torch.int32, device="cuda")
        return fixed_repeats(label, lambda: xqa_batch_decode_with_kv_cache(
            q, kv, workspace, table, lengths, 33, bmm1_scale=1.0, bmm2_scale=1.0,
            window_left=1023, kv_layout="HND", q_len_per_req=1))

    with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for path, count, positions in (("full", 33, [31, 32]), ("cached", 32, [31]), ("cached", 1, [32])):
            case = cases[path]
            label = path + "_m" + str(count)
            for name in contract.NORM_NAMES:
                width = 256 if name in {"q_norm", "k_norm", "v_norm"} else 2816
                source = np.stack([case["rows"][p][name + "_in"] for p in positions])
                indices = positions if count != 1 else [0]
                if width == 256:
                    storage = torch.zeros((count, 8192), dtype=torch.bfloat16, device="cuda")
                    offset = {"q_norm": 0, "k_norm": 4096, "v_norm": 6144}[name]
                    storage[indices, offset:offset + source.shape[1]] = device(source)
                    x = storage[:, offset:offset + source.shape[1]].unflatten(-1, (-1, 256))
                else:
                    x = torch.zeros((count, width), dtype=torch.bfloat16, device="cuda")
                    x[indices] = device(source)
                weight = None if name == "v_norm" else device(case["constants"][name + "_weight_bits"])
                out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
                def norm_call():
                    torch.ops._C.rms_norm(out, x, weight, 1e-6)
                    return out[indices].reshape(len(indices), -1)
                output = fixed_repeats(label + "_" + name, norm_call)
                expected = np.stack([case["rows"][p][name + "_out"] for p in positions])
                reports[label + "_" + name] = layer0.bit_difference(output, expected)
                if not reports[label + "_" + name]["raw_bits_equal"]:
                    raise RuntimeError("Native RMS did not reproduce the retained boundary: " + label + name)
            q = torch.zeros((count, 4096), dtype=torch.bfloat16, device="cuda")
            k = torch.zeros((count, 2048), dtype=torch.bfloat16, device="cuda")
            cache = torch.zeros((33, 256), dtype=torch.bfloat16, device="cuda")
            for i, p in zip(indices, positions):
                q[i] = device(case["rows"][p]["rope_q_in"])
                k[i] = device(case["rows"][p]["rope_k_in"])
                cache[p] = device(case["rows"][p]["rope_cache_bits"])
            position_tensor = torch.tensor(list(range(count)) if count != 1 else [32], device="cuda", dtype=torch.int64)
            def rope_call():
                qr, kr = q.clone(), k.clone()
                ops.rotary_embedding(position_tensor, qr, kr, 256, cache, True)
                return torch.cat((qr[indices], kr[indices]), dim=-1)
            output = fixed_repeats(label + "_rope", rope_call)
            expected = np.stack([np.concatenate((case["rows"][p]["rope_q_out"], case["rows"][p]["rope_k_out"])) for p in positions])
            reports[label + "_rope"] = layer0.bit_difference(output, expected)
            if not reports[label + "_rope"]["raw_bits_equal"]:
                raise RuntimeError("Native RoPE did not reproduce retained boundaries: " + label)

        full = prefill(cases["full"], 33, [31, 32], "full_fa2")
        cached_prefill = prefill(cases["cached"], 32, [31], "cached_fa2_m32")
        cached = decode(cases["cached"], "cached_xqa")
        for label, output, case, positions in (("full_fa2", full, cases["full"], [31, 32]),
                                             ("cached_fa2_m32", cached_prefill, cases["cached"], [31]),
                                             ("cached_xqa", cached, cases["cached"], [32])):
            expected = np.stack([case["rows"][p]["attention_out"].reshape(16, 256) for p in positions])
            reports[label] = layer0.bit_difference(output, expected)
            if not reports[label]["raw_bits_equal"]:
                raise RuntimeError("Attention did not reproduce the retained provider output: " + label)
        crossed_fa2 = prefill(cases["cached"], 33, [32], "cached_inputs_fa2")
        crossed_xqa = decode(cases["full"], "full_inputs_xqa")
        reports["same_full_inputs_provider_difference"] = layer0.bit_difference(full[1:], crossed_xqa)
        reports["same_cached_inputs_provider_difference"] = layer0.bit_difference(crossed_fa2, cached)
        reports["fa2_input_sensitivity"] = layer0.bit_difference(full[1:], crossed_fa2)
        reports["xqa_input_sensitivity"] = layer0.bit_difference(crossed_xqa, cached)
        for provider in ("fa2", "xqa"):
            control = (prefill(cases["full"], 33, [32], "constant_fa2", True) if provider == "fa2"
                       else decode(cases["full"], "constant_xqa", True))
            expected = np.full(control.shape, 0x3f80, dtype=np.uint16)
            reports["constant_" + provider] = layer0.bit_difference(control, expected)
            if not reports["constant_" + provider]["raw_bits_equal"]:
                raise RuntimeError("Exact constant attention control failed")

    torch.cuda.synchronize()
    trace = args.output / "trace.json"
    profiler.export_chrome_trace(str(trace))
    with trace.open("rb") as source, gzip.open(trace.with_suffix(".json.gz"), "wb") as target:
        target.write(source.read())
    trace.unlink()
    payload = args.output / "results.npz"
    np.savez_compressed(payload, **arrays)
    report = {"schema": 1, "repeats": 3, "cases": reports, "sources": sources, "retained_dispatch": trace_profiles,
              "results_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
              "trace_sha256": hashlib.sha256(trace.with_suffix(".json.gz").read_bytes()).hexdigest(),
              "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": props.name,
              "workspace_bytes": 128 << 20, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "xqa_table_capacity_tokens": 8448,
              "model_loaded": False, "timing_qualified": False, "whole_model_accepted": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"retained_boundaries_reproduced": True, "constant_controls_pass": True,
                      "same_full_inputs_provider_difference": reports["same_full_inputs_provider_difference"],
                      "same_cached_inputs_provider_difference": reports["same_cached_inputs_provider_difference"],
                      "peak_allocated_bytes": report["peak_allocated_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
