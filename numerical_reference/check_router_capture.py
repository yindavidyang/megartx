"""Read-only, bounded CPU checker for one observed router request.

Writes only the explicitly requested report. Original captures are never edited.
All numerical code comes from the independent sibling router_reference module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np

import router_reference as oracle


TARGETS = {0: [42, 82], 1: [126], 2: [89], 3: [7], 5: [12]}
MAX_NPZ_BYTES = 16 * 1024 * 1024
REFERENCE_FIELDS = {"weight_bits", "dimension_scale_bits", "expert_scale_bits", "root_raw_bits", "root_cast_bf16_bits"}
FORWARD_FIELDS = {"positions", "token_ids", "logits_f32", "selected_ids", "selected_weights_f32", "checked_indices", "residual_bits", "norm_bits", "projection_bits"}


def read_json(path: Path):
    if path.stat().st_size > MAX_NPZ_BYTES:
        raise ValueError("JSON evidence exceeds the bounded read size")
    return json.loads(path.read_text())


def read_npz(directory: Path, name: str, expected_digest: str, fields: set[str]):
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".npz"):
        raise ValueError("NPZ evidence must be a local basename")
    if not isinstance(expected_digest, str) or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
        raise ValueError("Missing or invalid NPZ SHA256")
    path = directory / name
    if path.stat().st_size > MAX_NPZ_BYTES:
        raise ValueError("NPZ evidence exceeds the bounded archive size")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
        raise ValueError("NPZ evidence SHA256 differs")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if {entry.filename for entry in entries} != {field + ".npy" for field in fields} or len(entries) != len(fields):
            raise ValueError("Unexpected or duplicate NPZ members")
        if sum(entry.file_size for entry in entries) > MAX_NPZ_BYTES:
            raise ValueError("NPZ expanded evidence exceeds the bounded read size")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("Object arrays are not evidence")
    return arrays


def validate_request(requests):
    if not isinstance(requests, list) or len(requests) != 1:
        raise ValueError("Router diagnostic must contain exactly one client request")
    request = requests[0]
    prompt, outputs = request["prompt_token_ids"], request["completion_token_ids"]
    if len(prompt) != 1025 or not isinstance(outputs, list) or not 1 <= len(outputs) <= 8:
        raise ValueError("Request exceeds the exact 1025-prefix/eight-output scope")
    if any(type(token) is not int or not 0 <= token < 262144 for token in prompt + outputs):
        raise ValueError("Client tokens are outside the frozen vocabulary")
    if hashlib.sha256(json.dumps(prompt).encode()).hexdigest() != request["prompt_sha256"]:
        raise ValueError("Client prefix SHA256 differs from actual token IDs")
    if request["usage"]["prompt_tokens"] != len(prompt) or request["usage"]["completion_tokens"] != len(outputs):
        raise ValueError("Client usage differs from actual token count")
    return request


def validate_reference(metadata, arrays):
    index = metadata["layer"]
    if index not in TARGETS or metadata["expert_ids"] != TARGETS[index]:
        raise ValueError("Original reference targets differ from the six frozen IDs")
    if metadata["loader_ordinal"] != index:
        raise ValueError("Loaded layer ordinal differs from the pinned literal layer")
    if metadata["projection_lane"] != "torch.mm(BF16,BF16,out_dtype=float32)":
        raise ValueError("Unsupported projection cast lane")
    flags = metadata["projection_flags"]
    if set(flags) != {"allow_specialized_router_gemm", "allow_fp32_router_gemm", "allow_bf16x3_router_gemm", "allow_ll_bf16_gemm", "allow_cublas_router_gemm"}:
        raise ValueError("Incomplete projection dispatch metadata")
    if flags["allow_cublas_router_gemm"] is not True or any(flags[key] is not False for key in flags if key != "allow_cublas_router_gemm"):
        raise ValueError("Projection does not use the pinned BF16 cuBLAS/F32 lane")
    if metadata["norm_eps"] != 1e-6 or any(metadata[key] is not False for key in ("norm_has_weight", "norm_pass_weight", "norm_pass_weight_add")) or metadata["norm_variance_size_override"] is not None:
        raise ValueError("Unsupported RMS norm contract")
    tensor_fields = {
        f"model.language_model.layers.{index}.router.proj.weight": ("weight_bits", (128, 2816)),
        f"model.language_model.layers.{index}.router.scale": ("dimension_scale_bits", (2816,)),
        f"model.language_model.layers.{index}.router.per_expert_scale": ("expert_scale_bits", (128,)),
    }
    original = metadata["original_tensors"]
    if len(original) != 3 or {row["tensor"] for row in original} != set(tensor_fields):
        raise ValueError("Incomplete or duplicate original router tensor identities")
    for row in original:
        label, shape = tensor_fields[row["tensor"]]
        bits = np.asarray(arrays[label])
        if bits.shape != shape or bits.dtype.kind != "u" or bits.dtype.itemsize != 2 or row["shape"] != list(shape) or row["dtype"] != "torch.bfloat16":
            raise ValueError("Original BF16 router tensor shape/dtype differs")
        oracle.bf16_decode(bits)
        payload = bits.astype("<u2", copy=False).tobytes(order="C")
        if row["bytes"] != len(payload) or hashlib.sha256(payload).hexdigest() != row["sha256"] or row["loaded_original_bytes_equal"] is not True:
            raise ValueError("Original router tensor hash/loaded-byte proof differs")
    raw = np.asarray(arrays["root_raw_bits"])
    root_cast = np.asarray(arrays["root_cast_bf16_bits"])
    if metadata["root_storage_shape"] not in ([], [1]) or raw.ndim != 1 or raw.dtype != np.uint8 or root_cast.shape != (1,):
        raise ValueError("Root storage shape/dtype differs")
    if metadata["root_storage_dtype"] == "torch.bfloat16" and len(raw) == 2:
        expected_cast = np.frombuffer(raw.tobytes(), dtype="<u2")
    elif metadata["root_storage_dtype"] == "torch.float32" and len(raw) == 4:
        expected_cast = oracle.bf16_rne_bits(np.frombuffer(raw.tobytes(), dtype="<f4"))
    else:
        raise ValueError("Unsupported captured root storage dtype")
    if not np.array_equal(expected_cast, root_cast) or float(oracle.bf16_decode(root_cast)[0]) <= 0:
        raise ValueError("Root castBF16 differs from its actual raw storage")
    expected_root = oracle.bf16_rne_bits(np.asarray([2816 ** -0.5], dtype=np.float32))
    return {
        "original_tensor_payloads_match_manifest_sha256": True,
        "loaded_original_byte_equality": "Producer assertion retained; source files were not reread by this CPU checker",
        "root_cast_matches_storage_bits": True,
        "root_value_matches_source_constant_f32_then_bf16": bool(np.array_equal(root_cast, expected_root)),
        "root_storage_dtype": metadata["root_storage_dtype"],
        "root_bf16_value": float(oracle.bf16_decode(root_cast)[0]),
    }


def _summary(target, records, row_metrics):
    projection = [r["projection_profile"] for r in records if r["projection_profile"] is not None]
    rms = [r["rms_semantic_diagnostic"] for r in records if r["rms_semantic_diagnostic"] is not None]
    values = sum(r["outputs_compared"] for r in projection)
    weights = sum(r["scoring"]["weight_values_compared"] for r in records)
    norms = sum(r["values_compared"] for r in rms)
    targets = []
    for expert in target["expert_ids"]:
        observations = [(row, t) for row in row_metrics for t in row["targets"] if t["expert"] == expert]
        closest_row, closest = max(observations, key=lambda pair: pair[1]["target_minus_kth_logit"])
        conditional = [t for r in records for checked in r["checked_row_rank_envelopes"] for t in checked["targets"] if t["expert"] == expert]
        targets.append({
            "expert": expert,
            "selected_count": sum(int(t["selected"]) for _, t in observations),
            "captured_bit_rank_minimum": min(t["pinned_bit_rank"] for _, t in observations),
            "captured_bit_rank_maximum": max(t["pinned_bit_rank"] for _, t in observations),
            "maximum_target_minus_8th_logit": closest["target_minus_kth_logit"],
            "closest_position": closest_row["position"], "closest_phase": closest_row["phase"],
            "minimum_target_minus_8th_logit": min(t["target_minus_kth_logit"] for _, t in observations),
            "semantic_full_probability_minimum": min(t["semantic_full_probability_f64"] for _, t in observations),
            "semantic_full_probability_maximum": max(t["semantic_full_probability_f64"] for _, t in observations),
            "numeric_tie_observations": sum(int(t["numeric_tie_minimum_rank"] != t["numeric_tie_maximum_rank"]) for _, t in observations),
            "conditional_checked_target_rows": len(conditional),
            "conditional_all_checked_outside_top8": bool(conditional) and all(t["conditional_topk_membership"] == "outside" for t in conditional),
        })
    return {
        "layer": target["layer"], "natural_rows": len(row_metrics),
        "prefill_rows": sum(row["phase"] == "prefill" for row in row_metrics),
        "decode_rows": sum(row["phase"] == "decode" for row in row_metrics),
        "checked_projection_rows": sum(r["checked_projection_rows"] for r in records),
        "selected_ids_match_pinned_bit_order": all(r["scoring"]["selected_ids_match_pinned_bit_order"] for r in records),
        "norm_root_dimension_projection_bits_equal": all(r["projection_reconstructed_from_norm_root_dimension_bits_equal"] for r in records),
        "projection_values_compared": values,
        "max_projection_absolute_error_f64": max(r["max_absolute_error"] for r in projection),
        "projection_ieee32_equal_fraction": sum(r["observed_ieee32_value_equal_count"] for r in projection) / values,
        "conditional_projection_envelope_pass": all(r["conditional_envelope_pass"] is True for r in projection),
        "max_error_over_conditional_envelope": max(r["max_error_over_conditional_envelope"] for r in projection),
        "rms_ideal_bf16_bits_equal_fraction": sum(r["ideal_bf16_bits_equal_count"] for r in rms) / norms,
        "rms_ideal_bf16_max_absolute_error": max(r["ideal_bf16_value_max_absolute_difference"] for r in rms),
        "max_selected_weight_semantic_error": max(r["scoring"]["weight_semantic_max_absolute_error"] for r in records),
        "max_selected_weight_named_f32_error": max(r["scoring"]["weight_named_f32_max_absolute_error"] for r in records),
        "named_f32_weight_equal_fraction": sum(r["scoring"]["weight_named_f32_value_equal_count"] for r in records) / weights,
        "minimum_selection_boundary_gap": min(row["kth_minus_next_logit"] for row in row_metrics),
        "targets": targets,
        "native_rms_accumulation_and_weight_kernel_qualified": False,
    }


def check_capture(directory: Path, requests_path: Path):
    directory = Path(directory)
    manifest = read_json(directory / "capture-manifest.json")
    request = validate_request(read_json(Path(requests_path)))
    if manifest["routing_unchanged"] is not True or manifest["checked_rows_limit_per_layer"] != 8 or manifest["cuda_capability"] != [12, 0]:
        raise ValueError("Manifest differs from the observational SM120 scope")
    targets = manifest["targets"]
    if len(targets) != len(TARGETS) or {t["layer"] for t in targets} != set(TARGETS):
        raise ValueError("Manifest does not cover exactly the five affected layers")
    required_flags = {"allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction_split_k"}
    if set(manifest["matmul_flags"]) != required_flags or any(type(v) is not bool for v in manifest["matmul_flags"].values()):
        raise ValueError("Incomplete actual matmul precision flags")
    originals, provenance = {}, {}
    for target in targets:
        arrays = read_npz(directory, target["reference_file"], target["reference_sha256"], REFERENCE_FIELDS)
        provenance[target["layer"]] = validate_reference(target, arrays)
        originals[target["layer"]] = arrays
    lines_path = directory / "records.jsonl"
    if lines_path.stat().st_size > MAX_NPZ_BYTES:
        raise ValueError("Capture metadata exceeds its bounded read size")
    metadata = [json.loads(line) for line in lines_path.read_text().splitlines()]
    if not 1 <= len(metadata) <= 16 * len(TARGETS):
        raise ValueError("Capture exceeds the bounded natural-forward count")
    if len({r["file"] for r in metadata}) != len(metadata) or len({(r["forward_id"], r["layer"]) for r in metadata}) != len(metadata):
        raise ValueError("Duplicate capture file/forward/layer identity")
    by_layer = {layer: [] for layer in TARGETS}
    rows_by_layer = {layer: [] for layer in TARGETS}
    for row in metadata:
        index = row["layer"]
        if index not in TARGETS or row["target_experts"] != TARGETS[index] or row["routing_unchanged"] is not True:
            raise ValueError("Capture target scope or unmodified-routing marker differs")
        if row["case_id"] != request["id"] or row["prompt_sha256"] != request["prompt_sha256"]:
            raise ValueError("Capture differs from the actual client prefix/case")
        if type(row["forward_id"]) is not int or not 0 <= row["forward_id"] < 16:
            raise ValueError("Forward identity exceeds the bounded scope")
        arrays = read_npz(directory, row["file"], row["sha256"], FORWARD_FIELDS)
        positions, tokens = arrays["positions"], arrays["token_ids"]
        if row["input_rows"] != len(positions) or row["checked_rows"] != len(arrays["checked_indices"]):
            raise ValueError("Capture row counts differ from the array payload")
        expected_tokens = request["prompt_token_ids"] + request["completion_token_ids"][:-1]
        if any(not 0 <= int(p) < len(expected_tokens) or int(t) != expected_tokens[int(p)] for p, t in zip(positions, tokens)):
            raise ValueError("Captured token/position differs from supplied/returned client token IDs")
        analysis = oracle.analyze_forward(arrays, originals[index], TARGETS[index])
        analysis.update({"forward_id": row["forward_id"], "file": row["file"], "sha256": row["sha256"]})
        for metrics, position in zip(analysis["scoring"]["row_metrics"], positions):
            metrics.update({"position": int(position), "forward_id": row["forward_id"], "phase": "prefill" if position < 1025 else "decode"})
            rows_by_layer[index].append(metrics)
        by_layer[index].append(analysis)
    layers = []
    expected_positions = list(range(1025 + len(request["completion_token_ids"]) - 1))
    expected_forward_ids = None
    for target in sorted(targets, key=lambda t: t["layer"]):
        index = target["layer"]
        rows, records = rows_by_layer[index], by_layer[index]
        if sorted(row["position"] for row in rows) != expected_positions:
            raise ValueError("Natural rows are duplicate/missing/outside the one-request token scope")
        forward_ids = sorted(r["forward_id"] for r in records)
        if forward_ids != list(range(len(forward_ids))) or (expected_forward_ids is not None and forward_ids != expected_forward_ids):
            raise ValueError("Affected layers do not cover the same consecutive model forwards")
        expected_forward_ids = forward_ids
        checked = [checked["position"] for r in records for checked in r["checked_row_rank_envelopes"]]
        if not 1 <= len(checked) <= 8 or len(set(checked)) != len(checked):
            raise ValueError("Independent input-row capture is missing/duplicate/over eight per layer")
        layers.append({"summary": _summary(target, records, rows), "reference_provenance": provenance[index], "forwards": records})
    return {
        "qualification": "One unchanged-routing request; independent checked-operand projection plus observed-score selection diagnostics",
        "checkpoint_revision": "a19cfe00be84568a6867111c9a68c9c44fdcffe6",
        "manifest_sha256": hashlib.sha256((directory / "capture-manifest.json").read_bytes()).hexdigest(),
        "records_sha256": hashlib.sha256(lines_path.read_bytes()).hexdigest(),
        "client_case": request["id"], "prefix_sha256": request["prompt_sha256"],
        "prefix_tokens": 1025, "output_tokens": len(request["completion_token_ids"]),
        "natural_input_rows_per_layer": len(expected_positions),
        "captured_files": len(metadata), "checked_rows_per_layer_maximum": 8,
        "numpy_version": np.__version__, "matmul_flags": manifest["matmul_flags"],
        "layers": layers,
        "whole_router_or_model_numerically_qualified": False,
        "remaining_scope": [
            "Native BF16 cuBLAS partial-reduction precision/order is not established; gamma2816 is conditional",
            "Native RMS IR provider and exp2/div/reduction instruction error contracts are unqualified",
            "Independent projection covers only retained eight input rows per layer, not all 1032 score rows",
            "Upstream hidden-state/model semantics, natural six-expert correction coverage and G0/G1 remain unqualified",
            "Original tensor provenance is verified against producer manifest hashes; original checkpoint files were not reread here",
        ],
    }


def resolve_report_output(capture_directory: Path, requests_path: Path, output: Path) -> Path:
    """Keep CLI reports separate from immutable capture and request evidence."""
    directory = Path(capture_directory).resolve()
    request = Path(requests_path).resolve()
    destination = Path(output).resolve()
    if destination.is_relative_to(directory) or destination == request:
        raise ValueError("Report output must be outside capture directory and distinct from request evidence")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request_path = args.requests or args.capture_directory.parent / "quality-requests.json"
    output_path = resolve_report_output(args.capture_directory, request_path, args.output)
    report = check_capture(args.capture_directory, request_path)
    payload = json.dumps(report, indent=2, allow_nan=False)
    with output_path.open("x", encoding="utf-8") as output:
        output.write(payload)
    print(json.dumps({"capture_files": report["captured_files"], "natural_rows_per_layer": report["natural_input_rows_per_layer"], "checked_rows_total": sum(layer["summary"]["checked_projection_rows"] for layer in report["layers"]), "all_selected_ids_match_pinned_bit_order": all(layer["summary"]["selected_ids_match_pinned_bit_order"] for layer in report["layers"]), "whole_router_or_model_numerically_qualified": False}))


if __name__ == "__main__":
    main()
