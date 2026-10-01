"""Compare fresh owned per-path runs using the preserved v14 operator checker.

The run map explicitly binds controlled origin, token/table hashes and each
full/cached/chunked arithmetic lane to its own immutable run directory. This
wrapper imports only the NumPy CPU checker; it never launches or contacts a
server. Native-only maps are supported before the complete paired matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import compare_controlled_capture as base


PATHS = ("full", "cached", "chunked")
MODES = ("native", "paired_reference", "gate_only_negative_control")
FROZEN_METADATA = ("controlled-manifest.json", "logits-records.jsonl", "cache-contract.json",
                   "kv-records.json", "kv-binding.json")
# Installed enum retains logical BHNC while permuting its physical axes.
# Literal names from vllm/v1/kv_cache_layout.py, independently retained source.
KV_LAYOUT_NAMES = frozenset({"LBHNC", "LBNHC", "LHBNC", "BLHNC", "BLNHC", "BHLNC"})
KV_LAYOUT_ENUM_SOURCE_SHA256 = "7a32e95b61238e4e2fd8b0c420830565002af3275ca2eded6d78bc60dcaf7df0"


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Run map contains a duplicate JSON key")
        result[key] = value
    return result


def load_run_map(filename, plan):
    filename = Path(filename).resolve()
    payload = json.loads(base.read_bytes(filename, 64 << 10), object_pairs_hook=unique_json_object)
    base.reference._controlled_origin(payload)
    expected = set(base.reference.CONTROLLED_ORIGIN) | {"token_sha256", "schedule_sha256", "runs"}
    if set(payload) != expected or any(payload[field] != plan[field] for field in ("token_sha256", "schedule_sha256")):
        raise ValueError("Run map differs from the controlled prefix/table/schema")
    runs = payload["runs"]
    if not isinstance(runs, dict) or not 1 <= len(runs) <= 3 or not set(runs) <= set(PATHS):
        raise ValueError("Run map exceeds the declared execution paths")
    resolved = {}
    for path in PATHS:
        if path not in runs:
            continue
        entries = runs[path]
        if not isinstance(entries, dict) or set(entries) not in ({"native"}, {"native", "paired_reference"}, set(MODES)):
            raise ValueError("Each path needs native, optional paired reference, then optional negative control")
        resolved[path] = {}
        for mode in MODES:
            if mode not in entries:
                continue
            value = entries[mode]
            if not isinstance(value, str) or not value or "\0" in value:
                raise ValueError("Run map needs explicit nonempty filesystem directories")
            directory = Path(value)
            if not directory.is_absolute():
                directory = filename.parent / directory
            directory = directory.resolve()
            if not directory.is_dir():
                raise ValueError("Mapped owned run directory is missing")
            resolved[path][mode] = directory
    return resolved


def metadata_hashes(directory, path):
    directory = Path(directory) / "controlled" / path
    binding = base.read_json(directory / "kv-binding.json")
    owners = binding["layers"]
    if not isinstance(owners, list) or len(owners) != 30 or any(item.get("resolved_layout") not in KV_LAYOUT_NAMES for item in owners):
        raise ValueError("K/V binding has an unknown installed physical layout enum")
    return {name: base.digest_file(directory / name, base.MAX_JSON) for name in FROZEN_METADATA}


def _compare_paths(candidate_logits, full_logits, candidate_cache, full_cache, plan):
    report = {"prompt_handoff_logits": base.reference.compare_logits(candidate_logits["primary"], full_logits["primary"], plan),
              "candidate_source_rows": candidate_logits["primary_source_rows"],
              "reference_source_rows": full_logits["primary_source_rows"],
              "head_batch_shape_confound_present": candidate_logits["primary_source_rows"] != full_logits["primary_source_rows"],
              "logical_cache": base.reference.compare_cache_collection(candidate_cache["snapshots"], full_cache["snapshots"], plan,
                  cache_contract=candidate_cache["contract"]),
              "quality_gate_passed": False, "independent_cache_correctness_qualified": False}
    if (32, 1) not in candidate_logits["rows"] or (32, 1) not in full_logits["rows"]:
        raise ValueError("Cross-path comparison misses the same single-row sampler at position32")
    report["same_single_row_sampler_position32"] = base.logit_value_difference(candidate_logits["rows"][32, 1], full_logits["rows"][32, 1])
    return report


def compare_positive_only_runs(plan_directory, native, paired, checkpoint, path):
    """Only two available lanes; do not manufacture a missing cached negative."""
    plan = base.load_plan(plan_directory)
    checkpoint = Path(checkpoint)
    if base.digest_file(checkpoint / "config.json", base.MAX_JSON) != base.reference.CONFIG_SHA256:
        raise ValueError("Positive-only original checkpoint config differs")
    reader = base.formats.CheckpointReader(checkpoint)
    nominal = base.reference.nominal_cache_contract(checkpoint / "config.json")
    cases = [base.load_case(directory, path, mode, plan) for directory, mode in
             ((native, "native"), (paired, "paired_reference"))]
    report = {**base.reference.CONTROLLED_ORIGIN, "qualification": "Positive controlled stages only; negative lane absent for this path",
              "checkpoint_revision": base.REVISION, "token_sha256": plan["token_sha256"], "schedule_sha256": plan["schedule_sha256"],
              "numpy_version": base.np.__version__, "paths": [], "quality_gate_passed": False,
              "paired_reference_and_negative_matrix_present": False, "negative_control_available_for_path": False,
              "native_mma_qualified": False, "whole_model_or_frozen_quantizer_qualified": False}
    result = {"path": path, "dispatch": [case["dispatch_check"] for case in cases],
              "traces": [case["trace"] for case in cases],
              "controlled_counters": [case["counter_check"] for case in cases], "experts": []}
    calibration, gates = {}, True
    for layer, expert in base.reference.TARGETS:
        key = layer, expert, base.reference.TARGET_POSITIONS[layer, expert]
        originals, hashes = base.reference.original_expert(reader, layer, expert)
        base.check_original_hashes(cases, originals, hashes, layer, expert)
        if layer not in calibration:
            calibration[layer] = base.selected_runtime_globals(reader, layer)
        expected_globals, calibration_hash = calibration[layer]
        stages = [case["stages"][key] for case in cases]
        for stage in stages:
            if not base.np.array_equal(stage["a1_bits"], expected_globals["a1"]) or not base.np.array_equal(stage["a2_bits"], expected_globals["a2"]):
                raise ValueError("Positive-only A1/A2 differs from original layer maxima")
        replays = [base.reference.replay_expert(stage, originals, a1_bits=stage["a1_bits"], a2_bits=stage["a2_bits"], mode=case["manifest"]["mode"])
                   for stage, case in zip(stages, cases)]
        weighting = [base.compare_weighting(stage) for stage in stages]
        pair = base.comparison.compare_positive_stages(stages[0], stages[1])
        gate = pair["strict_observed_stage_gate_pass"] and all(item[label]["raw_bf16_bits_equal"] for item in replays for label in ("gate", "up", "down")) and all(
            item["weighted_f32_bits_equal"] and item["combined_bf16"]["raw_bf16_bits_equal"] for item in weighting)
        gates &= gate
        result["experts"].append({"layer": layer, "expert": expert, "position": key[2], "original_projection_hashes": hashes,
            "calibration_source_digest": calibration_hash, "independent_replays": replays,
            "positive_paired_stages": pair, "weighting": weighting, "strict_observed_operator_gate_pass": gate})
    result["strict_observed_operator_gates_pass"] = gates
    if gates:
        logits = [base.load_logits(case, plan) for case in cases]
        caches = [base.load_cache(case, plan, nominal) for case in cases]
        result["raw_logit_capture_roles"] = [{"records": item["records"], "primary_source_rows": item["primary_source_rows"],
            "within_position_different_head_batches": item["within_position_different_head_batches"]} for item in logits]
        result["positive_logits"] = base.compare_logit_variants(logits[0], logits[1], plan)
        result["logical_cache_structure"] = [item["summary"] for item in caches]
        result["positive_logical_cache"] = base.reference.compare_cache_collection(caches[0]["snapshots"], caches[1]["snapshots"], plan,
            cache_contract=caches[0]["contract"])
    else:
        result["downstream_comparison_blocker"] = "Strict observed positive operator gate failed; logits/cache promotion withheld"
    report["paths"].append(result)
    report["all_strict_observed_operator_gates_pass"] = gates
    report["remaining_scope"] = ["No negative control was executed for this path; complete paired matrix remains false",
        "Positive reference shares native activation/quantization and untouched operators; no model/quality/native-MMA qualification"]
    return report


def compare_mapped_runs(plan_directory, map_filename, checkpoint, paths=None):
    plan = base.load_plan(plan_directory)
    mapped = load_run_map(map_filename, plan)
    paths = list(mapped) if paths is None else list(paths)
    if not 1 <= len(paths) <= 3 or len(set(paths)) != len(paths) or not set(paths) <= set(mapped):
        raise ValueError("Requested paths are duplicate or absent from the explicit run map")
    report = {**base.reference.CONTROLLED_ORIGIN, "qualification": "Bounded controlled operator and path comparisons; never natural quality",
              "token_sha256": plan["token_sha256"], "schedule_sha256": plan["schedule_sha256"],
              "run_map_sha256": base.digest_file(map_filename, 64 << 10), "per_path": [],
              "within_mode_path_comparisons": {}, "quality_gate_passed": False,
              "native_mma_qualified": False, "whole_model_or_frozen_quantizer_qualified": False}
    report["kv_layout_enum_source_sha256"] = KV_LAYOUT_ENUM_SOURCE_SHA256
    frozen, path_reports = {}, {}
    for path in paths:
        runs = mapped[path]
        frozen[path] = {mode: metadata_hashes(directory, path) for mode, directory in runs.items()}
        if set(runs) == {"native", "paired_reference"}:
            result = compare_positive_only_runs(plan_directory, runs["native"], runs["paired_reference"], checkpoint, path)
        else:
            result = base.compare_runs(plan_directory, runs["native"], runs.get("paired_reference"),
                                       runs.get("gate_only_negative_control"), checkpoint, [path])
        for mode, directory in runs.items():
            if metadata_hashes(directory, path) != frozen[path][mode]:
                raise ValueError("Captured metadata changed during bounded operator replay")
        path_reports[path] = result
        report["per_path"].append({"path": path, "result": result, "frozen_metadata_sha256": frozen[path]})
    report["all_strict_observed_operator_gates_pass"] = all(result["all_strict_observed_operator_gates_pass"] for result in path_reports.values())
    report["complete_paired_matrix_for_requested_paths"] = all(set(mapped[path]) == set(MODES) for path in paths)
    if "full" in path_reports and path_reports["full"]["all_strict_observed_operator_gates_pass"]:
        nominal = base.reference.nominal_cache_contract(Path(checkpoint) / "config.json")
        full_rows = {}
        for path in ("cached", "chunked"):
            if path not in path_reports:
                continue
            if not path_reports[path]["all_strict_observed_operator_gates_pass"]:
                report["within_mode_path_comparisons"][path] = {"blocker": "Path operator/negative-control gate failed; cross-path promotion withheld"}
                continue
            paired = {}
            for mode in MODES:
                if mode not in mapped[path] or mode not in mapped["full"]:
                    continue
                if mode not in full_rows:
                    full_case = base.load_case(mapped["full"][mode], "full", mode, plan)
                    full_rows[mode] = base.load_logits(full_case, plan), base.load_cache(full_case, plan, nominal)
                candidate = base.load_case(mapped[path][mode], path, mode, plan)
                candidate_rows, candidate_cache = base.load_logits(candidate, plan), base.load_cache(candidate, plan, nominal)
                for scope in ("full", path):
                    if metadata_hashes(mapped[scope][mode], scope) != frozen[scope][mode]:
                        raise ValueError("Captured metadata changed before cross-path comparison")
                paired[mode] = _compare_paths(candidate_rows, full_rows[mode][0], candidate_cache, full_rows[mode][1], plan)
            report["within_mode_path_comparisons"][path] = paired
    elif len(paths) > 1:
        report["cross_path_blocker"] = "Full path is absent or its strict operator/negative-control gate failed"
    pairs = [item for by_mode in report["within_mode_path_comparisons"].values() for mode, item in by_mode.items() if mode in MODES]
    report["within_mode_all_logical_cache_bits_equal"] = all(item["logical_cache"]["all_selected_logical_k_v_bits_equal"] for item in pairs) if pairs else None
    report["within_mode_all_same_sampler_logits_bits_equal"] = all(item["same_single_row_sampler_position32"]["raw_f32_bits_equal"] for item in pairs) if pairs else None
    report["controlled_handoff_equivalence_observed"] = bool(pairs) and report["all_strict_observed_operator_gates_pass"] and report["within_mode_all_logical_cache_bits_equal"] and report["within_mode_all_same_sampler_logits_bits_equal"]
    report["remaining_scope"] = ["Native activation, CUDA quantization, ordinary experts and the output head are shared with the paired reference",
        "Per-path original-weight replay qualifies only retained actual operands; native accumulation bounds remain conditional",
        "Prompt LM-head batch sizes differ across paths; compare position32 sampler M1 separately",
        "Logical KV equivalence uses retained registered-writer associations, not an independent scheduler/attention oracle",
        "Fixed artificial routes and singleton outputs cannot grant natural quality, G0/G1 or corrected timing acceptance"]
    report["cpu_checker_source_sha256"] = {"compare_controlled_capture.py": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        "compare_controlled_paths.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-map", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--paths", choices=PATHS, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = base.load_plan(args.plan)
    mapped = load_run_map(args.run_map, plan)
    inputs = [args.plan, args.run_map, args.checkpoint] + [directory for runs in mapped.values() for directory in runs.values()]
    destination = base.report_output_path(args.output, inputs)
    report = compare_mapped_runs(args.plan, args.run_map, args.checkpoint, args.paths)
    payload = json.dumps(report, indent=2, allow_nan=False)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(json.dumps({"all_strict_observed_operator_gates_pass": report["all_strict_observed_operator_gates_pass"],
                      "complete_paired_matrix_for_requested_paths": report["complete_paired_matrix_for_requested_paths"],
                      "quality_gate_passed": False}))
    return 0 if report["all_strict_observed_operator_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
