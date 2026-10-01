"""Compare matched full-vocabulary rows without fitting acceptance tolerances."""
import argparse
import json
from pathlib import Path

import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from megartx.nvfp4_qualification import activation, natural_coverage

parser = argparse.ArgumentParser()
parser.add_argument("candidate", type=Path)
parser.add_argument("reference", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(4)
activation(args.candidate)
activation(args.reference)
natural_coverage(args.candidate, "native")
natural_coverage(args.reference, "reference")
a = json.loads((args.candidate / "quality-requests.json").read_text())
b = json.loads((args.reference / "quality-requests.json").read_text())
if [x["id"] for x in a] != [x["id"] for x in b]:
    raise RuntimeError("Quality corpus differs")
report = {"qualification": "Selected six-expert original-weight correction under the chosen runtime quantizer; shared untouched model components are not independently qualified", "cases": []}
for c, r in zip(a, b):
    if c["prompt_sha256"] != r["prompt_sha256"]:
        raise RuntimeError("Teacher-forced prefixes differ")
    row = {"id": c["id"], "prompt_tokens": len(c["prompt_token_ids"]), "candidate_output": c["output_text"], "reference_output": r["output_text"], "greedy_output_equal": c["output_text"] == r["output_text"], "teacher_forced": c["teacher_forced"], "batches": []}
    scored = {}
    positions_seen = set()
    if c["teacher_forced"]:
        if len(c["logit_files"]) != len(r["logit_files"]):
            raise RuntimeError("Matched logit call count differs")
        for cf, rf in zip(c["logit_files"], r["logit_files"]):
            cx = torch.load(args.candidate / "logits" / cf, weights_only=True)
            rx = torch.load(args.reference / "logits" / rf, weights_only=True)
            if cx["source_rows"] != rx["source_rows"] or cx["logits"].shape != rx["logits"].shape:
                raise RuntimeError("Matched full-logit shape differs")
            for key in ("input_positions", "prediction_positions", "input_token_ids", "case_id", "prompt_sha256"):
                if key not in cx or cx[key] != rx.get(key):
                    raise RuntimeError(f"Matched token-row identity differs: {key}")
            if cx["case_id"] != c["id"] or cx["prompt_sha256"] != c["prompt_sha256"]:
                raise RuntimeError("Captured logits do not match the client prefix")
            x, y = cx["logits"].double(), rx["logits"].double()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise RuntimeError("Non-finite full-vocabulary logits")
            d = x - y
            lp, lq = x.log_softmax(-1), y.log_softmax(-1)
            kl = (lq.exp() * (lq - lp)).sum(-1)
            for j, position in enumerate(cx["prediction_positions"]):
                positions_seen.add(position)
                if position < len(c["prompt_token_ids"]) and position not in scored:
                    target = c["prompt_token_ids"][position]
                    scored[position] = (-lp[j, target].item(), -lq[j, target].item())
            row["batches"].append({"source_rows": cx["source_rows"], "compared_rows": x.shape[0], "vocabulary": x.shape[1], "input_positions": cx["input_positions"], "row_identity_verified": True, "max_abs_logit_difference": d.abs().max().item(), "logit_rmse": d.square().mean().sqrt().item(), "relative_logit_l2": (d.square().sum() / y.square().sum().clamp_min(1e-300)).sqrt().item(), "top1_agreement": (x.argmax(-1) == y.argmax(-1)).double().mean().item(), "reference_to_candidate_kl_mean": kl.mean().item(), "reference_to_candidate_kl_max": kl.max().item()})
    row["unique_teacher_forced_prediction_positions"] = len(positions_seen)
    row["synthetic_scored_positions"] = len(scored)
    if scored:
        import math
        cnll = sum(v[0] for v in scored.values()) / len(scored)
        rnll = sum(v[1] for v in scored.values()) / len(scored)
        row["synthetic_candidate_mean_nll"] = cnll
        row["synthetic_reference_mean_nll"] = rnll
        row["synthetic_ppl_ratio"] = math.exp(cnll - rnll)
    report["cases"].append(row)
report["all_full_logits_finite"] = True
report["teacher_forced_rows"] = sum(z["compared_rows"] for c in report["cases"] for z in c["batches"])
report["unique_teacher_forced_prediction_positions"] = sum(c["unique_teacher_forced_prediction_positions"] for c in report["cases"])
report["synthetic_scored_positions"] = sum(c["synthetic_scored_positions"] for c in report["cases"])


def route_coverage(directory, requests, mode):
    manifests = [json.loads(line) for line in (directory / "adapter-manifest.jsonl").read_text().splitlines()]
    expected = {(m["loader_ordinal"], e["expert"]) for m in manifests for e in m["affected_experts"]}
    prefixes = {r["id"]: r["prompt_sha256"] for r in requests}
    totals = {key: {"routed_rows": 0, "nonzero_route_weights": 0, "cases": set()} for key in expected}
    for line in (directory / "route-hits.jsonl").read_text().splitlines():
        hit = json.loads(line)
        key = hit["loader_ordinal"], hit["expert"]
        if key not in totals or hit["mode"] != mode or prefixes.get(hit["case_id"]) != hit["prompt_sha256"]:
            raise RuntimeError("Route audit does not match the loaded experts/client corpus")
        if not 0 <= hit["nonzero_route_weights"] <= hit["routed_rows"]:
            raise RuntimeError("Invalid route audit counts")
        totals[key]["routed_rows"] += hit["routed_rows"]
        totals[key]["nonzero_route_weights"] += hit["nonzero_route_weights"]
        totals[key]["cases"].add(hit["case_id"])
    return [{"loader_ordinal": key[0], "expert": key[1], **{k: v for k, v in values.items() if k != "cases"}, "cases": sorted(values["cases"])} for key, values in sorted(totals.items())]


candidate_routes = route_coverage(args.candidate, a, "native")
reference_routes = route_coverage(args.reference, b, "reference")
if [(h["loader_ordinal"], h["expert"]) for h in candidate_routes] != [(h["loader_ordinal"], h["expert"]) for h in reference_routes]:
    raise RuntimeError("Candidate/reference correction scope differs")
report["live_route_coverage"] = {"candidate": candidate_routes, "reference": reference_routes, "all_corrected_experts_exercised": bool(candidate_routes) and all(h["nonzero_route_weights"] > 0 for h in candidate_routes + reference_routes), "counts_equal": candidate_routes == reference_routes}
report["quality_gate_passed"] = False
report["remaining_scope"] = "Whole-checkpoint activation quantizer, independent untouched-model semantics/cache, frozen held-out quality margins remain unqualified. Small regression comparisons do not pass G1."
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps({"compared_teacher_forced_rows": report["teacher_forced_rows"], "max_abs_logit_difference": max(z["max_abs_logit_difference"] for c in report["cases"] for z in c["batches"]), "quality_gate_passed": False}), flush=True)
