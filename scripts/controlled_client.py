"""Exactly one predeclared artificial-route request; raw evidence stays local."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import requests

import controlled_reference as reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--path", required=True, choices=("full", "cached", "chunked"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.plan / "manifest.json").read_text())
    with np.load(args.plan / "route-table.npz", allow_pickle=False) as arrays:
        plan = {**reference.CONTROLLED_ORIGIN, **{k: arrays[k].copy() for k in ("tokens", "ids", "weight_bits")},
                "token_sha256": manifest["token_sha256"], "schedule_sha256": manifest["schedule_sha256"]}
    reference.validate_schedule(plan)
    directory = Path(os.environ["MEGARTX_CONTROLLED_DIR"])
    if directory != args.output / "controlled" or not directory.is_dir() or (directory / args.path).exists():
        raise RuntimeError("Controlled observer was not activated or a request already ran")
    tokens = plan["tokens"].tolist()
    supplied = tokens[:32] if args.path == "cached" else tokens
    expected = [tokens[32]] * (2 if args.path == "cached" else 1)
    payload = {"model": "gemma4-nvfp4", "prompt": supplied, "max_tokens": len(expected),
               "temperature": 0, "seed": 1234, "ignore_eos": False, "return_token_ids": True,
               "logprobs": 1, "prompt_logprobs": 1, "allowed_token_ids": [tokens[32]]}
    origin = {**reference.CONTROLLED_ORIGIN, "routing_unchanged": False}
    request = {**origin, "controlled_path": args.path, "schedule_sha256": plan["schedule_sha256"],
               "token_sha256": plan["token_sha256"], "id": "controlled-" + args.path,
               "teacher_forced": False, "prompt_token_ids": supplied,
               "prompt_sha256": hashlib.sha256(json.dumps(supplied).encode()).hexdigest()}
    marker = args.output / "capture-request.json"
    with marker.open("x") as stream:
        json.dump(request, stream)
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post("http://127.0.0.1:18000/v1/completions", json=payload, timeout=(5, 300))
        response.raise_for_status()
        data = response.json()
    finally:
        marker.unlink()
    choice, usage = data["choices"][0], data["usage"]
    if choice.get("token_ids") != expected or usage["prompt_tokens"] != len(supplied) or usage["completion_tokens"] != len(expected):
        raise RuntimeError("API did not forward the predeclared cached continuation")
    case = directory / args.path
    if os.environ.get("MEGARTX_M1_EXECUTION", "captured") == "capture-free":
        if args.path != "cached" or os.environ.get("MEGARTX_M1_PREPARATION") not in {"stock", "fused"}:
            raise RuntimeError("Capture-free client requires the bounded cached stock/fused request")
        # Only the client writes a scalar lifecycle record, after the response.
        # No tensor/trace proof is produced or accepted by this execution lane.
        case.mkdir(exist_ok=False)
        record = {**request, "usage": usage, "completion_token_ids": choice["token_ids"],
                  "finish_reason": choice["finish_reason"], "request_count": 1,
                  "payload_controls": {k: v for k, v in payload.items() if k != "prompt"},
                  "m1_execution": "capture-free", "quality_gate_passed": False,
                  "timing_qualified": False, "capture_comparison_available": False}
        with (case / "capture-free-request.json").open("x") as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps({**origin, "path": args.path, "request_count": 1,
                          "m1_execution": "capture-free", "capture_comparison_available": False}), flush=True)
        return
    captured = json.loads((case / "controlled-manifest.json").read_text())
    if captured["schedule_sha256"] != plan["schedule_sha256"] or captured["input_tokens"] != 33 or len(captured["executed_interventions"]) != 6:
        raise RuntimeError("Controlled registered corrections were not fully captured")
    positions = set()
    for line in (case / "logits-records.jsonl").read_text().splitlines():
        record = json.loads(line)
        path = case / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError("Raw-logit evidence hash differs")
        with np.load(path, allow_pickle=False) as arrays:
            positions.update(arrays["input_positions"].tolist())
    if positions != {31, 32}:
        raise RuntimeError("Both actual input positions 31 and 32 need pre-sampler raw logits")
    record = {**request, "usage": usage, "completion_token_ids": choice["token_ids"],
              "finish_reason": choice["finish_reason"], "request_count": 1,
              "payload_controls": {k: v for k, v in payload.items() if k != "prompt"},
              "observed_logit_positions": sorted(positions), "quality_gate_passed": False,
              "timing_qualified": False}
    with (case / "request.json").open("x") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps({**origin, "path": args.path, "request_count": 1, "actual_input_tokens": 33,
                      "completed_positive_corrections": 6, "observed_logit_positions": sorted(positions)}), flush=True)


if __name__ == "__main__":
    main()
