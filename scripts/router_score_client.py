"""Exactly one existing 1025-token prefix, at most eight natural output tokens.

No prompt generation, forced expert selection, timing claims, or quality gate.
Private observations are validated by a separate independent CPU checker.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import requests


def existing_prefix(path):
    records = json.loads(path.read_text())
    matches = [row for row in records if row["id"] == "context-1025"]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one recorded context-1025 prefix")
    row = matches[0]
    tokens = row["prompt_token_ids"]
    digest = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
    if len(tokens) != 1025 or digest != row["prompt_sha256"] or any(type(t) is not int or t < 0 for t in tokens):
        raise RuntimeError("Recorded prefix length/hash/token contract differs")
    return tokens, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tokens, prefix_sha = existing_prefix(args.prefix_manifest)
    directory = Path(os.environ["MEGARTX_ROUTER_SCORE_DIR"])
    if directory.parent != args.output or not (directory / "capture-manifest.json").exists():
        raise RuntimeError("The bounded production observer was not activated")
    if (directory / "records.jsonl").exists():
        raise RuntimeError("Refusing a second request in the score diagnostic")
    payload = {"model": "gemma4-nvfp4", "prompt": tokens, "max_tokens": 8, "temperature": 0, "seed": 1234, "ignore_eos": False, "return_token_ids": True, "logprobs": 1, "prompt_logprobs": 1}
    marker = args.output / "capture-request.json"
    request = {"id": "router-score-context-1025", "prompt_sha256": prefix_sha, "prompt_token_ids": tokens, "teacher_forced": False}
    with marker.open("x") as f:
        json.dump(request, f)
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post("http://127.0.0.1:18000/v1/completions", json=payload, timeout=(5, 300))
        response.raise_for_status()
        data = response.json()
    finally:
        marker.unlink()
    usage = data["usage"]
    choice = data["choices"][0]
    completion = choice.get("token_ids")
    if usage["prompt_tokens"] != 1025 or not 1 <= usage["completion_tokens"] <= 8 or not isinstance(completion, list) or len(completion) != usage["completion_tokens"]:
        raise RuntimeError("Server violated the one-prefix/eight-output-token contract")
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    if set(row["layer"] for row in records) != {0, 1, 2, 3, 5} or any(row["case_id"] != request["id"] or row["prompt_sha256"] != prefix_sha for row in records):
        raise RuntimeError("Production router capture does not cover the exact request")
    arrays = __import__("numpy")
    expected_tokens = tokens + completion
    observed = {}
    for row in records:
        path = directory / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("Router observation hash differs")
        with arrays.load(path, allow_pickle=False) as values:
            positions = values["positions"].tolist()
            actual_tokens = values["token_ids"].tolist()
            if any(p >= len(expected_tokens) or expected_tokens[p] != t for p, t in zip(positions, actual_tokens)):
                raise RuntimeError("Observed router input is not the supplied prefix/returned decode")
            observed.setdefault(row["layer"], []).extend(positions)
    reference_positions = observed[0]
    if reference_positions[:1025] != list(range(1025)) or any(p != reference_positions for p in observed.values()):
        raise RuntimeError("Missing/duplicated prefix rows or cross-layer position correspondence")
    if reference_positions[1025:] != list(range(1025, 1025 + len(reference_positions) - 1025)) or len(reference_positions) > 1033:
        raise RuntimeError("Unbounded or discontinuous decode observations")
    record = {**request, "usage": usage, "completion_token_ids": completion, "finish_reason": choice["finish_reason"], "output_text": choice["text"], "logit_files": sorted(p.name for p in (args.output / "logits").glob("logits-*.pt")), "payload_controls": {k: v for k, v in payload.items() if k != "prompt"}}
    (args.output / "quality-requests.json").write_text(json.dumps([record], indent=2))
    (args.output / "router-score-request.json").write_text(json.dumps({"request": record, "source_manifest_sha256": hashlib.sha256(args.prefix_manifest.read_bytes()).hexdigest(), "source_case_id": "context-1025", "request_count": 1, "output_cap": 8, "routing_unchanged": True, "verified_cross_layer_positions": True, "actual_input_rows_per_layer": len(reference_positions), "qualification": "Diagnostic only; no full-model quality/performance gate"}, indent=2))
    print(json.dumps({"request_count": 1, "prompt_tokens": 1025, "completion_tokens": usage["completion_tokens"], "router_input_rows_per_layer": len(reference_positions), "capture_files": len(records), "routing_unchanged": True}), flush=True)


if __name__ == "__main__":
    main()
