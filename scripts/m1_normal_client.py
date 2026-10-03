"""Exactly two fixed normal-routing requests with a constrained continuation."""
import argparse
import json
import os
from pathlib import Path
import sys

import requests

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from megartx.m1_normal_plan import load_plan, request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if Path(os.environ["MEGARTX_M1_NORMAL_DIR"]) != args.output/"normal":
        raise RuntimeError("normal M1 observer destination differs")
    session = requests.Session()
    session.trust_env = False
    for case in plan["cases"]:
        marker = args.output/"capture-request.json"
        record = request(plan,case)
        payload = {"model": "gemma4-nvfp4", "prompt": case["prompt_token_ids"], "max_tokens": 4,
                   "temperature": 0, "seed": 1234, "ignore_eos": False,
                   "return_token_ids": True, "allowed_token_ids": [plan["continuation_token_id"]]}
        with marker.open("x") as stream:
            json.dump(record,stream)
        try:
            response = session.post("http://127.0.0.1:18000/v1/completions",json=payload,timeout=(5,300))
            response.raise_for_status()
            data = response.json()
        finally:
            marker.unlink()
        choice,usage = data["choices"][0],data["usage"]
        if (choice.get("token_ids") != [plan["continuation_token_id"]]*4
                or usage["prompt_tokens"] != len(case["prompt_token_ids"]) or usage["completion_tokens"] != 4):
            raise RuntimeError("normal M1 API changed the declared input continuation")
        directory = args.output/"normal"/case["id"]
        manifest = json.loads((directory/"normal-manifest.json").read_text())
        logits = (directory/"logits-records.jsonl").read_text().splitlines()
        if manifest["request"] != record or len(logits) != 4:
            raise RuntimeError("normal M1 observer did not complete the declared request")
        with (directory/"request.json").open("x") as stream:
            json.dump({**record, "request_count": 1, "usage": usage,
                "completion_token_ids": choice["token_ids"], "finish_reason": choice["finish_reason"],
                "quality_qualified": False, "timing_qualified": False},stream,indent=2)
        print(json.dumps({"case": case["id"], "prompt_tokens": len(case["prompt_token_ids"]),
            "outputs": 4,"continuation_constrained": True,"routing_unchanged": True}),flush=True)


if __name__ == "__main__":
    main()
