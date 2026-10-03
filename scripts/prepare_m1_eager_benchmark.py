"""Prepare private exact-ID prompts and a frozen matched eager schedule; no CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from megartx.m1_eager_benchmark import (CONTEXTS, DRIVER_SOURCES, OUTPUTS, SCHEMA,
                                      PROFILE_DRIVER_SOURCES, PROFILE_SCHEMA, PROFILE_RULE, validate_plan)
from megartx.m1_execution import CONTROLLER_SOURCES
from megartx.m1_normal_plan import REVISION, digest


def make_plan(prompts, source_head, controller_hashes, driver_hashes, trials=6, warmups=2, seed=9471,
              metadata_help_timing=False, decode_profile=False):
    if type(decode_profile) is not bool:
        raise RuntimeError("decode profile intent must be explicit boolean")
    rng = random.Random(seed)
    schedule = []
    for phase, count in (("warmup", warmups), ("measurement", trials)):
        cells = [(str(n), i) for n in CONTEXTS for i in range(count)]
        rng.shuffle(cells)
        orders = {}
        for n in CONTEXTS:
            first = ["stock"] * ((count + 1) // 2) + ["fused"] * (count // 2)
            rng.shuffle(first)
            orders[str(n)] = first
        for case, i in cells:
            first = orders[case][i]
            for lane in (first, "stock" if first == "fused" else "fused"):
                schedule.append({"id": f"{phase}-{case}-{i}-{lane}", "phase": phase,
                                 "case": case, "trial": i, "seed": seed + i, "lane": lane})
    plan = {"schema": PROFILE_SCHEMA if decode_profile else SCHEMA,
            "checkpoint_revision": REVISION, "source_head": source_head,
            "controller_source_hashes": controller_hashes, "driver_source_hashes": driver_hashes,
            "outputs": OUTPUTS, "prefill_chunk": 256, "warmups": warmups, "trials": trials,
            "seed": seed, "cases": [{"id": str(n), "prompt_token_ids": prompts[n],
                                       "prompt_sha256": digest(prompts[n])} for n in CONTEXTS],
            "schedule": schedule, "metadata_help_timing": metadata_help_timing}
    if decode_profile:
        plan["diagnostic_admission"] = PROFILE_RULE
    plan["plan_sha256"] = digest(plan)
    return validate_plan(plan)


def exact_prompt(tokenizer, n):
    filler = "The observatory records the weather, maintains its instruments, and checks observations against the written log. "
    content = ("Read the following background notes. The secret label is COBALT.\n" + filler * (n // 12 + 20)
               + "\nWrite a detailed, coherent explanation of how an observatory maintains reliable records. Include the secret label.")
    tokens = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    if hasattr(tokens, "keys"):
        tokens = tokens["input_ids"]
    if not isinstance(tokens, list) or any(type(t) is not int for t in tokens) or len(tokens) < n:
        raise RuntimeError("tokenizer did not produce a sufficiently long flat token sequence")
    extra = len(tokens) - n
    return tokens[:256] + tokens[256 + extra:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9471)
    parser.add_argument("--m1-timing-metadata-help", action="store_true")
    parser.add_argument("--m1-decode-profile", action="store_true")
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("freeze and commit source before preparing a GPU plan")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    plan = make_plan({n: exact_prompt(tokenizer, n) for n in CONTEXTS},
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        {n: sha(ROOT / "src/megartx" / n) for n in CONTROLLER_SOURCES},
        {n: sha(ROOT / n) for n in (PROFILE_DRIVER_SOURCES if args.m1_decode_profile else DRIVER_SOURCES)},
        args.trials, args.warmups, args.seed,
        args.m1_timing_metadata_help, args.m1_decode_profile)
    with args.output.open("x") as stream:
        json.dump(plan, stream, indent=2)
    print(json.dumps({"plan_sha256": plan["plan_sha256"], "source_head": plan["source_head"],
                      "requests": len(plan["schedule"]), "pairs_per_context": plan["trials"]}))


if __name__ == "__main__":
    main()
