"""Run a concurrency-one benchmark over host-local HTTP; no model imports."""
import argparse
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import time

import requests
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--url", default="http://127.0.0.1:18000")
parser.add_argument("--trials", type=int, default=30)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--prepare-only", action="store_true")
args = parser.parse_args()
if not args.prepare_only:
    parser.error("Timing disabled: positive natural correction coverage and paired numerical/quality qualification remain unresolved; --prepare-only performs token validation without inference")
out = pathlib.Path(args.output)
out.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
session = requests.Session()
session.trust_env = False
events = (out / "phase-boundaries.jsonl").open("a", buffering=1)


def event(phase, **fields):
    events.write(json.dumps({"phase": phase, "monotonic_ns": time.perf_counter_ns(), "unix_ns": time.time_ns(), **fields}) + "\n")


def chat_tokens(content):
    encoded = tokenizer.apply_chat_template([{ "role": "user", "content": content}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    tokens = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    assert isinstance(tokens, list) and all(isinstance(x, int) for x in tokens), "Expected one flat integer token-ID sequence"
    return tokens


def exact_prompt(length):
    filler = "The observatory records the weather, maintains its instruments, and checks observations against the written log. "
    beginning = "Read the following background notes. The secret label is COBALT.\n"
    ending = "\nWrite a detailed, coherent explanation of how an observatory maintains reliable records. Include the secret label."
    tokens = chat_tokens(beginning + filler * (length // 12 + 20) + ending)
    # Remove only a middle region of neutral background. Preserve the chat
    # framing, beginning, final instruction, and generation suffix verbatim.
    extra = len(tokens) - length
    assert extra >= 0 and length > 512
    cut = 256
    tokens = tokens[:cut] + tokens[cut + extra:]
    assert len(tokens) == length
    return tokens


prompts = {n: exact_prompt(n) for n in (2048, 8192)}
(out / "prompts.json").write_text(json.dumps({str(k): {"token_ids": v, "decoded": tokenizer.decode(v, skip_special_tokens=False), "sha256": hashlib.sha256(json.dumps(v).encode()).hexdigest()} for k, v in prompts.items()}, indent=2))
if args.prepare_only:
    check = {"tokenizer_class": type(tokenizer).__name__, "return_dict": False, "exact_prompt_counts": {str(k): len(v) for k, v in prompts.items()}, "flat_integer_ids": True, "template_suffix": tokenizer.decode(prompts[2048][-15:], skip_special_tokens=False)}
    (out / "prompt-token-validation.json").write_text(json.dumps(check, indent=2))
    print(json.dumps(check, indent=2))
    events.close()
    raise SystemExit(0)


def completion(tokens, phase, trial, output_tokens=256, forced=True, logprobs=None):
    rid = f"{phase}-{len(tokens)}-{trial}"
    payload = {"model": "gemma4-nvfp4", "prompt": tokens, "add_special_tokens": False, "max_tokens": output_tokens, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 1234 + trial, "n": 1, "stream": True, "stream_interval": 1, "stream_options": {"include_usage": True}, "return_token_ids": True, "ignore_eos": forced, "skip_special_tokens": False, "request_id": rid}
    if logprobs is not None:
        payload["logprobs"] = logprobs
    encoded = json.dumps(payload).encode()
    event("request_start", request_id=rid, prompt_tokens=len(tokens), requested_output_tokens=output_tokens, kind=phase)
    start = time.perf_counter_ns()
    chunks, ids, times, texts, values = [], [], [], [], []
    usage = None
    server_metrics = None
    finish = None
    with session.post(args.url + "/v1/completions", data=encoded, headers={"Content-Type": "application/json"}, stream=True, timeout=(5, 180)) as response:
        header_time = time.perf_counter_ns()
        response.raise_for_status()
        for raw in response.iter_lines(chunk_size=None):
            if not raw.startswith(b"data: "):
                continue
            now = time.perf_counter_ns()
            raw = raw[6:]
            if raw == b"[DONE]":
                break
            data = json.loads(raw)
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            if data.get("usage"):
                usage = data["usage"]
            if data.get("metrics"):
                server_metrics = data["metrics"]
            for choice in data.get("choices", []):
                delta = choice.get("token_ids") or []
                chunks.append({"elapsed_ns": now - start, "token_ids": delta, "text": choice.get("text", ""), "finish_reason": choice.get("finish_reason")})
                texts.append(choice.get("text", ""))
                ids.extend(delta)
                times.extend([now - start] * len(delta))
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                lp = choice.get("logprobs") or {}
                values.extend(x for x in (lp.get("token_logprobs") or []) if x is not None)
                for candidates in lp.get("top_logprobs") or []:
                    values.extend((candidates or {}).values())
    end = time.perf_counter_ns()
    assert usage is not None, "Server did not report token usage"
    assert usage["prompt_tokens"] == len(tokens), (usage, len(tokens))
    assert usage["completion_tokens"] == len(ids), (usage, len(ids))
    if forced:
        assert len(ids) == output_tokens, (len(ids), output_tokens, finish)
    assert times, "No generated token IDs were streamed"
    assert all(math.isfinite(x) for x in values), "Non-finite returned log probability"
    itls = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
    # Multi-token chunks have unobservable internal timing. Preserve them in
    # raw data and flag them instead of inventing per-token measurements.
    multi = sum(len(c["token_ids"]) > 1 for c in chunks)
    record = {"request_id": rid, "phase": phase, "trial": trial, "prompt_tokens": len(tokens), "completion_tokens": len(ids), "seed": payload["seed"], "request_start_monotonic_ns": start, "request_end_monotonic_ns": end, "ttft_ms": times[0] / 1e6, "last_token_ms": times[-1] / 1e6, "response_ms": (end - start) / 1e6, "headers_ms": (header_time - start) / 1e6, "amortized_itl_ms": (times[-1] - times[0]) / max(1, len(times) - 1) / 1e6, "itl_p50_ms": statistics.median(itls) if itls and not multi else None, "itl_p95_ms": percentile(itls, .95) if itls and not multi else None, "multi_token_chunks": multi, "token_elapsed_ns": times, "token_ids": ids, "output_text": "".join(texts), "finish_reason": finish, "usage": usage, "server_metrics": server_metrics, "returned_logprobs_finite": all(math.isfinite(x) for x in values), "returned_logprob_count": len(values), "chunks": chunks}
    event("request_done", request_id=rid, kind=phase, completion_tokens=len(ids))
    return record


def percentile(items, q):
    items = sorted(items)
    if not items:
        return None
    p = (len(items) - 1) * q
    low, high = math.floor(p), math.ceil(p)
    return items[low] + (items[high] - items[low]) * (p - low)


rawfile = (out / "requests.jsonl").open("a", buffering=1)


def save(record):
    rawfile.write(json.dumps(record) + "\n")
    print(json.dumps({k: record.get(k) for k in ["request_id", "phase", "prompt_tokens", "completion_tokens", "ttft_ms", "amortized_itl_ms", "response_ms", "multi_token_chunks"]}), flush=True)


event("smoke_start")
smokes = []
for i, (question, expected) in enumerate([("What is two plus two? Reply with only the numeral.", r"\b4\b"), ("What is the capital of France? Reply with only its name.", r"\bParis\b")]):
    record = completion(chat_tokens(question), "numerical_smoke", i, output_tokens=32, forced=False, logprobs=5)
    record["expected_pattern"] = expected
    record["answer_check"] = bool(re.search(expected, record["output_text"], re.I))
    save(record)
    smokes.append(record)
(out / "numerical-smoke.json").write_text(json.dumps({"scope": "Finite returned sampled/top-5 logprobs and simple factual/arithmetical output checks; not a full BF16 or quantized numerical oracle.", "limited_smoke_checks_passed": all(x["answer_check"] and x["returned_logprobs_finite"] for x in smokes), "correctness_gate_passed": False, "gate_blocker": "Full frozen-checkpoint activation quantizer and broader quality/cache qualification remain pending; finite top-five probabilities are only a smoke check.", "cases": smokes}, indent=2))
if not all(x["answer_check"] for x in smokes):
    raise RuntimeError("Numerical/output smoke failed; benchmarks not started")

event("warmup_start")
for n in (2048, 8192):
    for i in range(2):
        save(completion(prompts[n], "warmup", i))
event("warmup_done")

records = []
schedule = [(n, i) for n in (2048, 8192) for i in range(args.trials)]
random.Random(9471).shuffle(schedule)
event("measurement_start", trials_per_context=args.trials, randomized_context_order=True)
for n, i in schedule:
    record = completion(prompts[n], "measurement", i)
    save(record)
    records.append(record)
event("measurement_done")

summary = {"qualification": "EXPLORATORY scale-corrected separate-projection runtime; bounded operator/full-model evidence is recorded separately. Frozen-checkpoint activation-quantizer equivalence and broader G1 quality/cache gates remain pending.", "timing_boundary": "Host-local HTTP client perf_counter_ns immediately before POST through SSE token delivery and [DONE]. Tokenization and JSON encoding excluded; server, head, sampling, cache, detokenization and host-local streaming included. SSH/Tailscale absent from timed path.", "fixed_length_policy": "Greedy 256 tokens, ignore_eos=True; natural-EOS numerical smokes recorded separately.", "warmup": "Two complete 256-token requests at each context before measurements; compile/load excluded.", "concurrency": 1, "contexts": {}}
for n in (2048, 8192):
    cells = [r for r in records if r["prompt_tokens"] == n]
    fields = ["ttft_ms", "amortized_itl_ms", "itl_p50_ms", "itl_p95_ms", "response_ms"]
    summary["contexts"][str(n)] = {"trials": len(cells), "prompt_tokens": n, "completion_tokens": 256, "multi_token_chunks": sum(r["multi_token_chunks"] for r in cells), **{f: {"p50": percentile([r[f] for r in cells if r[f] is not None], .5), "p95": percentile([r[f] for r in cells if r[f] is not None], .95)} for f in fields}}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({"summary": summary}), flush=True)

if args.profile:
    event("profile_start")
    response = session.post(args.url + "/start_profile", timeout=30)
    response.raise_for_status()
    try:
        save(completion(prompts[2048], "profile_separate", 0, output_tokens=32))
    finally:
        response = session.post(args.url + "/stop_profile", timeout=120)
        response.raise_for_status()
    event("profile_done")
events.close()
rawfile.close()
