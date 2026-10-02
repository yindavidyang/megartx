"""One exact two-prompt correctness plan; router choices remain unchanged."""
import hashlib
import json
from pathlib import Path

REVISION = "a19cfe00be84568a6867111c9a68c9c44fdcffe6"
ORIGIN = {"route_origin": "natural", "routing_intervention": False,
          "routing_unchanged": True, "scope": "bounded_normal_routing_constrained_continuation"}
CASES = (("split257", 257), ("window1023", 1023))
LIVE_CALLS, FALLBACK_CALLS, MODEL_FORWARDS = 210, 150, 12


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def frames(case):
    n = len(case["prompt_token_ids"])
    return [list(range(start, min(start + 256, n))) for start in range(0, n, 256)] + [[n+i] for i in range(3)]


def validate_plan(plan):
    if (plan.get("schema") != "megartx-m1-normal-plan-v1"
            or plan.get("checkpoint_revision") != REVISION
            or plan.get("outputs") != 4 or plan.get("prefill_chunk") != 256
            or plan.get("expected_live_calls") != LIVE_CALLS
            or plan.get("expected_fallback_calls") != FALLBACK_CALLS
            or plan.get("expected_model_forwards") != MODEL_FORWARDS
            or any(plan.get(k) != v or type(plan.get(k)) is not type(v) for k,v in ORIGIN.items())):
        raise RuntimeError("normal M1 scope differs from the exact approved plan")
    token = plan.get("continuation_token_id")
    eos = plan.get("eos_token_ids")
    if (type(token) is not int or not 0 <= token < 262144 or not isinstance(eos, list)
            or not eos or any(type(t) is not int or not 0 <= t < 262144 for t in eos) or token in eos):
        raise RuntimeError("normal M1 continuation must be a declared non-EOS token")
    cases = plan.get("cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise RuntimeError("normal M1 requires exactly two declared prompts")
    for case, (name, length) in zip(cases, CASES):
        tokens = case.get("prompt_token_ids")
        if (case.get("id") != name or not isinstance(tokens, list) or len(tokens) != length
                or any(type(t) is not int or not 0 <= t < 262144 for t in tokens)
                or case.get("prompt_sha256") != digest(tokens)):
            raise RuntimeError("normal M1 prompt identity/length differs")
    body = {k:v for k,v in plan.items() if k != "plan_sha256"}
    if plan.get("plan_sha256") != digest(body):
        raise RuntimeError("normal M1 plan digest differs")
    return plan


def load_plan(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 65536:
        raise RuntimeError("normal M1 plan must be bounded regular JSON")
    return validate_plan(json.loads(path.read_text()))


def request(plan, case):
    return {**ORIGIN, "id": case["id"], "plan_sha256": plan["plan_sha256"],
            "prompt_sha256": case["prompt_sha256"], "teacher_forced": True,
            "continuation_constrained": True, "continuation_token_id": plan["continuation_token_id"],
            "prompt_token_ids": case["prompt_token_ids"] + [plan["continuation_token_id"]] * 3}
