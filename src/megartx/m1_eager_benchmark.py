"""Bounded natural eager timing admission; no quality or graph admission.

Dynamic token/position copies, native guards and stream handoffs stay on the
measured path. Counters stay in memory until a separate, untimed drain request.
"""
import json
from pathlib import Path

from .m1_normal_plan import REVISION, digest

SCHEMA = "megartx-m1-eager-benchmark-v1"
CONTEXTS = (2048, 8192)
OUTPUTS = 256
DRIVER_SOURCES = ("scripts/m1_eager_benchmark_client.py", "scripts/prepare_m1_eager_benchmark.py",
                  "scripts/run_scale_validation.py")


def validate_plan(plan, source_hashes=None):
    from .m1_execution import CONTROLLER_SOURCES
    required = {"schema", "checkpoint_revision", "source_head", "controller_source_hashes", "driver_source_hashes",
                "outputs", "prefill_chunk", "warmups", "trials", "seed", "cases",
                "schedule", "plan_sha256"}
    if (set(plan) != required or plan["schema"] != SCHEMA
            or plan["checkpoint_revision"] != REVISION or type(plan["outputs"]) is not int
            or plan["outputs"] != OUTPUTS or type(plan["prefill_chunk"]) is not int
            or plan["prefill_chunk"] != 256):
        raise RuntimeError("eager benchmark scope differs")
    if (type(plan["warmups"]) is not int or not 1 <= plan["warmups"] <= 2
            or type(plan["trials"]) is not int or not 1 <= plan["trials"] <= 6
            or type(plan["seed"]) is not int or not 0 <= plan["seed"] < 2**31):
        raise RuntimeError("eager benchmark trial budget differs")
    hashes = plan["controller_source_hashes"]
    def sha(value, length=64):
        return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)
    if (not sha(plan["source_head"], 40) or not isinstance(hashes, dict)
            or set(hashes) != set(CONTROLLER_SOURCES) or not all(sha(h) for h in hashes.values())
            or (source_hashes is not None and hashes != source_hashes)):
        raise RuntimeError("eager benchmark source identity differs")
    drivers = plan["driver_source_hashes"]
    if not isinstance(drivers, dict) or set(drivers) != set(DRIVER_SOURCES) or not all(sha(h) for h in drivers.values()):
        raise RuntimeError("eager benchmark driver identity differs")
    cases = plan["cases"]
    if not isinstance(cases, list) or len(cases) != 2:
        raise RuntimeError("eager benchmark requires two contexts")
    for case, n in zip(cases, CONTEXTS):
        ids = case.get("prompt_token_ids")
        if (set(case) != {"id", "prompt_token_ids", "prompt_sha256"} or case["id"] != str(n)
                or not isinstance(ids, list) or len(ids) != n
                or any(type(t) is not int or not 0 <= t < 262144 for t in ids)
                or case["prompt_sha256"] != digest(ids)):
            raise RuntimeError("eager benchmark prompt identity differs")
    expected = {(phase, str(n), i, lane) for phase, count in (("warmup", plan["warmups"]), ("measurement", plan["trials"]))
                for n in CONTEXTS for i in range(count) for lane in ("stock", "fused")}
    schedule = plan["schedule"]
    if not isinstance(schedule, list) or len(schedule) != len(expected):
        raise RuntimeError("eager benchmark schedule differs")
    actual = []
    measurement_seen = False
    for row in schedule:
        if (set(row) != {"id", "phase", "case", "trial", "seed", "lane"} or type(row["trial"]) is not int
                or row["id"] != f'{row["phase"]}-{row["case"]}-{row["trial"]}-{row["lane"]}'
                or type(row["seed"]) is not int or row["seed"] != plan["seed"] + row["trial"]):
            raise RuntimeError("eager benchmark request identity differs")
        if row["phase"] == "measurement":
            measurement_seen = True
        elif measurement_seen:
            raise RuntimeError("eager benchmark warmup follows measurement")
        actual.append((row["phase"], row["case"], row["trial"], row["lane"]))
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise RuntimeError("eager benchmark schedule is incomplete or duplicated")
    measurements = [r for r in schedule if r["phase"] == "measurement"]
    for a, b in zip(measurements[::2], measurements[1::2]):
        if (a["case"], a["trial"], a["seed"]) != (b["case"], b["trial"], b["seed"]) or a["lane"] == b["lane"]:
            raise RuntimeError("eager benchmark trials must be adjacent matched pairs")
    for n in CONTEXTS:
        first_stock = sum(r["lane"] == "stock" for r in measurements[::2] if r["case"] == str(n))
        if abs(2 * first_stock - plan["trials"]) > 1:
            raise RuntimeError("eager benchmark pair order must be balanced")
    if plan["plan_sha256"] != digest({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise RuntimeError("eager benchmark plan digest differs")
    return plan


def load_plan(path, source_hashes=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 262144:
        raise RuntimeError("eager benchmark plan must be bounded regular JSON")
    return validate_plan(json.loads(path.read_text()), source_hashes)


def marker(plan, row):
    return {"schema": SCHEMA, "plan_sha256": plan["plan_sha256"], "id": row["id"]}


def drain_marker(plan):
    return {"schema": SCHEMA, "plan_sha256": plan["plan_sha256"], "id": "drain"}


class EagerBenchmark:
    """CPU ledger of real synchronous model frames, separate from collectors."""
    def __init__(self, plan, directory, lane):
        self.plan, self.directory, self.lane = plan, Path(directory), lane
        self.index, self.frame = -1, 0
        self.records, self.tokens = [], []
        self.counts = {"stock": 0, "fused": 0, "prefill_fallback": 0}
        self.pending = self.drained = False
        self.hits = {}
        self.call_limit = len(plan["schedule"]) * 30 * (OUTPUTS - 1)

    def finish_request(self):
        if self.index < 0:
            return
        row = self.plan["schedule"][self.index]
        n = int(row["case"])
        expected = {"stock": 30 * 255 if row["lane"] == "stock" else 0,
                    "fused": 30 * 255 if row["lane"] == "fused" else 0,
                    "prefill_fallback": 30 * (n // 256)}
        if self.pending or self.frame != n // 256 + 255 or self.counts != expected:
            raise RuntimeError("eager benchmark request dispatch/frame ledger incomplete")
        self.records.append({**row, "input_transcript_sha256": digest(self.tokens),
                             "frames": self.frame, "counts": dict(self.counts),
                             "natural_correction_selected_rows": dict(self.hits)})

    def begin(self, request, tokens, positions):
        if self.pending or self.drained:
            raise RuntimeError("eager benchmark nested or drained frame")
        if request == drain_marker(self.plan):
            if self.index != len(self.plan["schedule"]) - 1:
                raise RuntimeError("eager benchmark premature drain")
            self.finish_request()
            self.directory.mkdir(parents=True, exist_ok=True)
            report = {"schema": SCHEMA, "plan_sha256": self.plan["plan_sha256"],
                      "source_head": self.plan["source_head"], "records": self.records,
                      "observer_off": True, "quality_qualified": False, "graphs_qualified": False}
            with (self.directory / "dispatch.json").open("x") as stream:
                json.dump(report, stream, indent=2)
            self.drained = True
            return False
        if self.index < 0 or request != marker(self.plan, self.plan["schedule"][self.index]):
            self.finish_request()
            self.index += 1
            if self.index >= len(self.plan["schedule"]) or request != marker(self.plan, self.plan["schedule"][self.index]):
                raise RuntimeError("eager benchmark request order/identity differs")
            self.frame, self.tokens = 0, []
            self.counts = {"stock": 0, "fused": 0, "prefill_fallback": 0}
            self.hits = {}
            self.lane = self.plan["schedule"][self.index]["lane"]
        case = next(c for c in self.plan["cases"] if c["id"] == self.plan["schedule"][self.index]["case"])
        n = len(case["prompt_token_ids"])
        prefill = n // 256
        expected = list(range(self.frame * 256, (self.frame + 1) * 256)) if self.frame < prefill else [n + self.frame - prefill]
        if (self.frame >= prefill + 255 or positions != expected or len(tokens) != len(expected)
                or any(type(t) is not int or not 0 <= t < 262144 for t in tokens)
                or (self.frame < prefill and tokens != case["prompt_token_ids"][expected[0]:expected[-1] + 1])):
            raise RuntimeError("eager benchmark actual tokens/positions differ")
        self.tokens.extend(tokens)
        self.pending = True
        return True

    def backend(self, status):
        if not self.pending or status not in (0, 1):
            raise RuntimeError("eager benchmark backend outside active frame")
        self.counts["fused" if status == 1 else "stock"] += 1

    def correction_hit(self, layer, expert, rows):
        if not self.pending or type(rows) is not int or rows <= 0:
            raise RuntimeError("eager benchmark correction outside active frame")
        key = f"{layer}:{expert}"
        self.hits[key] = self.hits.get(key, 0) + rows

    def fallback(self):
        if not self.pending:
            raise RuntimeError("eager benchmark fallback outside active frame")
        self.counts["prefill_fallback"] += 1

    def end(self):
        if self.pending:
            n = int(self.plan["schedule"][self.index]["case"])
            # Native safety permits supported stock fallback; the measurement
            # ledger refuses to mislabel any such trial as fused preparation.
            expected = 30 * min(255, max(0, self.frame - n // 256 + 1))
            if (self.counts["stock"] + self.counts["fused"] != expected
                    or self.counts["prefill_fallback"] != 30 * min(self.frame + 1, n // 256)):
                raise RuntimeError("eager benchmark bypassed native dispatch")
            self.frame += 1
            self.pending = False
