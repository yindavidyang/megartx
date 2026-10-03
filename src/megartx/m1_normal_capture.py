"""Private observations of one fixed normal-routing plan, never quality/timing."""
import json
import os
from pathlib import Path
import re

import numpy as np

from .controlled_capture import bits, digest, save_npz
from .controlled_kv_capture import ControlledKV, gather_writer_rows, SOURCE_HASHES
from .m1_normal_plan import ORIGIN, frames, load_plan, request


class NormalKV(ControlledKV):
    def __init__(self, model):
        self.bind_owners(model)
        self.identities = None

    def capture_rows(self, directory, forward, positions, tokens):
        import torch
        slots, capacities, identities, bindings = self.read_frame(positions, tokens)
        if self.identities is not None and self.identities != identities:
            raise RuntimeError("normal K/V owners changed within the request")
        self.identities = identities
        records = []
        # Snapshot the actual final writer row immediately, before any window
        # recycling. No whole-cache copy or stale historical slot assumption.
        for layer, descriptor in enumerate(self.descriptors):
            slot = slots[layer][-1]
            keys, values = gather_writer_rows(self.layers[layer][2].kv_cache, [slot], descriptor["head_dim"], torch)
            name = f"kv-{forward:02d}-{layer:02d}.npz"
            sha = save_npz(directory/name, {"logical_positions": positions[-1:].copy(),
                "token_ids": tokens[-1:].copy(), "key_bits": keys, "value_bits": values})
            records.append({"file": name, "sha256": sha, "layer": layer, "forward_index": forward,
                            "writer_slot": slot, "capacity": capacities[layer], "binding": bindings[layer],
                            "position": int(positions[-1]), "source_sha256": SOURCE_HASHES,
                            "logical_mapping_verified": True,
                            "scope": "immediate_actual_writer_row_without_independent_scheduler_proof"})
        return records


class NormalCapture:
    def __init__(self, model, layers, mode):
        if mode != "native":
            raise RuntimeError("normal M1 capture requires native correction")
        self.plan = load_plan(os.environ["MEGARTX_M1_NORMAL_PLAN"])
        self.root = Path(os.environ["MEGARTX_M1_NORMAL_DIR"])
        self.root.mkdir(exist_ok=False)
        self.marker = self.root.parent / "capture-request.json"
        self.layers = {}
        for layer in layers:
            match = re.fullmatch(r"language_model.model.layers\.(\d+)\.moe.experts", layer.layer_name)
            if match is None or int(match[1]) != layer._megartx["ordinal"] or int(match[1]) in self.layers:
                raise RuntimeError("normal M1 routed layer identity differs")
            self.layers[int(match[1])] = layer
        if set(self.layers) != set(range(30)):
            raise RuntimeError("normal M1 needs all thirty registered layers")
        self.kv = NormalKV(model)
        self.context = self.profiler = self.case = None
        self.case_index = -1
        self.total_forwards = 0
        self.complete = False

    @property
    def active(self):
        return self.context is not None

    def begin(self, input_ids, positions):
        import torch
        if not self.marker.exists():
            if self.case is not None and not self.complete:
                raise RuntimeError("normal M1 request marker disappeared before completion")
            return
        marker = json.loads(self.marker.read_text())
        if self.case is None or marker["id"] != self.case["id"]:
            if self.case is not None and (not self.complete or self.logit_counter != 4):
                raise RuntimeError("prior normal M1 request was incomplete")
            self.case_index += 1
            if self.case_index >= 2:
                raise RuntimeError("normal M1 exceeded the two-prompt boundary")
            self.case = self.plan["cases"][self.case_index]
            self.directory = self.root/self.case["id"]
            self.directory.mkdir(exist_ok=False)
            self.records, self.roots, self.stages, self.kv_records = [], [], [], []
            self.forward_counter = self.logit_counter = 0
            self.complete = False
            self.kv.identities = None
            self.profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            self.profiler.__enter__()
        expected_request = request(self.plan, self.case)
        if marker != expected_request:
            raise RuntimeError("normal M1 request differs from immutable plan")
        if (self.complete or input_ids is None or input_ids.ndim != 1 or positions.ndim != 1
                or input_ids.numel() != positions.numel()):
            raise RuntimeError("normal M1 model frame is missing, nested or excessive")
        # Snapshot only completed producer data; this capture lane makes no timing claim.
        torch.cuda.synchronize()
        pos, tokens = positions.long().cpu().numpy().copy(), input_ids.long().cpu().numpy().copy()
        expected = frames(self.case)[self.forward_counter]
        if (pos.tolist() != expected or tokens.tolist() != [expected_request["prompt_token_ids"][p] for p in expected]):
            (self.directory/"rejected-frame.json").write_text(json.dumps({"forward_counter":self.forward_counter,
                "global_forward":self.total_forwards,"actual_positions":pos.tolist(),"actual_tokens":tokens.tolist(),
                "expected_positions":expected,"expected_tokens":[expected_request["prompt_token_ids"][p]for p in expected]}))
            raise RuntimeError("normal M1 actual input positions/tokens differ from fixed frames")
        self.context = {"positions": pos, "tokens": tokens, "seen": set(), "outputs_seen": set(),
                        "forward_index": self.total_forwards}

    def routes(self, layer, x, ids, weights):
        import torch
        if not self.active:
            return ids, weights
        index = layer._megartx["ordinal"]
        rows = len(self.context["positions"])
        if (self.layers.get(index) is not layer or index in self.context["seen"]
                or x.dtype != torch.bfloat16 or tuple(x.shape) != (rows,2816)
                or tuple(ids.shape) != (rows,8) or tuple(weights.shape) != (rows,8)
                or weights.dtype != torch.float32):
            raise RuntimeError("normal M1 actual routed input/owner differs")
        actual = ids.int().cpu().numpy()
        if np.any((actual < 0) | (actual >= 128)) or any(len(set(row.tolist())) != 8 for row in actual):
            raise RuntimeError("normal M1 requires eight distinct local router IDs")
        name = f"routes-{self.total_forwards:02d}-{index:02d}.npz"
        arrays = {"positions": self.context["positions"], "tokens": self.context["tokens"],
                  "ids": actual, "weight_bits": bits(weights)}
        if rows == 1:
            arrays["input_bits"] = bits(x)
        self.records.append({"file": name, "sha256": save_npz(self.directory/name, arrays),
            "layer": index, "forward_index": self.total_forwards, "routing_unchanged": True,
            "input_rows": rows, "classification": "single_row_prefill_tail" if self.case_index == 0 and self.forward_counter == 1 else "cached_decode" if rows == 1 else "prefill"})
        self.context["seen"].add(index)
        return ids, weights  # Exact caller-owned objects; no route replacement.

    def output(self, layer, result):
        if not self.active:
            return result
        index = layer._megartx["ordinal"]
        if index not in self.context["seen"] or index in self.context["outputs_seen"]:
            raise RuntimeError("normal M1 output lacks a unique actual route")
        self.context["outputs_seen"].add(index)
        if len(self.context["positions"]) == 1:
            value = result[1] if isinstance(result, tuple) else result
            name = f"output-{self.total_forwards:02d}-{index:02d}.npz"
            self.roots.append({"file": name, "sha256": save_npz(self.directory/name, {"routed_bits": bits(value)}),
                                 "layer": index, "forward_index": self.total_forwards})
        return result

    def expert(self, layer, expert, rows, slots, weights, stages, ordinary, combined):
        if not self.active or len(self.context["positions"]) != 1:
            raise RuntimeError("normal corrected-stage capture exceeds the M1 row boundary")
        arrays = {name: bits(value) for name,value in stages.items()}
        arrays.update(route_weight_bits=bits(weights), ordinary_bits=bits(ordinary), routed_bits=bits(combined),
                      rows=rows.long().cpu().numpy(), slots=slots.long().cpu().numpy())
        name = f"stage-{self.total_forwards:02d}-{layer._megartx['ordinal']:02d}-{expert.index:03d}.npz"
        self.stages.append({"file": name, "sha256": save_npz(self.directory/name, arrays),
                            "layer": layer._megartx["ordinal"], "expert": expert.index, "forward_index": self.total_forwards})

    def end(self):
        import torch
        if not self.active:
            return
        try:
            if self.context["seen"] != set(range(30)) or self.context["outputs_seen"] != set(range(30)):
                raise RuntimeError("normal M1 bypassed a registered route/output hook")
            self.kv_records.extend(self.kv.capture_rows(self.directory, self.total_forwards,
                self.context["positions"], self.context["tokens"]))
            self.forward_counter += 1
            self.total_forwards += 1
            if self.forward_counter == len(frames(self.case)):
                torch.cuda.synchronize()
                self.profiler.__exit__(None, None, None)
                trace = self.directory/"normal-trace.json.gz"
                self.profiler.export_chrome_trace(str(trace))
                self.profiler = None
                report = {**ORIGIN, "request": request(self.plan,self.case), "forward_calls": self.forward_counter,
                    "routes": self.records, "outputs": self.roots, "stages": self.stages, "kv": self.kv_records,
                    "trace_sha256": digest(trace), "execution_proof_sha256": digest(os.environ["MEGARTX_ACTIVATION_PROOF_PATH"]),
                    "quality_qualified": False, "graph_qualified": False, "timing_qualified": False}
                with (self.directory/"normal-manifest.json").open("x") as stream:
                    json.dump(report,stream,indent=2)
                self.complete = True
        finally:
            self.context = None

    def logits(self, result, positions, tokens, method, source_rows):
        expected = request(self.plan,self.case)["prompt_token_ids"]
        wanted = list(range(len(self.case["prompt_token_ids"])-1, len(expected)))
        selected = [i for i,p in enumerate(positions) if p in wanted]
        # The installed runner may evaluate a sampling head for an incomplete
        # prefill chunk. Those earlier rows are outside the four declared rows.
        if not selected and all(0 <= p < wanted[0] for p in positions):
            return
        if len(selected) != 1 or self.logit_counter >= 4 or positions[selected[0]] != wanted[self.logit_counter]:
            raise RuntimeError("normal M1 logits differ from the four declared prediction rows")
        index = selected[0]
        if tokens[index] != expected[positions[index]]:
            raise RuntimeError("normal M1 raw logits have the wrong actual token")
        values = result[index:index+1].detach().float().cpu().numpy().copy()
        if values.shape != (1,262144) or not np.isfinite(values).all():
            raise RuntimeError("normal M1 logits must be finite complete vocabulary rows")
        name = f"logits-{self.logit_counter:02d}.npz"
        sha = save_npz(self.directory/name,{"input_positions": np.asarray([positions[index]],dtype=np.int64),
            "input_token_ids": np.asarray([tokens[index]],dtype=np.int64), "logits": values})
        with (self.directory/"logits-records.jsonl").open("a") as stream:
            stream.write(json.dumps({"file": name, "sha256": sha, "row_correspondence": method,
                "source_rows": source_rows, "row_identity_verified": True})+"\n")
        self.logit_counter += 1

    def abort(self, error):
        self.context = None
        if self.profiler is not None:
            self.profiler.__exit__(type(error),error,error.__traceback__)
            self.profiler = None
        if self.case is not None:
            (self.directory/"INVALIDATED.json").write_text(json.dumps({"error": str(error), "qualified": False}))
