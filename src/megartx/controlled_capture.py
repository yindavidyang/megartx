"""Explicit artificial routing and private bounded evidence, never timing."""
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from .m1_execution import EXECUTION_MODES

ORIGIN = {"route_origin": "controlled", "routing_intervention": True,
          "routing_unchanged": False, "scope": "controlled_routing_fixture"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bits(tensor):
    import torch
    value = tensor.detach().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.int16).cpu().numpy().view(np.uint16)
    if value.dtype == torch.float32:
        return value.view(torch.int32).cpu().numpy().view(np.uint32)
    return value.view(torch.uint8).cpu().numpy()


def save_npz(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return digest(path)


class ControlledCapture:
    def __init__(self, model, layers, mode, execution_mode="captured"):
        import controlled_reference as ref
        self.ref, self.mode = ref, mode
        if execution_mode not in EXECUTION_MODES or (execution_mode == "capture-free" and mode != "native"):
            raise RuntimeError("Unknown or incompatible controlled execution mode")
        self.diagnostics = execution_mode == "captured"
        self.output = Path(os.environ["MEGARTX_CONTROLLED_DIR"])
        self.output.mkdir(exist_ok=False)
        self.marker = self.output.parent / "capture-request.json"
        source = Path(os.environ["MEGARTX_CONTROLLED_PLAN"])
        manifest = json.loads((source / "manifest.json").read_text())
        ref._controlled_origin(manifest)
        with np.load(source / "route-table.npz", allow_pickle=False) as arrays:
            self.plan = {**ORIGIN, **{k: arrays[k].copy() for k in ("tokens", "ids", "weight_bits")},
                         "token_sha256": manifest["token_sha256"], "schedule_sha256": manifest["schedule_sha256"]}
        ref.validate_schedule(self.plan)
        if mode not in {"native", "paired_reference", "gate_only_negative_control"}:
            raise RuntimeError("Controlled capture requires the declared arithmetic lanes")
        self.layers = {}
        for layer in layers:
            match = re.search(r"layers\.(\d+)\.", layer.layer_name)
            if match is None or int(match[1]) != layer._megartx["ordinal"]:
                raise RuntimeError("Original checkpoint and live routed-layer ordinals differ")
            if int(match[1]) in self.layers:
                raise RuntimeError("Duplicate live routed-layer identity")
            self.layers[int(match[1])] = layer
        if set(self.layers) != set(range(30)):
            raise RuntimeError("Controlled capture requires all thirty exact layer identities")
        from .controlled_kv_capture import ControlledKV
        self.kv = ControlledKV(model, self.plan, self.mode) if self.diagnostics else ControlledKV(
            model, self.plan, self.mode, diagnostics=False)
        self.context = None
        self.case = None
        self.profiler = None
        self.layer0 = None
        if os.environ.get("MEGARTX_LAYER0_BOUNDARIES") == "1":
            from .controlled_layer0_capture import ControlledLayer0
            self.layer0 = ControlledLayer0(model, self)

    @property
    def active(self):
        return self.context is not None

    def begin(self, input_ids, positions):
        import torch
        if not self.marker.exists():
            self.context = None
            return
        request = json.loads(self.marker.read_text())
        if any(request.get(k) is not v if type(v) is bool else request.get(k) != v for k, v in ORIGIN.items()):
            raise RuntimeError("Controlled request origin is missing or contradictory")
        case = request["controlled_path"]
        if case not in {"full", "cached", "chunked"} or request["schedule_sha256"] != self.plan["schedule_sha256"]:
            raise RuntimeError("Unknown or changed controlled request plan")
        if not self.diagnostics and case != "cached":
            raise RuntimeError("Capture-free execution requires the bounded cached controlled request")
        if input_ids is None or input_ids.ndim != 1 or positions.ndim != 1 or input_ids.numel() != positions.numel():
            raise RuntimeError("Controlled input needs actual one-sequence token/position rows")
        pos = positions.long().cpu().numpy()
        tokens = input_ids.long().cpu().numpy()
        if not 1 <= len(pos) <= 33 or len(set(pos.tolist())) != len(pos) or np.any(pos < 0) or np.any(pos >= 33) or not np.array_equal(tokens, self.plan["tokens"][pos]):
            raise RuntimeError("Controlled forward positions/tokens differ from immutable inputs")
        if case != self.case:
            if self.profiler is not None or (self.case is not None and not self.complete):
                raise RuntimeError("Previous controlled request was not completed")
            self.case = case
            self.directory = self.output / case
            if self.diagnostics:
                self.directory.mkdir(exist_ok=False)
            self.records, self.routes_seen, self.stages = [], {i: [] for i in range(30)}, []
            self.logit_counter, self.forward_counter, self.complete = 0, 0, False
            self.logit_positions = set()
            self.kv.begin_case(self.directory)
            if self.diagnostics:
                self.profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                self.profiler.__enter__()
        if self.complete or self.forward_counter >= 4:
            raise RuntimeError("Repeated or excessive controlled model forward")
        self.context = {"positions": pos, "tokens": tokens, "seen": set()}

    def abort(self, error):
        self.context = None
        if self.layer0 is not None:
            self.layer0.close()
        if self.profiler is not None:
            self.profiler.__exit__(type(error), error, error.__traceback__)
            self.profiler = None
        if self.case is not None and self.diagnostics:
            (self.directory / "INVALIDATED.json").write_text(json.dumps({**ORIGIN,
                "reason": "Controlled model capture failed", "error": str(error),
                "quality_gate_passed": False, "timing_qualified": False}, indent=2))

    def routes(self, layer, x, original_ids, original_weights):
        import torch
        if not self.active:
            return original_ids, original_weights
        ordinal = layer._megartx["ordinal"]
        if self.layers.get(ordinal) is not layer or ordinal in self.context["seen"]:
            raise RuntimeError("Controlled route hook changed or duplicated a live layer")
        pos = self.context["positions"]
        if x.dtype != torch.bfloat16 or tuple(x.shape) != (len(pos), 2816) or tuple(original_ids.shape) != (len(pos), 8) or tuple(original_weights.shape) != (len(pos), 8) or original_weights.dtype != torch.float32:
            raise RuntimeError("Controlled actual expert/route shape or dtype differs")
        if any(int(p) in self.routes_seen[ordinal] for p in pos):
            raise RuntimeError("A controlled logical position was processed twice")
        ids_np = self.plan["ids"][ordinal, pos].copy()
        weight_bits = self.plan["weight_bits"][ordinal, pos].copy()
        ids = torch.as_tensor(ids_np, device=original_ids.device, dtype=original_ids.dtype).clone()
        weights = torch.as_tensor(weight_bits.view(np.float32), device=original_weights.device).clone()
        if self.diagnostics:
            file = f"routes-{self.forward_counter:02d}-{ordinal:02d}.npz"
            sha = save_npz(self.directory / file, {"positions": pos, "tokens": self.context["tokens"], "ids": ids.int().cpu().numpy(), "weight_bits": bits(weights)})
            self.records.append({"file": file, "sha256": sha, "layer": ordinal, "layer_name": layer.layer_name, "forward": self.forward_counter})
        self.context["seen"].add(ordinal)
        self.routes_seen[ordinal].extend(pos.tolist())
        return ids, weights

    def expert(self, layer, expert, rows, slots, weights, stages, ordinary, combined):
        import torch
        if not self.active:
            raise RuntimeError("Controlled stage collection has no active input context")
        positions = self.context["positions"][rows.long().cpu().numpy()]
        key = (layer._megartx["ordinal"], expert.index)
        if (key not in self.ref.TARGET_POSITIONS or len(positions) != 1
                or int(positions[0]) != self.ref.TARGET_POSITIONS[key]):
            raise RuntimeError("Controlled expert did not execute its single declared row")
        if any((record["layer"], record["expert"]) == key for record in self.stages):
            raise RuntimeError("Controlled expert executed its declared row twice")
        down = stages["down"]
        if not torch.isfinite(down).all() or int((down != 0).sum().item()) == 0 or not bool((weights > 0).all()):
            raise RuntimeError("Controlled expert did not produce finite nonzero output with positive weight")
        if not self.diagnostics:
            # Preserve live row/finite/nonzero/positive-weight checks; the
            # retained stage count is an execution bound, not saved evidence.
            self.stages.append({"layer": layer._megartx["ordinal"], "expert": expert.index})
            return
        arrays = {}
        for name in ("input", "gate", "up", "activation", "down", "gate_alpha", "up_alpha", "down_alpha", "quant1_global", "quant2_global"):
            arrays[name + "_bits"] = bits(stages[name])
        for name in ("q1", "sf1", "q2", "sf2"):
            arrays[name] = bits(stages[name])
        arrays.update(a1_bits=bits(expert.a1), a2_bits=bits(expert.a2), route_weight_bits=bits(weights),
                      ordinary_bits=bits(ordinary[rows]), routed_bits=bits(combined[rows]),
                      weighted_bits=bits(down.float() * weights.float().unsqueeze(1)), positions=positions,
                      token_ids=self.plan["tokens"][positions])
        ordinal = layer._megartx["ordinal"]
        file = f"stage-{ordinal:02d}-{expert.index:03d}.npz"
        sha = save_npz(self.directory / file, arrays)
        self.stages.append({**ORIGIN, "file": file, "stage_capture_sha256": sha, "mode": self.mode,
                            "layer": ordinal, "layer_name": layer.layer_name, "expert": expert.index,
                            "position": int(positions[0]), "slot": int(slots[0].item()), "weight_bits": int(bits(weights)[0]),
                            "completed": True, "finite_output": True, "nonzero_output_elements": int((down != 0).sum().item()),
                            "registered_binding_verified": True,
                            "controlled_count_before": layer._megartx["controlled_hits"][expert.index] - 1,
                            "controlled_count_after": layer._megartx["controlled_hits"][expert.index]})

    def end(self):
        import torch
        if not self.active:
            return
        try:
            if self.context["seen"] != set(range(30)):
                raise RuntimeError("Controlled model bypassed a registered routed hook")
            self.kv.capture(self.context["positions"], self.context["tokens"])
            if self.layer0 is not None:
                self.layer0.end_forward()
            self.forward_counter += 1
            if all(sorted(rows) == list(range(33)) for rows in self.routes_seen.values()):
                if len(self.stages) != 6:
                    raise RuntimeError("Controlled model missed positive correction coverage")
                if {(record["layer"], record["expert"]) for record in self.stages} != set(self.ref.TARGETS):
                    raise RuntimeError("Controlled model executed the wrong correction set")
                if not self.diagnostics:
                    self.kv.finish_case()
                    self.complete = True
                    return
                torch.cuda.synchronize()
                self.profiler.__exit__(None, None, None)
                trace = self.directory / "controlled-trace.json.gz"
                self.profiler.export_chrome_trace(str(trace))
                self.profiler = None
                self.kv.finish_case()
                if self.layer0 is not None:
                    self.layer0.finish()
                proof = Path(os.environ["MEGARTX_ACTIVATION_PROOF_PATH"])
                proof_sha, trace_sha = digest(proof), digest(trace)
                for record in self.stages:
                    record.update(execution_proof_sha256=proof_sha, cuda_trace_sha256=trace_sha)
                report = {**ORIGIN, "mode": self.mode, "path": self.case, "token_sha256": self.plan["token_sha256"],
                          "schedule_sha256": self.plan["schedule_sha256"], "routes": self.records,
                          "executed_interventions": self.stages, "execution_proof_sha256": proof_sha,
                          "cuda_trace_sha256": trace_sha, "input_tokens": 33, "forward_calls": self.forward_counter,
                          "natural_hits": {str(i): self.layers[i]._megartx["natural_hits"] for i in range(30)},
                          "controlled_hits": {str(i): self.layers[i]._megartx["controlled_hits"] for i in range(30)},
                          "quality_gate_passed": False}
                with (self.directory / "controlled-manifest.json").open("x") as stream:
                    json.dump(report, stream, indent=2)
                self.complete = True
        finally:
            self.context = None

    def logits(self, result, positions, tokens, method, source_rows):
        if not self.marker.exists() or self.case is None:
            return
        indices = [i for i, p in enumerate(positions) if p in (31, 32)]
        if not indices:
            return
        if self.logit_counter >= 8:
            raise RuntimeError("Controlled raw-logit capture bound exceeded")
        selected_pos = np.asarray([positions[i] for i in indices], dtype=np.int64)
        actual_tokens = np.asarray([tokens[i] for i in indices], dtype=np.int64)
        if not np.array_equal(actual_tokens, self.plan["tokens"][selected_pos]):
            raise RuntimeError("Controlled raw-logit rows have wrong actual tokens")
        if not self.diagnostics:
            import torch
            selected = result[indices]
            if tuple(selected.shape) != (len(indices), 262144) or not bool(torch.isfinite(selected).all()):
                raise RuntimeError("Controlled raw logits are not finite full-vocabulary rows")
            self.logit_positions.update(int(p) for p in selected_pos)
            if self.complete and self.logit_positions != {31, 32}:
                raise RuntimeError("Both actual input positions 31 and 32 need validated pre-sampler raw logits")
            self.logit_counter += 1
            return
        # .cpu() is a separate copy before the sampler mutates CUDA storage.
        logits = result[indices].detach().float().cpu().numpy().copy()
        if logits.shape != (len(indices), 262144) or not np.isfinite(logits).all():
            raise RuntimeError("Controlled raw logits are not finite full-vocabulary rows")
        file = f"logits-{self.logit_counter:02d}.npz"
        sha = save_npz(self.directory / file, {"input_positions": selected_pos, "input_token_ids": actual_tokens, "logits": logits})
        with (self.directory / "logits-records.jsonl").open("a") as stream:
            stream.write(json.dumps({**ORIGIN, "file": file, "sha256": sha, "mode": self.mode, "path": self.case,
                                     "row_identity_verified": True, "row_correspondence": method,
                                     "source_rows": source_rows, "token_sha256": self.plan["token_sha256"],
                                     "schedule_sha256": self.plan["schedule_sha256"]}) + "\n")
        self.logit_counter += 1
