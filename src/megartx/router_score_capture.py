"""Opt-in, bounded observations of one naturally routed Gemma4 request.

Hooks always return None. The production router, logits, IDs, weights, and
model outputs are never replaced. Raw inputs and checkpoint tensors stay in
the explicitly selected private evidence directory, outside the repository.
"""
import hashlib
import json
import os
from pathlib import Path
import re


TARGETS = {0: (42, 82), 1: (126,), 2: (89,), 3: (7,), 5: (12,)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RouterScoreCapture:
    def __init__(self, model, routed_layers):
        import numpy as np
        import torch
        from safetensors import safe_open
        from vllm.model_executor.models.gemma4 import Gemma4Router

        self.np, self.torch = np, torch
        self.directory = Path(os.environ["MEGARTX_ROUTER_SCORE_DIR"])
        self.directory.mkdir(parents=True, exist_ok=False)
        self.marker = self.directory.parent / "capture-request.json"
        self.active = None
        self.pending = {}
        self.checked = {layer: 0 for layer in TARGETS}
        self.decode_checked = {layer: 0 for layer in TARGETS}
        self.initial_prefill = set()
        self.counter = 0
        self.forward_id = 0
        self.request_identity = None
        self.handles = []
        modules = dict(model.named_modules())
        routers = {}
        for name, module in modules.items():
            if isinstance(module, Gemma4Router):
                match = re.search(r"(?:^|\.)layers\.(\d+)\.router$", name)
                if match is None:
                    raise RuntimeError("Unrecognized Gemma4 router module identity")
                index = int(match.group(1))
                if index in TARGETS:
                    if index in routers:
                        raise RuntimeError("Duplicate affected router layer")
                    routers[index] = (name, module)
        if set(routers) != set(TARGETS):
            raise RuntimeError("Missing affected router modules for score diagnostic")
        by_index = {}
        for layer in routed_layers:
            match = re.search(r"layers\.(\d+)\.", layer.layer_name)
            if match is None:
                raise RuntimeError("Unrecognized routed layer identity")
            index = int(match.group(1))
            if index in TARGETS:
                if index in by_index:
                    raise RuntimeError("Duplicate affected routed layer")
                by_index[index] = layer
        if set(by_index) != set(TARGETS):
            raise RuntimeError("Missing affected registered routed layers")
        self.routed_layers = by_index
        checkpoint = Path(os.environ["MEGARTX_CHECKPOINT_PATH"])
        weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        manifest = {"scope": "One observational request; no numerical or performance qualification", "routing_unchanged": True, "checked_rows_limit_per_layer": 8, "targets": [], "cuda_capability": list(torch.cuda.get_device_capability()), "matmul_flags": {name: getattr(torch.backends.cuda.matmul, name) for name in ("allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction_split_k")}}
        for index, (name, router) in sorted(routers.items()):
            layer = by_index[index]
            if tuple(e.index for e in layer._megartx["experts"]) != TARGETS[index]:
                raise RuntimeError("Router target IDs differ from original-scale captures")
            parent = modules[name.rsplit(".", 1)[0]]
            if getattr(parent, "router", None) is not router or getattr(parent, "moe", None) is None:
                raise RuntimeError("Router is not the original decoder-layer router")
            owner = parent.moe
            if owner.experts is not layer._megartx["registry"].get(layer.layer_name):
                raise RuntimeError("Router decoder and registered MoE runner do not match")
            proj = router.proj
            flags = {key: bool(getattr(proj, key)) for key in ("allow_specialized_router_gemm", "allow_fp32_router_gemm", "allow_bf16x3_router_gemm", "allow_ll_bf16_gemm", "allow_cublas_router_gemm")}
            if any(flags[key] for key in flags if key != "allow_cublas_router_gemm") or not flags["allow_cublas_router_gemm"] or proj.out_dtype != torch.float32 or proj.weight.dtype != torch.bfloat16 or proj.bias is not None:
                raise RuntimeError("Score diagnostic requires the observed BF16 cuBLAS/F32 router lane")
            tensors = {"weight_bits": ("router.proj.weight", proj.weight), "dimension_scale_bits": ("router.scale", router.scale), "expert_scale_bits": ("router.per_expert_scale", owner.per_expert_scale)}
            arrays, hashes = {}, []
            for label, (suffix, loaded) in tensors.items():
                key = f"model.language_model.layers.{index}.{suffix}"
                with safe_open(str(checkpoint / weight_map[key]), framework="pt", device="cpu") as archive:
                    original = archive.get_tensor(key).contiguous()
                observed = loaded.detach().cpu().contiguous()
                if original.dtype != torch.bfloat16 or observed.dtype != original.dtype or observed.shape != original.shape or not torch.equal(original.view(torch.uint8), observed.view(torch.uint8)):
                    raise RuntimeError("Loaded router tensor differs from immutable checkpoint: " + key)
                arrays[label] = original.view(torch.uint16).numpy().copy()
                hashes.append({"tensor": key, "dtype": str(original.dtype), "shape": list(original.shape), "bytes": original.numel() * original.element_size(), "sha256": hashlib.sha256(original.view(torch.uint8).numpy().tobytes()).hexdigest(), "loaded_original_bytes_equal": True})
            root = router.root_size.detach().cpu().contiguous()
            arrays["root_raw_bits"] = root.reshape(-1).view(torch.uint8).numpy().copy()
            arrays["root_cast_bf16_bits"] = root.to(torch.bfloat16).reshape(-1).view(torch.uint16).numpy().copy()
            path = self.directory / f"reference-layer-{index:02d}.npz"
            self.write_npz(path, arrays)
            manifest["targets"].append({"layer": index, "expert_ids": list(TARGETS[index]), "router_module_name": name, "projection_prefix": proj.prefix, "routed_layer_name": layer.layer_name, "loader_ordinal": layer._megartx["ordinal"], "projection_lane": "torch.mm(BF16,BF16,out_dtype=float32)", "projection_flags": flags, "root_storage_dtype": str(root.dtype), "root_storage_shape": list(root.shape), "norm_eps": router.norm.variance_epsilon, "norm_has_weight": router.norm.has_weight, "norm_pass_weight": router.norm.pass_weight, "norm_pass_weight_add": router.norm.pass_weight_add, "norm_variance_size_override": router.norm.variance_size_override, "reference_file": path.name, "reference_sha256": digest(path), "original_tensors": hashes})
            self.handles.append(router.register_forward_pre_hook(self.router_pre(index)))
            self.handles.append(router.norm.register_forward_hook(self.norm_post(index)))
            self.handles.append(router.proj.register_forward_pre_hook(self.proj_pre(index)))
            self.handles.append(router.proj.register_forward_hook(self.proj_post(index)))
        if any(row["norm_has_weight"] or row["norm_pass_weight"] or row["norm_pass_weight_add"] or row["norm_variance_size_override"] is not None for row in manifest["targets"]):
            raise RuntimeError("Diagnostic requires the original weightless RMS norm")
        self.write_json(self.directory / "capture-manifest.json", manifest)

    def write_json(self, path, value):
        with path.open("x") as f:
            json.dump(value, f, indent=2)

    def write_npz(self, path, arrays):
        with path.open("xb") as f:
            self.np.savez(f, **arrays)

    def bits(self, tensor):
        if tensor.dtype != self.torch.bfloat16:
            raise RuntimeError("Observed router input stage is not BF16")
        return tensor.detach().contiguous().view(self.torch.uint16).cpu().numpy().copy()

    def begin(self, input_ids, positions):
        if self.active is not None or self.pending:
            raise RuntimeError("Prior router diagnostic forward was not consumed")
        if not self.marker.exists():
            return
        request = json.loads(self.marker.read_text())
        prompt = request["prompt_token_ids"]
        identity = (request["id"], request["prompt_sha256"])
        if len(prompt) != 1025 or hashlib.sha256(json.dumps(prompt).encode()).hexdigest() != identity[1]:
            raise RuntimeError("Router score diagnostic requires the exact 1025-token prefix")
        if self.request_identity is None:
            self.request_identity = identity
        if self.request_identity != identity or self.forward_id >= 16:
            raise RuntimeError("Router score diagnostic is bounded to one request")
        if input_ids is None or positions.ndim != 1 or input_ids.ndim != 1 or input_ids.numel() != positions.numel() or positions.numel() > 256:
            raise RuntimeError("Router capture requires verified unpadded single-request rows")
        pos = positions.detach().long().cpu().numpy().copy()
        ids = input_ids.detach().long().cpu().numpy().copy()
        if (pos < 0).any() or (pos >= 1033).any() or any(p < 1025 and t != prompt[p] for p, t in zip(pos.tolist(), ids.tolist())):
            raise RuntimeError("Router capture does not match the supplied prefix/bounded decode")
        self.active = {"request": request, "positions": pos, "token_ids": ids, "forward_id": self.forward_id, "seen": set()}
        self.forward_id += 1

    def end(self):
        if self.active is not None and (self.active["seen"] != set(TARGETS) or self.pending):
            raise RuntimeError("A natural forward bypassed a router score capture")
        self.active = None

    def router_pre(self, layer):
        def hook(module, args):
            if self.active is None:
                return None
            if layer in self.pending or layer in self.active["seen"]:
                raise RuntimeError("Duplicate affected router within one forward")
            x = args[0]
            if tuple(x.shape) != (len(self.active["positions"]), 2816):
                raise RuntimeError("Router residual rows differ from model input positions")
            pos = self.active["positions"]
            chosen = []
            if layer not in self.initial_prefill and (pos < 1025).any():
                chosen += sorted(set((0, len(pos) // 2, len(pos) - 1)))
                self.initial_prefill.add(layer)
            chosen += [i for i, p in enumerate(pos) if p == 1024 and i not in chosen]
            for i, p in enumerate(pos):
                if p >= 1025 and self.decode_checked[layer] < 4:
                    chosen.append(i)
                    self.decode_checked[layer] += 1
            if self.checked[layer] + len(chosen) > 8:
                raise RuntimeError("Router input capture exceeds eight rows per layer")
            self.checked[layer] += len(chosen)
            indices = self.np.asarray(chosen, dtype=self.np.int64)
            self.pending[layer] = {"checked_indices": indices, "residual_bits": self.bits(x[chosen])}
            return None
        return hook

    def norm_post(self, layer):
        def hook(module, args, output):
            if self.active is not None:
                p = self.pending[layer]
                p["norm_bits"] = self.bits(output[p["checked_indices"].tolist()])
            return None
        return hook

    def proj_pre(self, layer):
        def hook(module, args):
            if self.active is not None:
                p = self.pending[layer]
                p["projection_bits"] = self.bits(args[0][p["checked_indices"].tolist()])
            return None
        return hook

    def proj_post(self, layer):
        def hook(module, args, output):
            if self.active is not None:
                logits = output[0]
                if logits.dtype != self.torch.float32 or tuple(logits.shape) != (len(self.active["positions"]), 128):
                    raise RuntimeError("Observed original router logits differ from F32 contract")
                self.pending[layer]["logits_f32"] = logits.detach().cpu().numpy().copy()
            return None
        return hook

    def consume(self, routed_layer, topk_ids, topk_weights):
        if self.active is None or "fixture_mode" in routed_layer._megartx:
            return
        match = re.search(r"layers\.(\d+)\.", routed_layer.layer_name)
        layer = int(match.group(1))
        if layer not in TARGETS:
            return
        if routed_layer is not self.routed_layers[layer]:
            raise RuntimeError("Score capture saw a different registered layer")
        p = self.pending.pop(layer)
        if set(p) != {"checked_indices", "residual_bits", "norm_bits", "projection_bits", "logits_f32"}:
            raise RuntimeError("Incomplete production router stages")
        p.update({"positions": self.active["positions"], "token_ids": self.active["token_ids"], "selected_ids": topk_ids.detach().int().cpu().numpy().copy(), "selected_weights_f32": topk_weights.detach().float().cpu().numpy().copy()})
        if p["selected_ids"].shape != (len(p["positions"]), 8) or p["selected_weights_f32"].shape != p["selected_ids"].shape:
            raise RuntimeError("Natural top-eight rows differ from observed router logits")
        if not self.np.isfinite(p["logits_f32"]).all() or not self.np.isfinite(p["selected_weights_f32"]).all():
            raise RuntimeError("Nonfinite production router data")
        path = self.directory / f"router-{self.counter:04d}-layer-{layer:02d}.npz"
        self.write_npz(path, p)
        row = {"layer": layer, "target_experts": list(TARGETS[layer]), "forward_id": self.active["forward_id"], "case_id": self.active["request"]["id"], "prompt_sha256": self.active["request"]["prompt_sha256"], "input_rows": len(p["positions"]), "checked_rows": len(p["checked_indices"]), "file": path.name, "sha256": digest(path), "routing_unchanged": True, "root_multiply_stage": "inline production operation; reconstructed only in independent CPU reference"}
        with (self.directory / "records.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.counter += 1
        self.active["seen"].add(layer)
