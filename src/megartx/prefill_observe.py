"""Dormant callable instrumentation, with exact owner/source checks.

No global plugin registration, native imports or arithmetic replacement. Hooks
are installed only by an already admitted EngineCore-local provider; timing jobs
must never construct them. The provider owns actual cache, scheduler and CUPTI
observations, which cannot be inferred from Python module event ranges.
"""

import inspect
from pathlib import Path

from . import prefill_runner as runner


class OwnedHooks:
    """Wrap actual instance callables and restore only this owner's replacement."""

    def __init__(self, collector, event_clock, bindings):
        if collector.job["mode"] != "profile" or not collector.job["observer_enabled"]:
            raise ValueError("Callable event hooks are profile-only")
        self.collector, self.clock, self.bindings = collector, event_clock, bindings
        self.hooks, self.closed = [], False

    def wrap(self, owner, attribute, site_id, layer, component, shape):
        """shape(args, kwargs) returns actual M/N/K without tensor copies.

        The saved callable must have the bound source module/qualname/file hash.
        Binding inventory comes from reviewed installed source, never from a
        guessed class name or mutable upstream main. No weight/KV/route mutation.
        """
        if self.closed or any(o is owner and a == attribute for o, a, *_ in self.hooks):
            raise ValueError("Closed or duplicate callable hook owner")
        binding = self.bindings[site_id]
        runner.keys(binding, {"module", "qualname", "file_sha256", "site_sha256"}, "callable binding")
        for key in ("file_sha256", "site_sha256"):
            runner.sha(binding[key], key)
        saved = getattr(owner, attribute)
        function = getattr(saved, "__func__", saved)
        if (getattr(function, "__module__", None) != binding["module"]
                or getattr(function, "__qualname__", None) != binding["qualname"]):
            raise ValueError("Actual callable module/qualname differs: " + site_id)
        file = inspect.getsourcefile(function)
        if file is None:
            raise ValueError("Callable lacks inspectable source: " + site_id)
        file = Path(file)
        if file.is_symlink() or not file.is_file() or file.stat().st_size > 2**20:
            raise ValueError("Callable source absent/unbounded/symlink")
        import hashlib
        if hashlib.sha256(file.read_bytes()).hexdigest() != binding["file_sha256"]:
            raise ValueError("Actual callable source drift: " + site_id)
        # Retain absence of a per-instance override so detach preserves class dispatch.
        local = attribute in vars(owner)
        original_local = vars(owner).get(attribute)

        def observed(*args, **kwargs):
            if self.closed or getattr(owner, attribute) is not observed:
                self.collector.abort()
                raise ValueError("Callable observation ownership changed")
            try:
                with self.collector.stage(component, layer, site_id, binding["site_sha256"],
                                          shape(args, kwargs), self.clock):
                    return saved(*args, **kwargs)
            except BaseException:
                self.collector.abort()
                raise

        setattr(owner, attribute, observed)
        self.hooks.append((owner, attribute, observed, local, original_local))
        return observed

    def close(self):
        conflicts = []
        for owner, attribute, observed, local, original in reversed(self.hooks):
            if getattr(owner, attribute) is not observed:
                conflicts.append(attribute)
                continue
            if local:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)
        self.closed = True
        if conflicts:
            self.collector.abort()
            raise ValueError("Hook ownership changed; foreign replacement preserved: " + ",".join(conflicts))

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if value is None:
                raise
            if hasattr(value, "add_note"):
                value.add_note(str(cleanup))
        return False


def bind_gemma_modules(model, runtime_sites):
    """Discover thirty literal layer instances and selected real module callables.

    This is callable inventory, not hook installation or GPU qualification. It
    deliberately exposes combined routed/shared scopes where the pinned library
    does not expose individual launches. CUPTI/native providers must separately
    resolve fused suboperations, dispatch, bytes and scheduled expert geometry.
    """
    import re
    result, seen = [], set()
    for path, layer in model.named_modules():
        match = re.fullmatch(r"(?:.*\.)?layers\.(\d+)", path)
        if match is None:
            continue
        i = int(match[1])
        if not 0 <= i < 30 or i in seen:
            raise ValueError("Duplicate/out-of-scope Gemma layer owner")
        if type(layer).__module__ != "vllm.model_executor.models.gemma4" or type(layer).__name__ != "Gemma4DecoderLayer":
            raise ValueError("Unexpected Gemma layer class")
        seen.add(i)
        if layer.layer_idx != i:
            raise ValueError("Layer ordinal differs from literal module path")
        attention = layer.self_attn
        if (attention.is_kv_shared_layer or attention.is_sliding != (i not in runner.GLOBAL_LAYERS)
                or attention.head_dim != (512 if i in runner.GLOBAL_LAYERS else 256)
                or attention.num_kv_heads != (2 if i in runner.GLOBAL_LAYERS else 8)
                or attention.num_heads != 16):
            raise ValueError("Pinned local/global geometry or cache owner changed")
        objects = {"attention": attention.attn, "qkv": attention.qkv_proj,
                   "o_projection": attention.o_proj, "shared_mlp": layer.mlp,
                   "router": layer.router, "dispatch": layer.moe,
                   "q_norm": attention.q_norm, "k_norm": attention.k_norm,
                   "v_norm": attention.v_norm, "rope": attention.rotary_emb}
        for name, owner in objects.items():
            if name not in runtime_sites:
                continue
            result.append({"layer": i, "component": runtime_sites[name]["component"],
                           "site_id": name, "owner": owner, "attribute": runtime_sites[name].get("attribute", "forward"),
                           "scope": runtime_sites[name]["scope"]})
    if seen != set(range(30)):
        raise ValueError("All thirty literal Gemma layer owners required")
    return result


class GemmaCallableObserver:
    """Install real per-instance hooks for one separate profiling request.

    Runtime source bytes are checked by every wrap, before invoking any saved
    callable. This does not install the shared scale plugin or change its routed
    callable (which has its own exact __func__ binding). Native substage/kernel
    attribution remains the external profiler provider's responsibility.
    """
    def __init__(self, model, collector, event_clock, sites):
        self.hooks = OwnedHooks(collector, event_clock, {k: v["binding"] for k, v in sites.items()})
        try:
            if collector.job["region"] == "head":
                language_model = model.language_model
                self.hooks.wrap(language_model, "compute_logits", "head", 29, "head",
                    lambda args, kwargs: (self._tensor(args, kwargs, "hidden_states").shape[0], 262144, 2816))
            else:
                for site in bind_gemma_modules(model, sites):
                    if site["component"] not in runner.PROFILE_COMPONENTS[collector.job["region"]]:
                        continue
                    self.hooks.wrap(site["owner"], site["attribute"], site["site_id"], site["layer"],
                        site["component"], self._shape(site, collector))
        except BaseException:
            self.hooks.close()
            raise

    @staticmethod
    def _tensor(args, kwargs, key):
        value = args[0] if args else kwargs.get(key)
        if value is None or not hasattr(value, "shape"):
            raise ValueError("Observed callable lacks the actual input tensor")
        return value

    @classmethod
    def _shape(cls, site, collector):
        def observed(args, kwargs):
            value = cls._tensor(args, kwargs, "x" if site["site_id"] not in {"attention", "qkv", "o_projection"} else
                                "query" if site["site_id"] == "attention" else "input_")
            m = value.shape[0]
            name, owner = site["site_id"], site["owner"]
            if name in {"qkv", "o_projection"}:
                return m, owner.output_size, owner.input_size
            if name == "attention":
                first, end = collector.active["start"], collector.active["end"]
                kv_start = 0 if site["layer"] in runner.GLOBAL_LAYERS else max(0, first - 1023)
                return m, end - kv_start, 512 if site["layer"] in runner.GLOBAL_LAYERS else 256
            if name in {"q_norm", "k_norm", "v_norm"}:
                return m, value.shape[-2] * value.shape[-1], value.shape[-1]
            return m, {"shared_mlp": 2112, "router": 128, "dispatch": 128}[name], 2816
        return observed

    def close(self):
        self.hooks.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        return self.hooks.__exit__(kind, value, traceback)
