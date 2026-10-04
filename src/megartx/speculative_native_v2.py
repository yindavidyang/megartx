"""Source-bound default V2 ownership. Observation frames grant no authority.

CPU imports only. No batch preparation, metadata build, forward, or allocation
is performed here. The target-only receipt does not instantiate a speculator.
"""
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import types

from .speculative_native_probe import ProbeError, digest, inspect_sources

BASE = "c50ba1410a5c35cd470139f2cf5e90984d7c88f2"
RUNNER_MODULE = "vllm.v1.worker.gpu.model_runner"
PROTOCOL = "docs/design/speculative-native-v2-zero-forward-protocol.json"
BINDING = "docs/evidence/speculative-native-v2-source-binding.json"
POLICY = {"runner_module": RUNNER_MODULE, "runner_class": "GPUModelRunner",
          "resolved_v2": True, "selector_env": None, "selection_override_injected": False}


def source_binding():
    return json.loads((Path(__file__).resolve().parents[2] / BINDING).read_text())


def inspect_v2_sources(root):
    """Read installed bytes, including unchanged historical dependencies."""
    root = Path(root)
    if digest(Path(__file__).resolve().parents[2] /
              "docs/evidence/speculative-native-source-binding.json") != source_binding()["historical_source_manifest_sha256"]:
        raise ProbeError("Historical source manifest changed; no automatic pin refresh")
    result = inspect_sources(root)
    for name, expected in source_binding()["additional_sources"].items():
        path = root / name
        if path.is_symlink() or digest(path) != expected:
            raise ProbeError("V2 installed source drift: " + name)
        result[name] = expected
    return result


def class_source(obj, module, name, root):
    """Require the actual loaded class, module origin and exact installed bytes."""
    cls, loaded = type(obj), sys.modules.get(module)
    path = Path(root) / (module.replace(".", "/") + ".py")
    sources = source_binding()["additional_sources"]
    if module.replace(".", "/") + ".py" not in sources:
        from .speculative_native_probe import source_manifest
        sources = {p: v["sha256"] for p, v in source_manifest()["files"].items()}
    if (cls.__module__ != module or cls.__name__ != name or loaded is None
            or vars(loaded).get(name) is not cls or path.is_symlink()
            or inspect.getsourcefile(cls) is None
            or Path(inspect.getsourcefile(cls)).resolve() != path.resolve()
            or Path(getattr(loaded, "__file__", "")).resolve() != path.resolve()
            or digest(path) != sources[module.replace(".", "/") + ".py"]):
        raise ProbeError("V2 loaded owner/source identity differs: " + module + "." + name)


def verify_method(obj, name, root):
    """Authenticate a getter's loaded code; hashing its file alone is insufficient."""
    cls = type(obj)
    path = Path(root) / (cls.__module__.replace(".", "/") + ".py")
    class_source(obj, cls.__module__, cls.__name__, root)
    source = path.read_bytes()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == cls.__name__)
    method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    compiled = compile(source, str(path), "exec", dont_inherit=True, optimize=sys.flags.optimize)
    code = next(c for c in compiled.co_consts if isinstance(c, types.CodeType) and c.co_name == cls.__name__)
    expected = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == name)
    callback = cls.__dict__.get(name)
    decorators = method.decorator_list
    if decorators:
        if len(decorators) != 1 or not isinstance(decorators[0], ast.Name) or decorators[0].id != "property" or type(callback) is not property:
            raise ProbeError("Unreviewed V2 getter decorator")
        callback = callback.fget
    defaults = tuple(ast.literal_eval(v) for v in method.args.defaults) or None
    if (name in vars(obj) or not inspect.isfunction(callback) or hasattr(callback, "__wrapped__")
            or callback.__module__ != cls.__module__ or callback.__qualname__ != cls.__name__ + "." + name
            or callback.__globals__ is not sys.modules[cls.__module__].__dict__
            or callback.__closure__ is not None or callback.__code__ != expected
            or callback.__defaults__ != defaults or callback.__kwdefaults__ is not None):
        raise ProbeError("Loaded V2 getter differs: " + name)


def require_config(config):
    if (os.environ.get("VLLM_USE_V2_MODEL_RUNNER") is not None
            or config.use_v2_model_runner is not True or config.is_mm_encoder_only
            or config.speculative_config is not None or config.cache_config.enable_prefix_caching
            or config.scheduler_config.async_scheduling is not False
            or not config.model_config.enforce_eager or config.parallel_config.world_size != 1
            or config.parallel_config.enable_dbo or config.kv_transfer_config is not None
            or config.ec_transfer_config is not None):
        raise ProbeError("Exact target-only synchronous eager default V2 configuration required")
    return dict(POLICY)


def validate_ticket(ticket):
    from .speculative_native_receipt import receipt_preflight, _sequence
    if (type(ticket) is not dict or set(ticket) != {"purpose", "nonce", "engine_pid", "engine_start", "block_sizes", "groups"}
            or type(ticket["engine_pid"]) is not int or ticket["engine_pid"] < 1
            or type(ticket["engine_start"]) is not str):
        raise ProbeError("Exact V2 reservation ticket fields required")
    receipt_preflight(ticket)
    _sequence(ticket["block_sizes"], 30, "V2 ticket sizes")
    _sequence(ticket["groups"], 30, "V2 ticket groups")
    for pages in ticket["groups"]:
        _sequence(pages, 130, "V2 ticket pages")


class V2Owner:
    """Retain actual owners through receipt and release; never adopt a frame."""
    def __init__(self, runner, ticket):
        validate_ticket(ticket)
        cls = type(runner)
        if cls.__module__ != RUNNER_MODULE or cls.__name__ != "GPUModelRunner":
            raise ProbeError("Actual default V2 runner required")
        self.root = Path(inspect.getsourcefile(cls)).resolve().parents[4]
        self.sources = inspect_v2_sources(self.root)
        class_source(runner, RUNNER_MODULE, "GPUModelRunner", self.root)
        verify_method(runner, "get_model", self.root)
        verify_method(runner.vllm_config, "use_v2_model_runner", self.root)
        self.runner, self.config = runner, runner.vllm_config
        require_config(self.config)
        from .speculative_native_receipt import _owner_limits, _sequence, RECEIPT_LIMITS
        _owner_limits(runner, ticket)
        self.ticket = copy.deepcopy(ticket)
        for obj, module, name in ((runner.req_states, "vllm.v1.worker.gpu.states", "RequestState"),
                (runner.block_tables, "vllm.v1.worker.gpu.block_table", "BlockTables"),
                (runner.model_state, "vllm.v1.worker.gpu.model_states.default", "DefaultModelState")):
            class_source(obj, module, name, self.root)
        self.model = runner.get_model()
        class_source(self.model, "vllm.model_executor.models.gemma4_mm", "Gemma4ForConditionalGeneration", self.root)
        self.refs = (runner.req_states, runner.block_tables, runner.model_state, runner.kv_cache_config,
                     runner.compilation_config.static_forward_context, runner.attn_groups)
        self.kernel_sizes = tuple(_sequence(runner.kernel_block_sizes, RECEIPT_LIMITS["cache_groups"], "V2 kernel sizes"))
        self.builders, self.caches = [], []
        names = []
        for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            if group.host_resident or group.is_eagle_group:
                raise ProbeError("Host/draft V2 cache groups require a separate ownership protocol")
            covered = []
            for ag in runner.attn_groups[gid]:
                verify_method(ag, "get_metadata_builder", self.root)
                builder = ag.get_metadata_builder(0)
                class_source(builder, "vllm.v1.attention.backends.flashinfer", "FlashInferMetadataBuilder", self.root)
                if len(ag.metadata_builders) != 1 or builder.page_size != ticket["block_sizes"][gid]:
                    raise ProbeError("Unreviewed V2 metadata owner or page size")
                _sequence(ag.layer_names, RECEIPT_LIMITS["cache_layers"], "V2 builder layers")
                covered.extend(ag.layer_names)
                self.builders.append((ag, builder, tuple(ag.layer_names)))
            if len(covered) != len(group.layer_names) or set(covered) != set(group.layer_names):
                raise ProbeError("V2 metadata builders do not cover their exact cache group")
            for name in group.layer_names:
                layer = self.refs[4][name]
                self.caches.append((name, layer, layer.impl, layer.kv_cache))
                names.append(name)
        if len(names) != 30 or len(set(names)) != 30:
            raise ProbeError("Exactly thirty distinct V2 cache layers required")
        if len({id(b) for _, b, _ in self.builders}) != len(self.builders):
            raise ProbeError("V2 metadata builders share an owner")
        self.builder_pages = tuple(b.page_size for _, b, _ in self.builders)
        ids = [page for group in ticket["groups"] for page in group]
        if any(type(page) is not int or page < 1 for page in ids) or len(ids) != len(set(ids)):
            raise ProbeError("Null or aliased V2 reservation")
        self.sizes = tuple(ticket["block_sizes"])
        self.topology = self._topology()
        self.storage = None
        self.check()

    def _topology(self):
        return (tuple((id(g), id(g.kv_cache_spec), g.kv_cache_spec.block_size,
                       tuple(g.layer_names), g.host_resident, g.is_eagle_group)
                      for g in self.runner.kv_cache_config.kv_cache_groups),
                tuple((id(p), p.size, p.offset, p.layer_stride, p.block_stride, tuple(p.layers), p.host_resident)
                      for p in self.runner.kv_cache_config.kv_cache_tensors),
                tuple(tuple(id(ag) for ag in groups) for groups in self.runner.attn_groups))

    def seal_storage(self, layers):
        self.storage = [{key: record[key] for key in ("owner", "device", "dtype", "storage_ptr",
                         "storage_bytes", "storage_offset_bytes", "shape", "strides")} for record in layers]

    def check(self):
        runner = self.runner
        from .speculative_native_receipt import _owner_limits, _sequence
        _owner_limits(runner, self.ticket)
        for values in (runner.kernel_block_sizes, runner.block_tables.block_sizes,
                       runner.block_tables.kernel_block_sizes, runner.block_tables.blocks_per_kv_block):
            _sequence(values, 30, "V2 block geometry")
        verify_method(runner, "get_model", self.root)
        verify_method(self.config, "use_v2_model_runner", self.root)
        for ag, _, _ in self.builders:
            verify_method(ag, "get_metadata_builder", self.root)
        require_config(self.config)
        current = (runner.req_states, runner.block_tables, runner.model_state, runner.kv_cache_config,
                   runner.compilation_config.static_forward_context, runner.attn_groups)
        if (runner.vllm_config is not self.config or any(a is not b for a, b in zip(current, self.refs))
                or any(getattr(runner, field) is not getattr(self.config, field) for field in
                       ("model_config", "cache_config", "parallel_config", "compilation_config"))
                or runner.get_model() is not self.model or runner.model_state.model is not self.model
                or type(runner.req_states.req_id_to_index) is not dict or runner.req_states.req_id_to_index
                or type(runner.req_states.index_to_req_id) is not dict or runner.req_states.index_to_req_id
                or runner.execute_model_state is not None or runner.speculator is not None
                or runner.model_state.rope_state is not None
                or any(getattr(runner, name) is not None for name in
                       ("pcp_manager", "ubatch_runner", "batch_sharder", "fast_prefill"))
                or tuple(runner.kernel_block_sizes) != self.kernel_sizes
                or self.kernel_sizes != self.sizes
                or tuple(runner.block_tables.block_sizes) != self.sizes
                or tuple(runner.block_tables.kernel_block_sizes) != self.sizes
                or runner.block_tables.num_kv_cache_groups != len(self.sizes)
                or tuple(runner.block_tables.blocks_per_kv_block) != (1,) * len(self.sizes)):
            raise ProbeError("V2 owner changed, work pending, or manager/kernel splitting unsupported")
        if self._topology() != self.topology:
            raise ProbeError("V2 owner topology changed")
        if (any(ag.get_metadata_builder(0) is not builder or tuple(ag.layer_names) != names
                for ag, builder, names in self.builders)
                or tuple(b.page_size for _, b, _ in self.builders) != self.builder_pages
                or any(self.refs[4].get(name) is not layer or layer.impl is not impl or layer.kv_cache is not cache
                       for name, layer, impl, cache in self.caches)):
            raise ProbeError("V2 cache/metadata owner reference changed")
        if self.storage is not None:
            from .speculative_native_receipt import _tensor_record
            if self.storage != [_tensor_record(cache, name) for name, _, _, cache in self.caches]:
                raise ProbeError("V2 cache storage identity changed")

    def private_identity(self):
        return {"runner_policy": dict(POLICY), "request_state_identity": id(self.refs[0]),
                "block_tables_identity": id(self.refs[1]), "model_state_identity": id(self.refs[2]),
                "metadata_builder_identities": [id(b) for _, b, _ in self.builders],
                "mutable_verifier_lease_granted": False, "drafter_loaded": False,
                "metadata_built": False, "input_batch_prepared": False}


def freeze(project):
    """Freeze source/protocol for independent CPU review, never GPU admission."""
    project = Path(project).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(project), *args], text=True).strip()
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ProbeError("V2 freeze requires clean committed source")
    subprocess.run(["git", "-C", str(project), "merge-base", "--is-ancestor", BASE, "HEAD"], check=True)
    from .speculative_native_plan import OWNED_FILES, object_digest, LIMITS, engine_kwargs
    from .speculative_native_v2_lifecycle import WORKER_EXTENSION
    kwargs = {**engine_kwargs(), "worker_extension_cls": WORKER_EXTENSION}
    owned = (*OWNED_FILES, "src/megartx/speculative_native_v2.py",
             "src/megartx/speculative_native_v2_receipt.py", "src/megartx/speculative_native_v2_lifecycle.py", PROTOCOL, BINDING)
    result = {"schema": "megartx-native-v2-source-review-plan-v1", "source_base": BASE,
              "source_head": git("rev-parse", "HEAD"), "source_tree": git("rev-parse", "HEAD^{tree}"),
              "source_sha256": {p: digest(project / p) for p in owned}, "runner_policy": dict(POLICY),
              "installed_source_binding": source_binding(), "limits": dict(LIMITS),
              "prospective_engine_kwargs": kwargs,
              "gpu_authorized": False, "target_probe_authorized": False, "drafter_authorized": False,
              "preflight_blockers": ["independent_V2_review_pending", "shared_V2_registration_and_client_not_composed",
                                     "parent_GPU_slot_not_assigned", "actual_fit_FFI_temporary_bounds_unresolved"]}
    result["plan_sha256"] = object_digest(result)
    return result
