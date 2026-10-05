"""Exact-source CPU fixtures for the route-observer prototype.

These are source references, not installed vLLM/Triton pins. No third-party
runtime is imported. Supply only CPU mocks for globals used by loaded functions.
SOURCES retains byte-for-byte UTF-8 slices, including indentation and decorators.
load() dedents for parsing and strips only annotations/decorators in the AST.
Methods load as plain functions: e.g. property(load("CompiledKernel.run"))
restores the property descriptor. HookChain and LazyDict load as full classes.
No class/function source executes until explicitly loaded by a test.

vLLM excerpts: SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
Triton excerpts: upstream Triton reference sources; see manifest source URLs.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import textwrap
from typing import Callable, Generic, TypeVar

MANIFEST = json.loads(
    Path(__file__).with_name("route_observer_reference_manifest.json").read_text(
        encoding="utf-8"
    )
)

# Literal source slices are intentionally not reformatted.
SOURCES = {
    'MoERunner._apply_quant_method': r'''    def _apply_quant_method(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        shared_experts_overlapping: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | UnfinalizedMoEOutput]:
        """Run expert routing and the fused MoE kernel via the quant method.

        Orchestrates shared expert execution (before/after), expert selection
        via the router, and the actual fused MoE computation. Returns
        (shared_expert_output, fused_expert_output).

        `shared_experts_overlapping` should be True only if using multi-stream
        overlap. Then the shared expert was already launched in a separate
        stream, so the results only have to be awaited here.
        """
        self._maybe_apply_shared_experts(
            shared_experts_input, SharedExpertsOrder.NO_OVERLAP
        )

        if self.routed_experts.quant_method.is_monolithic:
            # Monolithic kernels: pass router_logits to routed_experts
            fused_out = self.routed_experts.forward_monolithic(
                x=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
        else:
            # Modular kernels: select experts first, then call routed_experts
            topk_weights, topk_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=self._quant_method.topk_indices_dtype,
                input_ids=input_ids,
            )

            fused_out = self.routed_experts.forward_modular(
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts=self._shared_experts,
                shared_experts_input=shared_experts_input,
            )

        if shared_experts_overlapping:
            assert self._shared_experts is not None
            self._shared_experts.wait()

        return (
            self._shared_experts.output if self._shared_experts is not None else None,
            fused_out,
        )
''',

    'FusedMoERouter.__init__': r'''    def __init__(self, eplb_state: EplbLayerState | None = None):
        self._routing_replay_out: torch.Tensor | None = None
        self.eplb_state = eplb_state
        # Deepseek V4.1 vision checkpoints carry a second per-expert routing
        # bias for image sentinel tokens; attached by model code when present.
        self.bias_vl: torch.Tensor | None = None
        self.image_sentinel_lo: int = 0
''',

    'FusedMoERouter.select_experts': r'''    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_indices_dtype: torch.dtype | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        Returns:
            (topk_weights, topk_ids)
            (tuple[torch.Tensor, torch.Tensor]):
            The weights and expert ids computation result.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """

        topk_weights, topk_ids = self._select_experts(
            hidden_states,
            router_logits,
            topk_indices_dtype=topk_indices_dtype,
            input_ids=input_ids,
        )

        # Write routing data for non-monolithic path (Triton, etc.)
        # (set by bind_routing_capture_to_model during capturer init)
        if self._routing_replay_out is not None:
            self._routing_replay_out[: topk_ids.shape[0]].copy_(
                topk_ids.to(torch.int16)
            )

        return topk_weights, topk_ids
''',

    'BaseRouter._validate_eplb_state': r'''    def _validate_eplb_state(self) -> None:
        """Validate that EPLB state is properly initialized if EPLB is enabled."""
        if self.eplb_state is not None:
            eplb_state = self.eplb_state
            if eplb_state.expert_load_view is None:
                raise ValueError("EPLB requires expert_load_view != None")
            if eplb_state.logical_to_physical_map is None:
                raise ValueError("EPLB requires logical_to_physical_map != None")
            if eplb_state.logical_replica_count is None:
                raise ValueError("EPLB requires logical_replica_count != None")
            if eplb_state.should_record_tensor is None:
                raise ValueError("EPLB requires should_record_tensor != None")
            if eplb_state.num_unpadded_tokens_tensors is None:
                raise ValueError("EPLB requires num_unpadded_tokens_tensors != None")
''',

    'BaseRouter._apply_eplb_mapping': r'''    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Apply EPLB mapping to convert logical expert IDs to physical expert IDs."""
        if self.eplb_state is not None:
            eplb_state = self.eplb_state
            assert eplb_state.expert_load_view is not None
            assert eplb_state.logical_to_physical_map is not None
            assert eplb_state.logical_replica_count is not None
            assert eplb_state.should_record_tensor is not None
            assert eplb_state.num_unpadded_tokens_tensors is not None
            return eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                logical_to_physical_map=eplb_state.logical_to_physical_map,
                logical_replica_count=eplb_state.logical_replica_count,
                expert_load_view=eplb_state.expert_load_view,
                record_enabled=eplb_state.should_record_tensor,
                num_unpadded_tokens=eplb_state.num_unpadded_tokens_tensors[
                    dbo_current_ubatch_id()
                ],
            )
        return topk_ids
''',

    'BaseRouter._convert_indices_dtype': r'''    def _convert_indices_dtype(
        self, topk_ids: torch.Tensor, indices_type: torch.dtype | None
    ) -> torch.Tensor:
        """Convert topk_ids to the desired dtype if needed."""
        if (indices_type is not None) and topk_ids.dtype != indices_type:
            topk_ids = topk_ids.to(dtype=indices_type)

        assert topk_ids.dtype == indices_type or indices_type is None
        return topk_ids
''',

    'BaseRouter._select_experts': r'''    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_indices_dtype: torch.dtype | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        This method implements the template method pattern:
        1. Validates EPLB state
        2. Calls _compute_routing() to get topk_weights and topk_ids
        3. Applies EPLB mapping if enabled
        4. Converts indices dtype if needed

        Returns:
            (topk_weights, topk_ids)
            (tuple[torch.Tensor, torch.Tensor]):
            The weights and expert ids computation result.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """
        # Step 1: Validate EPLB state
        self._validate_eplb_state()

        # Step 2: Compute routing (delegated to subclass)
        topk_weights, topk_ids = self._compute_routing(
            hidden_states, router_logits, topk_indices_dtype, input_ids=input_ids
        )

        # Capture logical ids before EPLB mapping.
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        # Step 3: Apply EPLB mapping
        topk_ids = self._apply_eplb_mapping(topk_ids)

        # Step 4: Convert indices dtype
        topk_ids = self._convert_indices_dtype(topk_ids, topk_indices_dtype)

        return topk_weights, topk_ids
''',

    'CustomRoutingRouter._compute_routing': r'''    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using the custom routing function."""
        topk_weights, topk_ids = self.custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )

        return topk_weights.to(torch.float32), topk_ids.to(
            torch.int32 if indices_type is None else indices_type
        )
''',

    'gemma4_fused_routing_kernel_triton': r'''def gemma4_fused_routing_kernel_triton(
    gating_output: torch.Tensor,
    topk: int,
    per_expert_scale: torch.Tensor,
    num_warps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    gating_output = gating_output.contiguous()
    per_expert_scale = per_expert_scale.contiguous()
    T, E = gating_output.shape
    weights = torch.empty(T, topk, dtype=torch.float32, device=gating_output.device)
    ids = torch.empty(T, topk, dtype=torch.int32, device=gating_output.device)
    BLOCK_E = triton.next_power_of_2(E)
    _gemma4_routing_kernel[(T,)](
        gating_output,
        per_expert_scale,
        weights,
        ids,
        E,
        topk,
        BLOCK_E,
        num_warps=num_warps,
    )
    return weights, ids
''',

    'KernelInterface.__getitem__': r'''    def __getitem__(self, grid) -> T:
        """
        A JIT function is launched with: fn[grid](*args, **kwargs).
        Hence JITFunction.__getitem__ returns a callable proxy that
        memorizes the grid.
        """
        return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
''',

    'compute_cache_key': r'''def compute_cache_key(kernel_key_cache, specialization, options):
    key = (tuple(specialization), str(options))
    cache_key = kernel_key_cache.get(key, None)
    if cache_key is not None:
        return cache_key

    # Replace JITCallable objects with their hash, so the cache key will change if the src is updated
    def replace_callables(obj):
        if isinstance(obj, list):
            return [replace_callables(arg) for arg in obj]
        elif is_namedtuple(obj):
            results = [replace_callables(arg) for arg in obj]
            return obj.__class__(*results)
        elif isinstance(obj, tuple):
            return tuple(replace_callables(arg) for arg in obj)
        elif isinstance(obj, JITCallable):
            return obj.cache_key
        return obj

    cache_key = str(replace_callables(specialization)) + str(options)
    kernel_key_cache[key] = cache_key
    return cache_key
''',

    'JITFunction.run': r'''    def run(self, *args, grid, warmup, **kwargs):
        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
        kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode

        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
        # specialization is list[tuple[str, Any]], where first element of tuple is
        # the type and the second parameter is the 'specialization' value.
        bound_args, specialization, options = binder(*args, **kwargs)

        # add a cache field to the kernel specializations for kernel specific
        # pass pipelines
        if knobs.runtime.add_stages_inspection_hook is not None:
            inspect_stages_key, inspect_stages_hash = knobs.runtime.add_stages_inspection_hook()
            specialization.append(f'("custom_pipeline", {inspect_stages_hash})')

        key = compute_cache_key(kernel_key_cache, specialization, options)
        kernel = kernel_cache.get(key, None)

        # Kernel is not cached; we have to compile.
        if kernel is None:
            options, signature, constexprs, attrs = self._pack_args(backend, kwargs, bound_args, specialization,
                                                                    options)

            kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)
            if kernel is None:
                return None

        # Check that used global values have not changed.
        not_present = object()
        for (name, _), (val, globals_dict) in self.used_global_vals.items():
            if (newVal := globals_dict.get(name, not_present)) != val:
                raise RuntimeError(
                    f"Global variable {name} has changed since we compiled this kernel, from {val} to {newVal}")

        if not warmup:
            # canonicalize grid
            assert grid is not None
            if callable(grid):
                grid = grid(bound_args)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1
            # launch kernel
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
        return kernel
''',

    'LazyDict': r'''class LazyDict:

    def __init__(self, data):
        self.data = data
        self.extras = []

    def get(self):
        for func, args in self.extras:
            self.data = self.data | func(*args)
        self.extras.clear()
        return self.data

    def add(self, func, args):
        self.extras.append((func, args))
''',

    'CompiledKernel._init_handles': r'''    def _init_handles(self):
        if self.module is not None:
            return

        def raise_(err):
            # clone the exception object so that the one saved in the closure
            # of the partial function below doesn't get assigned a stack trace
            # after the subsequent raise. otherwise, the CompiledKernel instance
            # saved in the (global) kernel cache will keep references to all the
            # locals in the traceback via the exception instance in the closure.
            cloned_err = copy.deepcopy(err)
            self._run = functools.partial(_raise_error, cloned_err)
            raise err

        device = driver.active.get_current_device()
        # create launcher
        self._run = driver.active.launcher_cls(self.src, self.metadata)
        # not enough shared memory to run the kernel
        max_shared = max_shared_mem(device)
        if self.metadata.shared > max_shared:
            raise_(OutOfResources(self.metadata.shared, max_shared, "shared memory"))
        if hasattr(self.metadata, "tmem_size") and self.metadata.tmem_size is not None:
            # Use blackwell max tmem size for now, this should be moved in device properties
            max_tmem_size = 512  # tmem size in number of columns
            if self.metadata.tmem_size > max_tmem_size:
                raise_(OutOfResources(self.metadata.tmem_size, max_tmem_size, "tensor memory"))
        if knobs.runtime.kernel_load_start_hook is not None:
            knobs.runtime.kernel_load_start_hook(self.module, self.function, self.name, self.metadata_group, self.hash)
        # TODO: n_regs, n_spills should be metadata generated when calling `ptxas`
        self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = driver.active.utils.load_binary(
            self.name, self.kernel, self.metadata.shared, device)
        warp_size = driver.active.get_current_target().warp_size
        if self.metadata.num_warps * warp_size > self.n_max_threads:
            raise_(OutOfResources(self.metadata.num_warps * warp_size, self.n_max_threads, "threads"))
        if knobs.runtime.kernel_load_end_hook is not None:
            knobs.runtime.kernel_load_end_hook(self.module, self.function, self.name, self.metadata_group, self.hash)
''',

    'CompiledKernel.run': r'''    @property
    def run(self):
        if self._run is None:
            self._init_handles()
        return self._run
''',

    'CompiledKernel.launch_metadata': r'''    def launch_metadata(self, grid, stream, *args):
        if knobs.runtime.launch_enter_hook is None:
            return None
        self._init_handles()
        ret = LazyDict({"name": self.name, "function": self.function, "stream": stream})
        if not isinstance(self.src, ASTSource) or self.src.fn.launch_metadata is None:
            return ret
        arg_dict = {name: arg for name, arg in zip(self.src.fn.arg_names, args)}
        ret.add(self.src.fn.launch_metadata, (grid, self.metadata, arg_dict))
        return ret
''',

    'CudaLauncher': r'''class CudaLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)

        launcher = triton.runtime.driver.active.utils.launch
        expanded_signature = expand_signature(signature.values(), tensordesc_meta)
        self.arg_annotations = annotate_arguments(expanded_signature)
        self.kernel_signature = make_kernel_signature(expanded_signature)
        self.num_ctas = getattr(metadata, "num_ctas", 1)
        self.launch = wrap_handle_tensordesc(launcher, signature, tensordesc_meta)
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.launch_pdl = metadata.launch_pdl

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                 launch_exit_hook, *args):

        def allocate_scratch(size, align, allocator):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * self.num_ctas * size
                alloc_fn = allocator.get()
                return alloc_fn(alloc_size, align, stream)
            return None

        global_scratch = allocate_scratch(self.global_scratch_size, self.global_scratch_align, _allocation._allocator)
        profile_scratch = allocate_scratch(self.profile_scratch_size, self.profile_scratch_align,
                                           _allocation._profile_allocator)

        self.launch(gridX, gridY, gridZ, stream, function, self.launch_cooperative_grid, self.launch_pdl,
                    kernel_metadata, launch_metadata, launch_enter_hook, launch_exit_hook, global_scratch,
                    profile_scratch, self.arg_annotations, self.kernel_signature, args)
''',

    'CudaLauncher.__init__': r'''    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)

        launcher = triton.runtime.driver.active.utils.launch
        expanded_signature = expand_signature(signature.values(), tensordesc_meta)
        self.arg_annotations = annotate_arguments(expanded_signature)
        self.kernel_signature = make_kernel_signature(expanded_signature)
        self.num_ctas = getattr(metadata, "num_ctas", 1)
        self.launch = wrap_handle_tensordesc(launcher, signature, tensordesc_meta)
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.launch_pdl = metadata.launch_pdl
''',

    'CudaLauncher.__call__': r'''    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                 launch_exit_hook, *args):

        def allocate_scratch(size, align, allocator):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * self.num_ctas * size
                alloc_fn = allocator.get()
                return alloc_fn(alloc_size, align, stream)
            return None

        global_scratch = allocate_scratch(self.global_scratch_size, self.global_scratch_align, _allocation._allocator)
        profile_scratch = allocate_scratch(self.profile_scratch_size, self.profile_scratch_align,
                                           _allocation._profile_allocator)

        self.launch(gridX, gridY, gridZ, stream, function, self.launch_cooperative_grid, self.launch_pdl,
                    kernel_metadata, launch_metadata, launch_enter_hook, launch_exit_hook, global_scratch,
                    profile_scratch, self.arg_annotations, self.kernel_signature, args)
''',

    'HookChain': r'''class HookChain(Generic[F]):
    """A chain of hooks of the same type F to be called in order.
    """

    def __init__(self, reversed: bool = False):
        self.calls: list[F] = []
        self.reversed = reversed

    def add(self, func: F) -> None:
        if func not in self.calls:
            self.calls.append(func)

    def remove(self, func: F) -> None:
        if func in self.calls:
            self.calls.remove(func)

    def __call__(self, *args, **kwargs):
        for call in self.calls if not self.reversed else reversed(self.calls):
            call(*args, **kwargs)
''',

    'HookChain.__init__': r'''    def __init__(self, reversed: bool = False):
        self.calls: list[F] = []
        self.reversed = reversed
''',

    'HookChain.__call__': r'''    def __call__(self, *args, **kwargs):
        for call in self.calls if not self.reversed else reversed(self.calls):
            call(*args, **kwargs)
''',

}


class _StripAnnotationsAndDecorators(ast.NodeTransformer):
    """Remove declaration metadata; retain executable statements and defaults."""

    def visit_FunctionDef(self, node):
        node.decorator_list = []
        node.returns = None
        node.type_comment = None
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        node.decorator_list = []
        return self.generic_visit(node)

    def visit_arg(self, node):
        node.annotation = None
        node.type_comment = None
        return node

    def visit_AnnAssign(self, node):
        # All annotated assignments in these excerpts have values. Reject new
        # bare declarations rather than inventing an executable replacement.
        if node.value is None:
            raise ValueError("bare annotated declaration is not supported")
        replacement = ast.Assign(targets=[node.target], value=node.value)
        return ast.copy_location(self.generic_visit(replacement), node)


def validate_sources():
    """Check every literal against its provenance entry without external files."""
    if SOURCES.keys() != MANIFEST["slices"].keys():
        raise ValueError("source and manifest names differ")
    for name, source in SOURCES.items():
        entry = MANIFEST["slices"][name]
        data = source.encode("utf-8")
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"source excerpt digest mismatch: {name}")
        if len(source.splitlines()) != entry["end_line"] - entry["start_line"] + 1:
            raise ValueError(f"source excerpt line count mismatch: {name}")
    return True


def load(name, namespace=None):
    """Load one definition with explicitly supplied CPU dependencies.

    A shallow namespace copy is used, so caller mappings are not modified.
    HookChain's unchanged Generic[F] base receives stdlib typing defaults.
    Missing runtime globals fail normally when the returned function is called.
    No upstream module imports, rewritten executable bodies, or runtime wiring
    occur here. In particular, callers own mocking driver/allocator operations.
    """
    source = SOURCES[name]
    entry = MANIFEST["slices"][name]
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != entry["sha256"]:
        raise ValueError(f"source excerpt digest mismatch: {name}")
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.ClassDef)):
        raise ValueError(f"expected one source definition: {name}")
    symbol = tree.body[0].name
    if symbol != name.rsplit(".", 1)[-1]:
        raise ValueError(f"source symbol mismatch: {name}")
    tree = _StripAnnotationsAndDecorators().visit(tree)
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, entry["start_line"] - 1)
    globals_ = {
        "__name__": "route_observer_reference",
        "Generic": Generic,
        "F": TypeVar("F", bound=Callable),
    }
    globals_.update(namespace or {})
    origin = MANIFEST["sources"][entry["source"]]
    filename = f"reference:{origin['revision']}:{entry['source']}"
    exec(compile(tree, filename, "exec"), globals_)
    return globals_[symbol]
