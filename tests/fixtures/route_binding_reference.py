"""Additive exact-source CPU fixtures for route binding and ownership.

No vLLM, Torch, Triton, CUDA, driver, or native module is imported. SOURCES
retains byte-for-byte UTF-8 slices, including original decorators/indentation.
load() strips only annotations/decorators, as in route_observer_reference.py.
Methods load as plain functions; callers restore property descriptors explicitly.
Constructors that use super() require a caller-supplied CPU super mock when run
standalone. Tests must supply all globals; defaults and statements stay intact.

load_routing_function(owner, namespace) places the exact nested routing_function
AST inside a synthetic factory(self), then returns the function. This is only
AST placement, not a rewritten routing body or execution of Gemma4MoE.__init__.
The function retains a real self closure cell, and reads self.per_expert_scale
and the routing/platform globals at call time. Its lexical qualname names the
synthetic fixture factory rather than claiming execution of the original init.

Manifest source digests match the saved intake, not an installed/loaded runtime.
Neither native-artifact validation nor GPU execution is performed or implied.

vLLM excerpts: SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import textwrap

MANIFEST = json.loads(
    Path(__file__).with_name("route_binding_reference_manifest.json").read_text(
        encoding="utf-8"
    )
)

# Literal source slices are intentionally not reformatted.
SOURCES = {
    'Gemma4MoE.__init__': r'''    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts

        # Per-expert output scale folded into routing weights so that
        # MoERunner's fused kernel computes: Σ_e (expert_e * w_e * scale_e)
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

        # Gemma4 routing: softmax over ALL experts → top-k → renormalize.
        # MoERunner's built-in fused_topk scopes softmax differently, so
        # a custom routing function is needed for numerical correctness.
        # NOTE: self.per_expert_scale is read at call time (not captured into
        # a local) so that torch.func.functional_call parameter substitution
        # reaches the routing function correctly.
        def routing_function(
            hidden_states: torch.Tensor,
            gating_output: torch.Tensor,
            topk: int,
            renormalize: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if current_platform.is_cuda_alike() or current_platform.is_xpu():
                return gemma4_fused_routing_kernel_triton(
                    gating_output, topk, self.per_expert_scale
                )

            return gemma4_routing_function_torch(
                gating_output, topk, self.per_expert_scale
            )

        # MoERunner experts with custom Gemma4 routing
        intermediate_size = getattr(
            config,
            "moe_intermediate_size",
            getattr(config, "expert_intermediate_size", None),
        )
        if intermediate_size is None:
            raise ValueError("Gemma4 MoE requires an expert intermediate size")

        self.experts = FusedMoEFactory(
            num_experts=config.num_experts,
            top_k=config.top_k_experts,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            custom_routing_function=routing_function,
            activation="gelu_tanh",
        )
''',

    'Gemma4MoE.__init__.routing_function': r'''        def routing_function(
            hidden_states: torch.Tensor,
            gating_output: torch.Tensor,
            topk: int,
            renormalize: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if current_platform.is_cuda_alike() or current_platform.is_xpu():
                return gemma4_fused_routing_kernel_triton(
                    gating_output, topk, self.per_expert_scale
                )

            return gemma4_routing_function_torch(
                gating_output, topk, self.per_expert_scale
            )
''',

    'Gemma4MoE.forward': r'''    def forward(self, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        return self.experts(x, router_logits)
''',

    'MoERunner.__init__': r'''    def __init__(
        self,
        layer_name: str,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_experts: RoutedExperts,
        enable_dbo: bool = False,
        gate: torch.nn.Module | None = None,
        shared_experts: torch.nn.Module | None = None,
        shared_expert_gate: torch.nn.Module | None = None,
        routed_input_transform: torch.nn.Module | None = None,
        routed_output_transform: torch.nn.Module | None = None,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.router = router
        self.routed_input_transform = routed_input_transform
        self.routed_output_transform = routed_output_transform
        self.routed_scaling_factor = routed_scaling_factor
        self.gate = gate
        self.shared_expert_gate = shared_expert_gate
        self.routed_experts = routed_experts
        self.enable_dbo = enable_dbo

        # When both gates are present and FSE is enabled, fuse their
        # weight matrices into [num_experts + num_shared, hidden] so one
        # F.linear produces combined logits. The topk kernel can then
        # apply routing softmax and shared expert activation (sigmoid)
        # in a single launch.
        self._fse_fuse_gate = gate is not None and shared_expert_gate is not None
        self._combined_gate_weight: torch.Tensor | None = None

        self._shared_experts: SharedExperts | None = None
        if shared_experts is not None:
            can_overlap = lambda: self._quant_method.mk_can_overlap_shared_experts
            self._shared_experts = SharedExperts(
                shared_experts,
                moe_config=moe_config,
                enable_dbo=enable_dbo,
                mk_can_overlap_shared_experts=can_overlap,
            )

        # Needed for string -> MoERunner layer lookup in custom ops.
        self.layer_name = layer_name

        self._forward_entry = self._select_forward()

        # For smuggling this layer into the fused moe custom op
        register_layer_for_moe_forward_op(get_current_vllm_config(), self)
''',

    'MoERunner._quant_method': r'''    @property
    def _quant_method(self) -> FusedMoEMethodBase:
        return self.routed_experts.quant_method
''',

    'MoERunner.is_monolithic': r'''    @property
    def is_monolithic(self) -> bool:
        return self.routed_experts.quant_method.is_monolithic
''',

    'CustomRoutingRouter.__init__': r'''    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        custom_routing_function: Callable,
        eplb_state: EplbLayerState | None = None,
        renormalize: bool = True,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )
        self.custom_routing_function = custom_routing_function
        self.renormalize = renormalize
''',

    'RoutedExperts.__init__': r'''    def __init__(
        self,
        layer_name: str,
        params_dtype: torch.dtype,
        moe_config: FusedMoEConfig,
        quant_config: QuantizationConfig | None,
        expert_map_manager: ExpertMapManager,
        ckpt_gate_proj_name: str = "gate_proj",
        ckpt_down_proj_name: str = "down_proj",
        ckpt_up_proj_name: str = "up_proj",
        is_fused_checkpoint_transposed: bool = False,
        #
        # Extra params that are needed by quant_methods, pass along for now
        # Prefer getting these from other sources, e.g. moe_config or
        # router object
        #
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__()
        self.layer_name = layer_name
        self.moe_config = moe_config
        self.quant_config = quant_config
        self.ckpt_gate_proj_name = ckpt_gate_proj_name
        self.ckpt_down_proj_name = ckpt_down_proj_name
        self.ckpt_up_proj_name = ckpt_up_proj_name
        self.is_fused_checkpoint_transposed = is_fused_checkpoint_transposed
        self.expert_map_manager = expert_map_manager
        self.hidden_size = moe_config.hidden_dim
        self.global_num_experts = moe_config.num_experts
        self.local_num_experts = moe_config.num_local_experts
        self.params_dtype = params_dtype

        # Register buffers for state_dict compatibility
        self.update_expert_map_info()

        self.rocm_aiter_fmoe_enabled = moe_config.rocm_aiter_fmoe_enabled

        # It would be good to eventually codify these in FusedMoEConfig
        # or some other config.
        self.top_k = self.moe_config.experts_per_token
        self.activation = self.moe_config.activation
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_limit = swiglu_limit
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input
        # End random parameters
        self._loaded_expert_biases: set[str] = set()

        self.quant_method = self._get_quant_method(
            self.layer_name,
            self.quant_config,
            self.moe_config,
        )

        # Round up hidden size and update moe_config.
        # TODO: move roundup to _get_quant_method?
        self.hidden_size, self.intermediate_size_per_partition = (
            self.quant_method.maybe_roundup_sizes(
                self.hidden_size,
                self.moe_config.intermediate_size_per_partition,
                self.moe_config.in_dtype,
                self.moe_config.moe_parallel_config,
            )
        )
        self.moe_config.hidden_dim = self.hidden_size
        self.moe_config.intermediate_size_per_partition = (
            self.intermediate_size_per_partition
        )

        if (
            self.moe_config.moe_parallel_config.enable_eplb
            and not self.quant_method.supports_eplb
        ):
            # TODO: Add support for additional quantization methods.
            # The implementation for other quantization methods does not
            # contain essential differences, but the current quant API
            # design causes duplicated work when extending to new
            # quantization methods, so I'm leaving it for now.
            # If you plan to add support for more quantization methods,
            # please refer to the implementation in `Fp8MoEMethod`.
            raise NotImplementedError(
                f"EPLB is not supported {self.quant_method.__class__.__name__}."
            )

        moe_quant_params: dict[str, Any] = {
            "num_experts": moe_config.num_local_experts,
            "hidden_size": self.hidden_size,
            "unpadded_hidden_size": self.moe_config.hidden_dim_unpadded,
            "intermediate_size_per_partition": (
                self.moe_config.intermediate_size_per_partition
            ),
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
            "global_num_experts": moe_config.num_experts,
        }

        self.quant_method.create_weights(layer=self, **moe_quant_params)

        self.lora_base_layer_prefix = ""
''',

    'RoutedExperts.forward_modular': r'''    def forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None" = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Execute routed experts using the quantization method's apply function.

        This is called by the runner after router selection (for modular kernels)
        quant_method.apply() which accesses the weights on this RoutedExperts
        instance.

        Args:
            x: Input tensor after any transforms
            topk_weights: Routing weights from router (for modular kernels)
            topk_ids: Selected expert IDs from router (for modular kernels)
            shared_experts: The shared experts (if any)
            shared_experts_input: Input for shared experts (if any)

        Returns:
            Output tensor from routed experts.
        """
        assert not self.quant_method.is_monolithic

        # Modular kernels use pre-computed routing
        return self.quant_method.apply(
            layer=self,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
''',

    'RoutedExperts.forward_monolithic': r'''    def forward_monolithic(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        """
        Execute routed experts using the quantization method's apply function.

        This is called by the runner after router selection (for modular kernels)
        or with router logits (for monolithic kernels). It delegates to
        quant_method.apply() which accesses the weights on this RoutedExperts
        instance.

        Args:
            x: Input tensor after any transforms
            router_logits: Router logits (for monolithic kernels)
            input_ids: input ids for DeepSeek V4

        Returns:
            Finalized routed states or a deferred-finalize output.
        """
        assert self.quant_method.is_monolithic

        # Monolithic kernels handle routing internally
        return self.quant_method.apply_monolithic(
            layer=self,
            x=x,
            router_logits=router_logits,
            input_ids=input_ids,
        )
''',

    'FusedMoEModularMethod.__init__': r'''    def __init__(
        self, old_quant_method: FusedMoEMethodBase, moe_kernel: FusedMoEKernel
    ):
        super().__init__(moe_kernel.moe_config)
        self.moe_quant_config = old_quant_method.moe_quant_config
        self.moe_kernel = moe_kernel
        self.old_quant_method = old_quant_method
        logger.debug("Swapping out %s", self.old_quant_method.__class__.__name__)
''',

    'FusedMoEModularMethod.apply': r'''    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
''',

    'FusedMoEKernelModularImpl.__init__': r'''    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        fused_experts: FusedMoEExpertsModular,
    ):
        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts
        self.shared_experts: SharedExperts | None = None
        moe_parallel_config = fused_experts.moe_config.moe_parallel_config
        self.moe_parallel_config = moe_parallel_config
        self.is_dp_ep = (
            moe_parallel_config is not None
            and moe_parallel_config.dp_size > 1
            and moe_parallel_config.use_ep
        )
''',

    'FusedMoEKernelModularImpl._prepare': r'''    def _prepare(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        ExpertTokensMetadata | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        The _prepare method is a wrapper around self.prepare_finalize.prepare
        that handles DBO and async.
        """

        if not self.prepare_finalize.supports_async():
            # We shouldn't be running an a2a kernel that doesn't
            # support async prepare/finalize
            # TODO(lucas): enable in follow-up
            assert not dbo_enabled()

            (
                a1q,
                a1q_scale,
                expert_tokens_meta,
                _expert_topk_ids,
                _expert_topk_weights,
            ) = self.prepare_finalize.prepare(
                hidden_states,
                topk_weights,
                topk_ids,
                global_num_experts,
                expert_map,
                apply_router_weight_on_input,
                self.fused_experts.quant_config,
                defer_input_quant=self.fused_experts.expects_unquantized_inputs,
            )
        else:
            # Overlap shared expert compute with all2all dispatch.
            dbo_maybe_run_recv_hook()
            prepare_ret = self.prepare_finalize.prepare_async(
                hidden_states,
                topk_weights,
                topk_ids,
                global_num_experts,
                expert_map,
                apply_router_weight_on_input,
                self.fused_experts.quant_config,
                defer_input_quant=self.fused_experts.expects_unquantized_inputs,
            )

            # TODO(lucas): refactor this in the alternative schedules followup
            # currently unpack if we have hook + receiver pair or just
            # receiver (see finalize_async docstring)
            hook, receiver = (
                prepare_ret if isinstance(prepare_ret, tuple) else (None, prepare_ret)
            )

            if hook is not None:
                if dbo_enabled():
                    # If DBO is being used, register the hook with the ubatch
                    # context and call it in dbo_maybe_run_recv_hook instead of
                    #  passing it to the receiver.
                    dbo_register_recv_hook(hook)
                    dbo_yield()
                else:
                    hook()

            (
                a1q,
                a1q_scale,
                expert_tokens_meta,
                _expert_topk_ids,
                _expert_topk_weights,
            ) = receiver()

        # Maybe prepare gathered topk_ids and topk_weights from other EP ranks.
        topk_ids = topk_ids if _expert_topk_ids is None else _expert_topk_ids
        topk_weights = (
            topk_weights if _expert_topk_weights is None else _expert_topk_weights
        )

        return a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights
''',

    'FusedMoEKernelModularImpl._finalize': r'''    def _finalize(
        self,
        output: torch.Tensor,
        fused_out: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        The _finalize method is a wrapper around self.prepare_finalize.finalize
        that handles DBO, async and shared expert overlap.

        Args:
            shared_experts: SharedExperts | None. The shared experts if any.
            shared_experts_input: Optional separate input for shared experts.
                When latent MoE is used, hidden_states is the latent-projected
                tensor (smaller dimension) used by routed experts, while
                shared_experts_input is the original hidden_states (full
                dimension) needed by the shared expert MLP.
        """
        if not self.prepare_finalize.supports_async():
            assert not dbo_enabled()

            self.prepare_finalize.finalize(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                self.fused_experts.finalize_weight_and_reduce_impl(),
            )
        else:
            finalize_ret = self.prepare_finalize.finalize_async(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                self.fused_experts.finalize_weight_and_reduce_impl(),
            )
            self._maybe_apply_shared_experts(shared_experts, shared_experts_input)

            # TODO(lucas): refactor this in the alternative schedules followup
            # currently unpack if we have hook + receiver pair or just
            # receiver (see finalize_async docstring)
            hook, receiver = (
                finalize_ret
                if isinstance(finalize_ret, tuple)
                else (None, finalize_ret)
            )

            if hook is not None:
                if dbo_enabled():
                    # If DBO is being used, register the hook with the ubatch
                    # context and call it in dbo_maybe_run_recv_hook instead of
                    #  passing it to the receiver.
                    dbo_register_recv_hook(hook)
                    dbo_yield()
                else:
                    hook()

            receiver()

        return output
''',

    'FusedMoEKernel.is_monolithic': r'''    @property
    def is_monolithic(self) -> bool:
        return isinstance(self.impl, FusedMoEKernelMonolithicImpl)
''',

    'FusedMoEKernel.prepare_finalize': r'''    @property
    def prepare_finalize(self) -> FusedMoEPrepareAndFinalize:
        return self.impl.prepare_finalize
''',

    'FusedMoEKernel.fused_experts': r'''    @property
    def fused_experts(self) -> FusedMoEExperts:
        return self.impl.fused_experts
''',

    'FusedMoEKernel.moe_config': r'''    @property
    def moe_config(self) -> FusedMoEConfig:
        return self.fused_experts.moe_config
''',

    'FusedMoEKernel.apply': r'''    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        shared_experts: SharedExperts | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert isinstance(self.impl, FusedMoEKernelModularImpl)
        return self.impl.apply(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
''',

    '_quantize_input': r'''def _quantize_input(
    a1: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    defer_input_quant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Defer input quant to moe kernel for backends (e.g. AITER, FI)
    # which use a single kernel call for quant + experts.
    if defer_input_quant:
        return a1, None

    input_sf = (
        quant_config.a1_gscale if quant_config.use_nvfp4_w4a4 else quant_config.a1_scale
    )
    a1q, a1q_scale = moe_kernel_quantize_input(
        a1,
        input_sf,
        quant_dtype=quant_config.quant_dtype,
        per_act_token_quant=quant_config.per_act_token_quant,
        block_shape=quant_config.block_shape,
        is_scale_swizzled=quant_config.is_scale_swizzled,
        mx_alignment=quant_config.mx_alignment,
    )

    return a1q, a1q_scale
''',

    'MoEPrepareAndFinalizeNoDPEPModular.prepare': r'''    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        a1q, a1q_scale = _quantize_input(a1, quant_config, defer_input_quant)

        return a1q, a1q_scale, None, None, None
''',

    'MoEPrepareAndFinalizeNoDPEPModular.finalize': r'''    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()
        weight_and_reduce_impl.apply(
            output=output,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
''',

    'MoEPrepareAndFinalizeNoDPEPMonolithic.prepare': r'''    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareMonolithicResultType:
        a1q, a1q_scale = _quantize_input(a1, quant_config, defer_input_quant)
        return a1q, a1q_scale, router_logits
''',

    'MoEPrepareAndFinalizeNoDPEPMonolithic.finalize': r'''    def finalize(
        self,
        fused_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        return fused_expert_output
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
        if node.value is None:
            raise ValueError("bare annotated declaration is not supported")
        replacement = ast.Assign(targets=[node.target], value=node.value)
        return ast.copy_location(self.generic_visit(replacement), node)


def validate_sources():
    """Validate literal digests and line counts without reading external files."""
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


def _definition_tree(name):
    source = SOURCES[name]
    entry = MANIFEST["slices"][name]
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != entry["sha256"]:
        raise ValueError(f"source excerpt digest mismatch: {name}")
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError(f"expected one source function: {name}")
    if tree.body[0].name != name.rsplit(".", 1)[-1]:
        raise ValueError(f"source symbol mismatch: {name}")
    tree = _StripAnnotationsAndDecorators().visit(tree)
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, entry["start_line"] - 1)
    return tree


def _compile(tree, name, namespace):
    entry = MANIFEST["slices"][name]
    origin = MANIFEST["sources"][entry["source"]]
    globals_ = {"__name__": "route_binding_reference"}
    globals_.update(namespace or {})
    filename = f"reference:{origin['revision']}:{entry['source']}"
    exec(compile(tree, filename, "exec"), globals_)
    return globals_


def load(name, namespace=None):
    """Load a plain method/function with explicitly supplied CPU dependencies.

    A shallow namespace copy is used. Missing runtime globals fail normally.
    Nested routing_function requires load_routing_function so that self remains
    a true closure cell rather than becoming a manufactured global variable.
    """
    if MANIFEST["slices"][name]["kind"] == "nested_function":
        raise ValueError("nested routing function requires load_routing_function")
    tree = _definition_tree(name)
    return _compile(tree, name, namespace)[tree.body[0].name]


def load_routing_function(owner, namespace=None):
    """Return the exact nested routing body with owner in its self closure cell.

    Only the factory wrapper and return are synthetic. The source function's
    executable statements, arguments, and defaults remain unchanged. This does
    not execute the source constructor or create any runtime/GPU objects.
    """
    name = "Gemma4MoE.__init__.routing_function"
    definition = _definition_tree(name).body[0]
    tree = ast.parse("def _make_reference_routing_function(self):\n    pass\n")
    factory = tree.body[0]
    factory.body = [definition, ast.Return(value=ast.Name(id=definition.name, ctx=ast.Load()))]
    factory.lineno = definition.lineno
    factory.end_lineno = definition.end_lineno
    factory.args.args[0].lineno = definition.lineno
    factory.args.args[0].end_lineno = definition.lineno
    ast.copy_location(factory.body[-1], definition)
    ast.fix_missing_locations(tree)
    globals_ = _compile(tree, name, namespace)
    return globals_[factory.name](owner)
