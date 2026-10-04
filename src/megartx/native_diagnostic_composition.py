"""CPU-only selection for mutually exclusive, source-bound diagnostic purposes.

Registration creates no engine or lease. The receipt client retains all admission,
EMPTY-engine, pause, reservation, drain, release and shutdown responsibilities.
"""
from pathlib import Path

PARENTS = ("418e1ecf5af0b5974ecf767a50c6d54f5578bee0",
           "8027588365ed4b23c6ccbc73de2c8f710e2e166f")
PROTOCOL = "docs/design/native-diagnostic-composition-protocol.json"
V1_FLAGS = ("MEGARTX_NATIVE_DIAGNOSTIC", "MEGARTX_NATIVE_RECEIPT_EVIDENCE")
V2_FLAGS = ("MEGARTX_NATIVE_V2_DIAGNOSTIC", "MEGARTX_NATIVE_V2_RECEIPT_EVIDENCE")
PREFILL_FLAGS = ("MEGARTX_PREFILL_NATIVE_PLAN", "MEGARTX_PREFILL_NATIVE_DIR",
                 "MEGARTX_PREFILL_NATIVE_SOURCE_ROOT", "MEGARTX_PREFILL_NATIVE_DEADLINE")
CLIENT_LANES = {"native-v1-receipt": "v1-legacy", "native-v2-receipt": "v2"}


V2_COMPOSITION = {"source_parents": list(PARENTS), "protocol": PROTOCOL,
    "launcher_client": "native-v2-receipt",
    "registration": ["megartx.speculative_native_v2_lifecycle.install_native_v2_diagnostic",
                     "megartx.speculative_native_evidence.install_native_v2_receipt_evidence"],
    "environment": {key: "1" for key in V2_FLAGS},
    "worker_extension_cls": "megartx.speculative_native_v2_lifecycle.NativeV2DiagnosticWorkerExtension",
    "resolved_runner_required": "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
    "prefill_observation_grants_mutation_lease": False,
    "target_probe_authorized": False, "drafter_authorized": False}


def diagnostic_mode(environment):
    """Reject mixed, partial or unknown opt-ins before runtime imports."""
    known = set(V1_FLAGS + V2_FLAGS + PREFILL_FLAGS)
    for key in environment:
        if key.startswith(("MEGARTX_NATIVE_", "MEGARTX_PREFILL_NATIVE_")) and key not in known:
            raise RuntimeError("Unknown native diagnostic hook: " + key)
    selected = []
    for mode, flags in (("v1-legacy", V1_FLAGS), ("v2", V2_FLAGS), ("prefill", PREFILL_FLAGS)):
        if any(key in environment for key in flags):
            if any(key not in environment or not environment[key] for key in flags):
                raise RuntimeError("Partial native diagnostic mode: " + mode)
            if mode != "prefill" and any(environment[key] != "1" for key in flags):
                raise RuntimeError("Native diagnostic opt-ins must be exactly 1")
            selected.append(mode)
    if "MEGARTX_M1_PREPARATION" in environment:
        if environment["MEGARTX_M1_PREPARATION"] not in ("stock", "fused"):
            raise RuntimeError("Unknown native M1 preparation mode")
        selected.append("native-m1")
    if len(selected) > 1:
        raise RuntimeError("Prefill, legacy V1, V2 and native M1 modes are mutually exclusive")
    if not selected:
        return None
    if (environment.get("MEGARTX_SCALE_MODE") != "native"
            or environment.get("VLLM_PLUGINS") != "megartx_scale_adapter"):
        raise RuntimeError("Native diagnostics require the exclusive native scale plugin")
    if selected[0] in ("v1-legacy", "v2", "prefill") and "VLLM_USE_V2_MODEL_RUNNER" in environment:
        raise RuntimeError("Native diagnostic runner selection forbids an inherited override")
    if selected[0] in ("v1-legacy", "v2") and any(
            key.startswith("MEGARTX_M1_") or key in (
                "MEGARTX_CONTROLLED_DIR", "MEGARTX_CONTROLLED_PLAN", "MEGARTX_LOGITS_DIR",
                "MEGARTX_ROUTE_AUDIT_PATH", "MEGARTX_ROUTING_COVERAGE_PATH", "MEGARTX_ROUTER_SCORE_DIR",
                "MEGARTX_LAYER0_BOUNDARIES", "MEGARTX_LAYER0_CAPTURE_POLICY")
            for key in environment):
        raise RuntimeError("Zero-forward receipt excludes unrelated observer/preparation hooks")
    return selected[0]


def install_diagnostic_hooks(selected):
    """Lifecycle first, evidence second, even before adapter idempotence checks.

    Duplicate or unknown installed utilities are rejected by the original
    source-verified installers. Read-only prefill never registers these utilities.
    """
    if selected == "v2":
        from .speculative_native_v2_lifecycle import install_native_v2_diagnostic
        from .speculative_native_evidence import install_native_v2_receipt_evidence
        if install_native_v2_diagnostic() is not True or install_native_v2_receipt_evidence() is not True:
            raise RuntimeError("Explicit V2 lifecycle/evidence registration did not complete")
    elif selected == "v1-legacy":
        from .speculative_native_lifecycle import install_native_diagnostic
        from .speculative_native_evidence import install_native_receipt_evidence
        if install_native_diagnostic() is not True or install_native_receipt_evidence() is not True:
            raise RuntimeError("Explicit legacy V1 lifecycle/evidence registration did not complete")
    elif selected not in (None, "prefill", "native-m1"):
        raise RuntimeError("Unknown native diagnostic registration mode")


def validate_launcher_environment(environment, client=None):
    # The launcher owns its child environment. Never silently strip or reinterpret
    # an inherited receipt/prefill flag as an authorization or another purpose.
    if any(key.startswith(("MEGARTX_NATIVE_", "MEGARTX_PREFILL_NATIVE_")) for key in environment):
        raise RuntimeError("Shared launcher requires explicit CLI selection without inherited diagnostic hooks")
    if client in CLIENT_LANES and ("VLLM_USE_V2_MODEL_RUNNER" in environment
            or any(key.startswith("MEGARTX_M1_") or key in (
                "MEGARTX_CONTROLLED_DIR", "MEGARTX_CONTROLLED_PLAN", "MEGARTX_LOGITS_DIR",
                "MEGARTX_ROUTE_AUDIT_PATH", "MEGARTX_ROUTING_COVERAGE_PATH", "MEGARTX_ROUTER_SCORE_DIR",
                "MEGARTX_LAYER0_BOUNDARIES", "MEGARTX_LAYER0_CAPTURE_POLICY") for key in environment)):
        raise RuntimeError("Receipt launcher excludes inherited runner overrides and other diagnostic modes")


def receipt_client_argv(args, project):
    """Validate CLI/source/purpose, then delegate the unchanged owned client."""
    from .speculative_native_plan import LIMITS, read_json, validate_plan, validate_authorization, plan_api
    lane = CLIENT_LANES.get(args.client)
    paths = (args.native_receipt_plan, args.native_receipt_authorization, args.native_receipt_directory)
    if lane is None:
        if any(value is not None for value in (*paths, args.native_receipt_checkpoint_manifest)):
            raise RuntimeError("Receipt arguments require their explicit native V1 or V2 receipt client")
        return None
    if (any(value is None for value in paths) or args.mode != "native" or args.trials != 1
            or args.backend != "flashinfer_cutlass" or args.kv != "bfloat16" or args.prefill_chunk != 256
            or args.m1_execution != "captured" or args.layer0_capture_policy != "synchronous"
            or any(getattr(args, name) for name in (
                "profile", "m1_decode_profile", "prefill_native_plan", "prefill_native_clearance",
                "m1_eager_benchmark_plan", "m1_private_aot", "m1_timing_metadata_help",
                "controlled_plan", "controlled_path", "layer0_boundaries", "activation_only",
                "routing_diagnostic", "router_score_only", "router_prefix_manifest", "m1_preparation",
                "m1_bridge", "m1_build_receipt", "m1_route_controls", "m1_normal_plan", "m1_external_observer"))):
        raise RuntimeError("Native receipt requires its exact default-off one-session plan/authorization and no other mode")
    plan = validate_plan(read_json(args.native_receipt_plan, LIMITS["plan_bytes"]), project)
    if plan_api(plan) is not plan_api(runner_lane=lane):
        raise RuntimeError("Shared launcher client and frozen receipt runner lane differ")
    validate_authorization(read_json(args.native_receipt_authorization, LIMITS["authorization_bytes"]), plan)
    result = ["--execute", "--plan", str(Path(args.native_receipt_plan).resolve()),
              "--authorization", str(Path(args.native_receipt_authorization).resolve()),
              "--private-directory", str(Path(args.native_receipt_directory).resolve())]
    if args.native_receipt_checkpoint_manifest is not None:
        result.extend(("--checkpoint-manifest", str(Path(args.native_receipt_checkpoint_manifest).resolve())))
    return result
