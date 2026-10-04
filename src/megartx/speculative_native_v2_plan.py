"""Explicit V2 client proposal/admission for the exact shared composition.

The legacy V1 plan remains a distinct schema. Selecting this lane observes the
resolved default V2; it never changes vLLM selection or grants draft/probe fit.
"""
import copy
import math
from pathlib import Path

from . import speculative_native_plan as legacy
from . import speculative_native_v2 as owner
from .native_diagnostic_composition import PARENTS, V2_COMPOSITION, diagnostic_mode
from .speculative_native_probe import ProbeError, ADAPTER_FILES, REVISION

PURPOSE = "first_v2_zero_forward_layout_workspace_measurement"
SCHEMA = "megartx-native-v2-receipt-client-plan-v1"
AUTH_SCHEMA = "megartx-native-v2-receipt-authorization-v1"
REVIEWED_OWNER = "95f7b20d91b08ed83e81fd3d9af82b91f3f6a31e"
PROTOCOL = "docs/design/speculative-native-v2-client-protocol.json"
CLIENT_FILES = ("src/megartx/speculative_native_v2_plan.py", PROTOCOL,
                "schemas/speculative-native-v2-receipt-client-plan.schema.json",
                "schemas/speculative-native-v2-receipt-authorization.schema.json",
                "src/megartx/native_diagnostic_composition.py",
                "docs/design/native-diagnostic-composition-protocol.json")
PREFLIGHT_BLOCKERS = []  # This exact source composes the shared hooks; review/CI/slot admission remains mandatory.
LIMITS = legacy.LIMITS
RUNNER_BINDING = {"expected_model_runner_class": owner.RUNNER_MODULE + ".GPUModelRunner",
    "resolved_config_requirement": "use_v2_model_runner_is_True_before_EngineCoreClient_make_client",
    "runner_environment_override": None, "actual_loaded_runner_identity": None,
    "automatic_selection_source": "V2_when_Triton_and_no_unsupported_features",
    "selection_source_sha256": legacy.RUNNER_SELECTION_SOURCES}
FORBIDDEN_ENV = ("MEGARTX_NATIVE_DIAGNOSTIC", "MEGARTX_NATIVE_RECEIPT_EVIDENCE",
    "VLLM_USE_V2_MODEL_RUNNER", "MEGARTX_M1_PREPARATION")


def engine_kwargs():
    from .speculative_native_v2_lifecycle import WORKER_EXTENSION
    return {**legacy.engine_kwargs(), "worker_extension_cls": WORKER_EXTENSION}


def engine_argv():
    from .speculative_native_v2_lifecycle import WORKER_EXTENSION
    result = legacy.engine_argv()
    result[result.index("--worker-extension-cls") + 1] = WORKER_EXTENSION
    return result


def environment(project, private):
    result = legacy.environment(project, private)
    del result["MEGARTX_NATIVE_DIAGNOSTIC"]
    del result["MEGARTX_NATIVE_RECEIPT_EVIDENCE"]
    result.update(MEGARTX_NATIVE_V2_DIAGNOSTIC="1", MEGARTX_NATIVE_V2_RECEIPT_EVIDENCE="1")
    if diagnostic_mode(result) != "v2":
        raise ProbeError("V2 client environment differs from shared composition")
    return result


def freeze(project, *, client_mode="async"):
    # Reuse the historical collector/identity verification without changing pins.
    historical = legacy.freeze(project, client_mode=client_mode)
    project = Path(project).resolve()
    legacy._git(project, "merge-base", "--is-ancestor", REVIEWED_OWNER, historical["source_head"])
    for parent in PARENTS:
        legacy._git(project, "merge-base", "--is-ancestor", parent, historical["source_head"])
    reviewed = owner.freeze(project)
    composition = legacy.read_json(project / V2_COMPOSITION["protocol"], 64 << 10)
    if (legacy.canonical(composition.get("v2")) != legacy.canonical(V2_COMPOSITION)
            or engine_kwargs()["worker_extension_cls"] != V2_COMPOSITION["worker_extension_cls"]
            or engine_argv()[engine_argv().index("--worker-extension-cls") + 1] != V2_COMPOSITION["worker_extension_cls"]):
        raise ProbeError("Shared V2 protocol/worker/argv binding differs")
    result = {**historical, "schema": SCHEMA, "purpose": PURPOSE, "runner_lane": "v2",
        "reviewed_owner_commit": REVIEWED_OWNER, "source_tree": reviewed["source_tree"],
        "v2_owner_plan_sha256": reviewed["plan_sha256"],
        "source_sha256": {**reviewed["source_sha256"],
            **{p: legacy.hash_file(project / p, 1 << 20) for p in CLIENT_FILES}},
        "engine_kwargs": engine_kwargs(), "engine_argv": engine_argv(),
        "runner_binding": copy.deepcopy(RUNNER_BINDING), "runner_policy": dict(owner.POLICY),
        "installed_v2_source_binding": owner.source_binding(), "drafter_authorized": False,
        "preflight_blockers": list(PREFLIGHT_BLOCKERS),
        "shared_composition": copy.deepcopy(V2_COMPOSITION)}
    del result["plan_sha256"]
    result["plan_sha256"] = legacy.object_digest(result)
    return copy.deepcopy(result)


def validate_plan(plan, project):
    if type(plan) is not dict or plan.get("schema") != SCHEMA or plan.get("purpose") != PURPOSE:
        raise ProbeError("Wrong V2 client plan schema or purpose")
    if legacy.canonical(plan) != legacy.canonical(freeze(project, client_mode=plan.get("client_mode"))):
        raise ProbeError("V2 client plan differs from exact committed source/configuration")
    return plan


def installed_preflight(root):
    result = legacy.installed_preflight(root)
    result["source_sha256"].update(owner.inspect_v2_sources(root))
    result["runner_policy"] = dict(owner.POLICY)
    return result


def validate_authorization(auth, plan):
    required = {"schema": AUTH_SCHEMA, "purpose": PURPOSE,
        "plan_sha256": plan["plan_sha256"], "source_head": plan["source_head"],
        "runner_lane": "v2", "v2_owner_plan_sha256": plan["v2_owner_plan_sha256"],
        "independent_review_clear": True, "exact_head_ci_green": True,
        "parent_source_protocol_accepted": True, "gpu_slot_assigned": True,
        "owned_lifecycle_verified": True, "target_probe_authorized": False, "drafter_authorized": False}
    refs = {"parent_slot", "parent_acceptance_reference", "independent_review_reference", "ci_reference"}
    if (type(auth) is not dict or set(auth) != set(required) | refs
            or any(type(auth.get(k)) is not type(v) or auth.get(k) != v for k, v in required.items())
            or any(type(auth[k]) is not str or not 1 <= len(auth[k]) <= 512 for k in refs)):
        raise ProbeError("Missing exact V2 client source/review/CI/parent-slot authorization")
    if (plan.get("schema") != SCHEMA or plan.get("purpose") != PURPOSE
            or plan.get("runner_lane") != "v2" or plan.get("runner_policy") != owner.POLICY
            or plan.get("gpu_authorized") is not False or plan.get("target_probe_authorized") is not False
            or plan.get("drafter_authorized") is not False or plan["preflight_blockers"]
            or plan.get("collector_materialization_guard") != legacy.COLLECTOR_GUARD
            or legacy.canonical(plan.get("shared_composition")) != legacy.canonical(V2_COMPOSITION)):
        raise ProbeError("V2 source preflight remains blocked; authorization cannot override composition or bounds")
    return auth


def receipt_admission(plan, auth, checkpoint, *, deadline):
    validate_authorization(auth, plan)
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ProbeError("Nonfinite V2 monotonic deadline")
    return {"phase": "zero_forward_v2_receipt", "independent_review_clear": True,
        "parent_source_protocol_accepted": True, "owned_lifecycle_verified": True,
        "max_concurrent_gpu_jobs": 1, "max_extra_gpu_bytes": 8 << 20,
        "compiler_memory_bytes": 2 << 30, "compiler_timeout_seconds": 300, "compiler_scope": "shared_host",
        "checkpoint_revision": REVISION, "checkpoint_manifest_verified": checkpoint["full_shards_rehashed"] is True,
        "tokenizer_template_verified": checkpoint["tokenizer_template_verified"] is True,
        "v2_plan_sha256": plan["v2_owner_plan_sha256"], "source_head": plan["source_head"],
        "v2_protocol_sha256": plan["source_sha256"][owner.PROTOCOL],
        "adapter_source_sha256": {p: plan["source_sha256"]["src/megartx/" + p] for p in ADAPTER_FILES},
        "deadline_monotonic": deadline, "parent_slot": auth["parent_slot"],
        "client_purpose": PURPOSE, "client_plan_sha256": plan["plan_sha256"],
        "client_mode": plan["client_mode"], "runner_lane": "v2",
        "target_probe_authorized": False, "drafter_authorized": False}


def check_client_admission(admission, project):
    """Worker/Core independently bind client configuration before acquisition."""
    if type(admission) is not dict:
        raise ProbeError("Exact V2 client admission required")
    plan = freeze(project, client_mode=admission.get("client_mode"))
    expected = {"client_purpose": PURPOSE, "client_plan_sha256": plan["plan_sha256"],
        "runner_lane": "v2", "source_head": plan["source_head"],
        "v2_plan_sha256": plan["v2_owner_plan_sha256"],
        "client_evidence_source_sha256": plan["source_sha256"]["src/megartx/speculative_native_evidence.py"],
        "target_probe_authorized": False, "drafter_authorized": False}
    if (plan["preflight_blockers"] or any(type(admission.get(k)) is not type(v)
            or admission.get(k) != v for k, v in expected.items())):
        raise ProbeError("V2 client admission is uncomposed or source/purpose differs")
    return plan
