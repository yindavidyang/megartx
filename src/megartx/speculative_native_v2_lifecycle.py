"""Default-off actual V2 receipt owner. No verifier or drafter entry is admitted.

The shared plugin/launcher must be composed and reviewed separately. This file
does not install itself, import a model runtime at module import, or serve work.
"""
import copy
import inspect
import math
import os
from pathlib import Path
import time

from . import speculative_native_lifecycle as common
from .speculative_native_probe import ProbeError, GPU_CAP, REVISION, digest, _process_start
from .speculative_native_v2 import (V2Owner, PROTOCOL, POLICY, RUNNER_MODULE, freeze,
                                    inspect_v2_sources, require_config, verify_method, class_source, validate_ticket,
                                    device_identity)

PURPOSE = "exclusive_native_v2_zero_forward_receipt"
WORKER_EXTENSION = "megartx.speculative_native_v2_lifecycle.NativeV2DiagnosticWorkerExtension"
UTILITIES = ("megartx_owned_native_v2_receipt", "megartx_owned_native_v2_probe", "megartx_owned_native_v2_release")


def check_receipt_admission(admission):
    project = Path(__file__).resolve().parents[2]
    from .speculative_native_v2_plan import check_client_admission
    check_client_admission(admission, project)
    plan = freeze(project)
    required = {"phase": "zero_forward_v2_receipt", "independent_review_clear": True,
        "parent_source_protocol_accepted": True, "owned_lifecycle_verified": True,
        "max_concurrent_gpu_jobs": 1, "max_extra_gpu_bytes": GPU_CAP,
        "compiler_memory_bytes": 2 << 30, "compiler_timeout_seconds": 300,
        "compiler_scope": "shared_host", "checkpoint_revision": REVISION,
        "checkpoint_manifest_verified": True, "tokenizer_template_verified": True,
        "v2_plan_sha256": plan["plan_sha256"], "source_head": plan["source_head"],
        "v2_protocol_sha256": digest(project / PROTOCOL),
        "adapter_source_sha256": {name: digest(Path(__file__).parent / name) for name in common.ADAPTER_FILES},
        "target_probe_authorized": False, "drafter_authorized": False}
    if type(admission) is not dict or any(type(admission.get(k)) is not type(v) or admission.get(k) != v
                                        for k, v in required.items()):
        raise ProbeError("Missing exact V2 receipt source/protocol/owned startup admission")
    deadline = admission.get("deadline_monotonic")
    if (type(deadline) not in (int, float) or not math.isfinite(deadline)
            or not time.monotonic() < deadline <= time.monotonic() + 1800
            or type(admission.get("parent_slot")) is not str or not admission["parent_slot"]):
        raise ProbeError("Missing bounded V2 receipt deadline or sole-owner parent slot")


def _require_core(core, utility):
    root = Path(inspect.getsourcefile(type(core))).resolve().parents[3]
    verify_method(core.vllm_config, "use_v2_model_runner", root)
    require_config(core.vllm_config)
    state = common._require_core(core, utility, worker_extension=WORKER_EXTENSION, use_v2=True)
    groups = core.scheduler.kv_cache_config.kv_cache_groups
    if not 1 <= len(groups) <= 30:
        raise ProbeError("V2 reservation group acquisition limit exceeded")
    return state


def _check_result(receipt):
    from .speculative_native_plan import canonical
    if (receipt.get("schema") != "megartx-native-v2-zero-forward-receipt-v1"
            or receipt.get("purpose") != PURPOSE or receipt.get("decision", {}).get("admitted") is not False
            or receipt.get("v2_owner", {}).get("mutable_verifier_lease_granted") is not False
            or receipt.get("v2_owner", {}).get("drafter_loaded") is not False
            or receipt.get("v2_owner", {}).get("metadata_built") is not False
            or receipt.get("v2_owner", {}).get("input_batch_prepared") is not False
            or canonical(receipt.get("v2_owner", {}).get("runner_policy")) != canonical(POLICY)):
        raise ProbeError("Unbound V2 receipt or unexpected later-stage authority")


def sanitized_receipt(receipt):
    from .speculative_native_receipt import sanitized_receipt as scalar_summary
    from .speculative_native_v2 import POLICY
    return {**scalar_summary(receipt), "runner_policy": dict(POLICY),
            "mutable_verifier_lease_granted": False, "drafter_loaded": False,
            "metadata_built": False, "input_batch_prepared": False}


def owned_receipt(core, admission):
    return common._owned_receipt(core, admission, admission_check=check_receipt_admission,
        core_check=_require_core, utility=UTILITIES[0], purpose=PURPOSE, result_check=_check_result, result_summary=sanitized_receipt)


def owned_probe(core, *args, **kwargs):
    raise ProbeError("V2 verifier/drafter interface requires separate source and budget review")


def owned_release(core):
    _require_core(core, UTILITIES[2])
    return common._release(core)


def install_native_v2_diagnostic():
    flag = os.environ.get("MEGARTX_NATIVE_V2_DIAGNOSTIC")
    if flag is None:
        return False
    if (flag != "1" or os.environ.get("MEGARTX_NATIVE_DIAGNOSTIC") is not None
            or os.environ.get("MEGARTX_SCALE_MODE") != "native"
            or os.environ.get("VLLM_PLUGINS") != "megartx_scale_adapter"
            or os.environ.get("MEGARTX_M1_PREPARATION")
            or os.environ.get("VLLM_USE_V2_MODEL_RUNNER") is not None):
        raise ProbeError("Explicit default-off V2 registration configuration differs")
    from vllm.v1.engine.core import EngineCore, EngineCoreProc
    root = Path(inspect.getsourcefile(EngineCore)).resolve().parents[3]
    inspect_v2_sources(root)
    for cls in (EngineCore, EngineCoreProc):
        if (cls.__module__ != "vllm.v1.engine.core" or cls.__name__ not in ("EngineCore", "EngineCoreProc")
                or any(hasattr(cls, name) for name in (*common.UTILITIES, *UTILITIES))):
            raise ProbeError("Duplicate/conflicting or wrong V2 registration owner")
    common._verify_loaded_methods(EngineCore, EngineCoreProc, root / "vllm/v1/engine/core.py")
    _process_start(os.getpid())
    active = common._active_core_birth(EngineCore)
    common._register_classes(EngineCore, EngineCoreProc, active, utilities=UTILITIES,
        receipt_callback=owned_receipt, probe_callback=owned_probe, release_callback=owned_release)
    return True


class NativeV2DiagnosticWorkerExtension:
    def _megartx_v2_check_worker(self):
        runner = self._megartx_v2_runner
        config, device = self._megartx_v2_config, self._megartx_v2_device
        if (self.model_runner is not runner or self.vllm_config is not config or runner.vllm_config is not config
                or self.device is not device or runner.device is not device
                or device_identity(device) != self._megartx_v2_device_identity):
            raise ProbeError("V2 runner is not bound to the original worker config/device")
        root = Path(inspect.getsourcefile(type(runner))).resolve().parents[4]
        class_source(runner, RUNNER_MODULE, "GPUModelRunner", root)
        class_source(self, "vllm.v1.worker.gpu_worker", "Worker", root)
        verify_method(runner, "get_model", root)

    def megartx_native_receipt(self, ticket, admission):
        check_receipt_admission(admission)
        common._class_source(self, "vllm.v1.worker.gpu_worker", "Worker")
        validate_ticket(ticket)
        common._worker_ticket(ticket, purpose=PURPOSE)
        if ticket.get("purpose") != PURPOSE or getattr(self, "_megartx_v2_receipt_used", False):
            raise ProbeError("V2 worker capability is wrong-purpose or already consumed")
        # Retain runner/ticket before anything that can fail or touch the device.
        self._megartx_v2_receipt_used = True
        self._megartx_v2_ticket = copy.deepcopy(ticket)
        self._megartx_v2_runner = self.model_runner
        self._megartx_v2_config = self.vllm_config
        self._megartx_v2_device = self.device
        self._megartx_v2_released = False
        self._megartx_v2_poisoned = False
        self._megartx_v2_device_identity = device_identity(self._megartx_v2_device)
        self._megartx_v2_check_worker()
        self._megartx_v2_owner = V2Owner(self.model_runner, ticket)
        class_source(self, "vllm.v1.worker.gpu_worker", "Worker", self._megartx_v2_owner.root)
        import torch
        from .speculative_native_v2_receipt import collect_receipt
        with torch.inference_mode():
            receipt = collect_receipt(self._megartx_v2_owner, ticket, admission)
        self._megartx_v2_check_worker()
        _check_result(receipt)
        return receipt

    def megartx_native_probe(self, *args, **kwargs):
        raise ProbeError("V2 receipt grants no verifier or drafter mutation capability")

    def megartx_native_release(self, ticket):
        common._class_source(self, "vllm.v1.worker.gpu_worker", "Worker")
        validate_ticket(ticket)
        common._worker_ticket(ticket, purpose=PURPOSE)
        if (ticket != getattr(self, "_megartx_v2_ticket", None)
                or getattr(self, "_megartx_v2_released", True)
                or getattr(self, "_megartx_v2_poisoned", True)
                or self.model_runner is not getattr(self, "_megartx_v2_runner", None)):
            raise ProbeError("Unknown, changed, poisoned or revoked V2 worker lease")
        # Failure after entry is single-use too. EngineCore retains real refs.
        self._megartx_v2_poisoned = True
        self._megartx_v2_check_worker()
        owner = getattr(self, "_megartx_v2_owner", None)
        if owner is not None:
            owner.check()
        import torch
        torch.cuda.synchronize(self._megartx_v2_device)
        self._megartx_v2_check_worker()
        if owner is not None:
            owner.check()
        self._megartx_v2_released = True
        return {"drained": True, "revoked": True}
