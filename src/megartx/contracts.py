"""Small, strict validator for the JSON Schema subset used by this package.

JSON schemas exported from SCHEMAS are interoperable Draft 2020-12 documents.
This validator is intentionally not a general-purpose JSON Schema engine.
"""

import json
import math
import re
from pathlib import Path

from .schema import DTYPE_BITS, SCHEMAS


class ContractError(ValueError):
    """A structural or cross-field experiment contract violation."""


def storage_bytes(shape, dtype):
    """Physical byte count, including a partial final packed byte; [] is a scalar."""
    if dtype not in DTYPE_BITS:
        raise ContractError(f"unsupported storage dtype {dtype!r}")
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in shape):
        raise ContractError("storage dimensions must be positive integers")
    return (math.prod(shape) * DTYPE_BITS[dtype] + 7) // 8


def tensor_storage_totals(tensors):
    """Check whole-storage alias consistency and explicit physical byte totals."""
    storages = {}
    without_dedup = 0
    for tensor in tensors:
        expected = storage_bytes(tensor["storage_shape"], tensor["storage_dtype"])
        if tensor["storage_bytes"] != expected:
            raise ContractError(f"{tensor['name']}: storage bytes disagree with physical shape/dtype")
        signature = (expected, tensor["storage_dtype"], tuple(tensor["storage_shape"]))
        key = tensor["storage_id"]
        if key in storages and storages[key] != signature:
            raise ContractError(f"{key}: aliased storage metadata disagree")
        storages[key] = signature
        without_dedup += expected
    unique = sum(value[0] for value in storages.values())
    return {"tensor_count": len(tensors), "unique_storage_count": len(storages),
            "unique_storage_bytes": unique, "alias_bytes_excluded": without_dedup - unique}


def validate_kv_layers(layers):
    _check(layers, SCHEMAS["memory"]["properties"]["kv_layers"])
    ids = [layer["layer_id"] for layer in layers]
    if len(ids) != len(set(ids)):
        raise ContractError("duplicate KV layer ID")
    for layer in layers:
        if layer["storage_policy"] == "bounded_window" and layer["window_tokens"] is None:
            raise ContractError("bounded_window requires an explicit window")


def load_json(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ContractError(f"non-finite JSON constant: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def _check(value, schema, path="$", root=None):
    root = schema if root is None else root
    if "$ref" in schema:
        target = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        return _check(value, target, path, root)
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                _check(value, option, path, root)
                return
            except ContractError:
                pass
        raise ContractError(f"{path}: no allowed variant matches")
    expected = schema.get("type")
    matches = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if expected and not matches[expected]:
        raise ContractError(f"{path}: expected {expected}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{path}: must be finite")
    if "const" in schema and value != schema["const"]:
        raise ContractError(f"{path}: expected {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ContractError(f"{path}: unsupported value {value!r}")
    for limit in ("minimum", "maximum"):
        if limit in schema and ((value < schema[limit]) if limit == "minimum" else
                                (value > schema[limit])):
            raise ContractError(f"{path}: violates {limit} {schema[limit]}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ContractError(f"{path}: string too short")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ContractError(f"{path}: invalid format")
    if isinstance(value, dict):
        missing = set(schema.get("required", [])) - value.keys()
        if missing:
            raise ContractError(f"{path}: missing fields {sorted(missing)}")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = value.keys() - props.keys()
            if extra:
                raise ContractError(f"{path}: unknown fields {sorted(extra)}")
        for key, item in value.items():
            if key in props:
                _check(item, props[key], f"{path}.{key}", root)
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ContractError(f"{path}: too few entries")
        if schema.get("uniqueItems") and len({json.dumps(x, sort_keys=True) for x in value}) != len(value):
            raise ContractError(f"{path}: duplicate entries")
        for index, item in enumerate(value):
            _check(item, schema.get("items", {}), f"{path}[{index}]", root)


def validate(document, kind=None):
    if not isinstance(document, dict):
        raise ContractError("contract root must be a JSON object")
    kind = kind or document.get("kind")
    if not isinstance(kind, str) or kind not in SCHEMAS:
        raise ContractError(f"unknown contract kind: {kind!r}")
    _check(document, SCHEMAS[kind])
    if kind == "experiment":
        workload = document["workload"]
        if workload["primary_context_tokens"] not in workload["context_probes_tokens"]:
            raise ContractError("primary context must be an enabled context probe")
        if document["status"] == "frozen":
            if any(value is None for value in document["checkpoint"].values()):
                raise ContractError("frozen experiment needs immutable checkpoint/tokenizer/template hashes")
            if document["acceptance"]["status"] != "frozen":
                raise ContractError("frozen experiment needs frozen acceptance margins")
            if not document["owner_approval_reference"]:
                raise ContractError("frozen experiment needs an owner approval reference")
            if workload["prompt_set_sha256"] is None:
                raise ContractError("frozen experiment needs a hashed prompt corpus")
            if "pending_owner_freeze" in (workload["sampling_policy"], workload["eos_policy"]):
                raise ContractError("frozen experiment needs explicit sampling and EOS policies")
    elif kind == "compatibility":
        keys = [item["feature"] for item in document["paths"]]
        if len(set(keys)) != len(keys):
            raise ContractError("duplicate compatibility feature")
        for item in document["paths"]:
            if item["status"] != "unverified" and not item["evidence_reference"]:
                raise ContractError("supported/unsupported paths need evidence")
    elif kind == "tensors":
        names = [item["name"] for item in document["tensors"]]
        if len(names) != len(set(names)):
            raise ContractError("duplicate tensor name")
        if document["complete"] and (not names or not document["checkpoint_revision"]):
            raise ContractError("complete tensor inventory needs tensors and immutable revision")
        tensor_storage_totals(document["tensors"])
    elif kind == "memory":
        validate_kv_layers(document["kv_layers"])
        if document["complete"] and (not document["kv_layers"] or document["device_total_bytes"] is None
                                     or any(v is None for v in document["other_bytes"].values())):
            raise ContractError("complete memory budget needs KV geometry, device capacity and all byte components")
    elif kind == "environment" and document["status"] == "pinned":
        optional = {"b12x_revision"}
        if any(value is None for key, value in document.items() if key not in optional):
            raise ContractError("pinned environment needs all core versions, revisions, lock hash and dirty-source state")
    elif kind == "fixtures":
        ids = [item["id"] for item in document["cases"]]
        if len(ids) != len(set(ids)):
            raise ContractError("duplicate fixture ID")
        for case in document["cases"]:
            if case["status"] == "captured":
                required = ("input_sha256", "expected_sha256", "oracle_revision", "tolerances")
                if any(case[field] is None for field in required):
                    raise ContractError("captured fixtures need hashed inputs/outputs and frozen oracle/tolerances")
                if not case["tolerance_approval_reference"]:
                    raise ContractError("captured fixture tolerances need approval rationale")
    elif kind == "results":
        ids = [item["pair_id"] for item in document["pairs"]]
        if len(ids) != len(set(ids)):
            raise ContractError("duplicate request pair ID")
        if document["provenance"] == "measured":
            required = ("checkpoint_revision", "environment_sha256", "experiment_sha256",
                        "raw_events_sha256", "code_revision")
            if any(document[field] is None for field in required):
                raise ContractError("measured results need pinned provenance and raw events")
            if not document["command"] or not document["pairs"]:
                raise ContractError("measured results need a command and raw request pairs")
    return document


def readiness(experiment):
    """Freeze readiness is separate from structural validity and all GPU gates."""
    validate(experiment, "experiment")
    blockers = []
    if experiment["status"] != "frozen":
        blockers.append("experiment choices remain proposed")
    for key, value in experiment["checkpoint"].items():
        if value is None:
            blockers.append(f"checkpoint.{key} is not pinned")
    if experiment["acceptance"]["status"] != "frozen":
        blockers.append("acceptance thresholds await owner approval")
    if not experiment["owner_approval_reference"]:
        blockers.append("owner freeze approval reference is missing")
    if experiment["workload"]["prompt_set_sha256"] is None:
        blockers.append("prompt corpus is not hashed")
    if "pending_owner_freeze" in (experiment["workload"]["sampling_policy"], experiment["workload"]["eos_policy"]):
        blockers.append("sampling and EOS policies await owner freeze")
    return {"freeze_ready": not blockers, "blockers": blockers,
            "G0": "pending_gpu_evidence", "G1": "pending_oracles_and_tuned_baseline"}
