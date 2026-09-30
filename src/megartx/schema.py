"""Canonical schema definitions; export with `python -m megartx schemas`.

Structural validation does not establish truth of evidence or a passed GPU gate.
Cross-field constraints are enforced separately by contracts.validate.
"""


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def arr(items, minimum=0, unique=False):
    return {"type": "array", "items": items, "minItems": minimum, "uniqueItems": unique}


def enum(*values):
    return {"type": "string", "enum": list(values)}


def integer(minimum=0):
    return {"type": "integer", "minimum": minimum}


def number(minimum=0):
    return {"type": "number", "minimum": minimum}


def nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


TEXT = {"type": "string", "minLength": 1}
BOOL = {"type": "boolean"}
SHA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
REV = {"type": "string", "pattern": "^[0-9a-f]{40}$"}
LANE = enum("nvfp4_w4a4", "w4a16", "bf16")
PENDING = enum("pending", "pass", "fail")
DTYPE_BITS = {"bool": 8, "uint8": 8, "int8": 8, "float8_e4m3fn": 8, "float8_e5m2": 8,
              "uint16": 16, "int16": 16, "float16": 16, "bfloat16": 16,
              "uint32": 32, "int32": 32, "float32": 32, "uint64": 64,
              "int64": 64, "float64": 64, "packed_fp4": 4}
DTYPE = enum(*DTYPE_BITS)


def document(kind, properties):
    schema = obj({"schema_version": {"type": "integer", "const": 1},
                  "kind": {"type": "string", "const": kind}, **properties})
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = f"megartx {kind} v1"
    return schema


EXPERIMENT = document("experiment", {
    "status": enum("proposed", "frozen"),
    "owner_approval_reference": nullable(TEXT),
    "hardware": obj({"gpu_name": {"type": "string", "const": "RTX 5090"},
                     "gpu_count": {"type": "integer", "const": 1},
                     "sm_target": {"type": "integer", "const": 120}}),
    "checkpoint": obj({"repository": TEXT, "revision": nullable(REV),
                       "tokenizer_sha256": nullable(SHA), "chat_template_sha256": nullable(SHA)}),
    "numerical_lane": LANE,
    "workload": obj({"text_only": {"type": "boolean", "const": True},
                     "concurrency": {"type": "integer", "const": 1},
                     "dynamic_batching": {"type": "boolean", "const": False},
                     "context_probes_tokens": arr(integer(1), 1, True),
                     "conditional_context_tokens": arr(integer(1), 0, True),
                     "primary_context_tokens": integer(1), "output_capacity_tokens": integer(1),
                     "kv_dtype": enum("bfloat16"), "sampling_policy": enum("greedy", "pending_owner_freeze"),
                     "eos_policy": enum("fixed_length_ignore_eos", "natural_eos_separate_cell", "pending_owner_freeze"),
                     "prompt_set_sha256": nullable(SHA)}),
    "acceptance": obj({"status": enum("proposed", "frozen"),
                       "min_paired_requests": integer(2),
                       "median_itl_reduction_fraction": {"type": "number", "minimum": 0, "maximum": 1},
                       "stretch_reduction_fraction": {"type": "number", "minimum": 0, "maximum": 1},
                       "max_p95_regression_fraction": number(),
                       "max_ttft_regression_fraction": number(),
                       "max_total_response_regression_fraction": number(),
                       "max_relative_perplexity_increase": number(),
                       "max_task_score_loss_percentage_points": number(),
                       "memory_reserve_bytes": integer(1)})})

COMPATIBILITY = document("compatibility", {
    "checkpoint_revision": nullable(REV), "environment_sha256": nullable(SHA),
    "paths": arr(obj({"feature": TEXT, "status": enum("unverified", "supported", "unsupported"),
                      "source_reference": nullable(TEXT), "evidence_reference": nullable(TEXT),
                      "notes": TEXT}), 1)})

ENVIRONMENT = document("environment", {
    "status": enum("unverified", "pinned"),
    "target_sm": {"type": "integer", "const": 120},
    "driver_version": nullable(TEXT), "cuda_toolkit_version": nullable(TEXT),
    "cuda_runtime_version": nullable(TEXT), "compiler_version": nullable(TEXT),
    "python_version": nullable(TEXT), "pytorch_version": nullable(TEXT),
    "host_runtime_name": nullable(TEXT), "host_runtime_revision": nullable(REV),
    "flashinfer_revision": nullable(REV), "cutlass_revision": nullable(REV),
    "b12x_revision": nullable(REV), "build_flags": arr(TEXT),
    "package_lock_sha256": nullable(SHA), "dirty_source": nullable(BOOL)})

TENSORS = document("tensors", {
    "complete": BOOL, "checkpoint_revision": nullable(REV),
    "tensors": arr(obj({"name": TEXT, "logical_shape": arr(integer(1)),
                        "storage_shape": arr(integer(1)), "storage_dtype": DTYPE,
                        "storage_id": TEXT, "storage_bytes": integer(),
                        "quantization_role": enum("payload", "scale", "unquantized", "other"),
                        "ownership": enum("text", "multimodal", "shared", "unknown")}))})

MEMORY = document("memory", {
    "complete": BOOL, "device_total_bytes": nullable(integer(1)),
    "reserve_bytes": integer(), "cached_context_tokens": integer(1),
    "output_capacity_tokens": integer(1),
    "kv_layers": arr(obj({"layer_id": TEXT, "kv_heads": integer(1),
                          "key_head_dim": integer(1), "value_head_dim": integer(1),
                          "key_dtype": enum("bfloat16", "float16", "float32"),
                          "value_dtype": enum("bfloat16", "float16", "float32"),
                          "storage_policy": enum("full_context", "bounded_window"),
                          "window_tokens": nullable(integer(1))})),
    "other_bytes": obj({key: nullable(integer()) for key in (
        "retained_weight_copies", "runtime_repacks", "persistent_buffers", "graph_pools",
        "scratch_peak", "prefill_peak_increment", "runtime_and_display")})})

TOLERANCES = obj({"absolute_error": number(), "relative_error": number(),
                  "normalized_rmse": number(), "min_cosine_similarity": {
                      "type": "number", "minimum": -1, "maximum": 1},
                  "max_routing_mismatches": integer()})
FIXTURES = document("fixtures", {
    "checkpoint_revision": nullable(REV),
    "cases": arr(obj({"id": TEXT, "numerical_lane": LANE,
                      "status": enum("pending", "captured"), "cases_required": arr(TEXT, 1),
                      "intermediates": arr(TEXT, 1), "input_sha256": nullable(SHA),
                      "expected_sha256": nullable(SHA), "oracle_revision": nullable(REV),
                      "tolerances": nullable(TOLERANCES),
                      "tolerance_approval_reference": nullable(TEXT)}), 1)})

DISPATCH = document("dispatch", {
    "classification": enum("unverified", "unsupported", "fallback", "verified_flashinfer"),
    "checkpoint_revision": nullable(REV), "environment_sha256": nullable(SHA),
    "numerical_lane": LANE, "host_runtime": nullable(TEXT), "host_revision": nullable(REV),
    "flashinfer_revision": nullable(REV),
    "observations": arr(obj({"component": enum("attention", "moe"), "provider": TEXT,
                             "kernel_names": arr(TEXT, 1), "native_nvfp4": BOOL})),
    "non_flashinfer_components": arr(TEXT),
    "artifact_relative_path": nullable(TEXT), "artifact_sha256": nullable(SHA),
    "reviewer_reference": nullable(TEXT)})

PAIR = obj({"pair_id": TEXT, "order": enum("baseline_first", "candidate_first"),
            "prompt_sha256": SHA, "seed": integer(),
            "baseline_itl_ms": arr(number(0.000000001), 1),
            "candidate_itl_ms": arr(number(0.000000001), 1),
            "baseline_ttft_ms": number(0.000000001),
            "candidate_ttft_ms": number(0.000000001),
            "baseline_total_response_ms": number(0.000000001),
            "candidate_total_response_ms": number(0.000000001)})
RESULTS = document("results", {
    "run_id": TEXT, "provenance": enum("template", "synthetic", "measured"),
    "work_package": enum("WP0", "WP1"), "numerical_lane": LANE,
    "boundary": enum("delivered_token_end_to_end", "gpu_resident", "microbenchmark"),
    "checkpoint_revision": nullable(REV), "environment_sha256": nullable(SHA),
    "experiment_sha256": nullable(SHA), "code_revision": nullable(REV),
    "raw_events_sha256": nullable(SHA), "command": arr(TEXT),
    "correctness": PENDING, "quality": PENDING, "memory": PENDING,
    "baseline_dispatch": enum("unverified", "unsupported", "fallback", "verified_flashinfer"),
    "omitted_critical_path_costs": arr(TEXT), "pairs": arr(PAIR)})

SCHEMAS = {schema["properties"]["kind"]["const"]: schema for schema in (
    EXPERIMENT, COMPATIBILITY, ENVIRONMENT, TENSORS, MEMORY, FIXTURES, DISPATCH, RESULTS)}
