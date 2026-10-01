"""Bounded, CPU-only intake/checking of M1 preparation probe artifacts.

Never compiles, loads a library, launches an operator, or selects a candidate.
Host layout evidence and hash-consistent captures do not prove installed dispatch.
"""

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct

import m1_preparation_reference as oracle


SCHEMA = "megartx.m1-preparation-capture.v1"
GEOMETRY = {"m": 1, "h": 2816, "e": 128, "top_k": 8, "f": 704}
MAX_JSON, MAX_FILE, MAX_BUNDLE = 1 << 20, 4 << 20, 32 << 20
SF_BYTES = 2883584
FILE_SIZES = {
    "ids": 32, "weights": 32, "aq": 1408, "sf": 22528,
    "input_after": 24000,  # Concatenated immutable ids/weights/AQ/SF after preparation.
    "slot_to_sorted": 32, "sorted_to_slot": 32, "expert_offsets": 1032,
    "expanded_aq": 11264, "permuted_weight_bits": 32,
    "fc1_shapes": 3072, "fc2_shapes": 3072,
    "sf_before": SF_BYTES + 64, "sf_after": SF_BYTES + 64,
}
REFERENCE_FILES = (
    ("flashinfer", "csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh",
     "fd9e2e976496ab318bda6d133d2b68f45b3451a6482978f452acd7a86e029841"),
    ("flashinfer", "csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h",
     "eca60a5a7f30b70b7a4f426fb833085ee2ae320445fb217fdd8642c9480b23ab"),
    ("flashinfer", "csrc/nv_internal/tensorrt_llm/kernels/quantization_utils.cuh",
     "c2a860da4407f70c281c84981db796ec111520d141a3753d77fe66c6c11f3928"),
    ("cutlass", "include/cutlass/detail/sm100_blockscaled_layout.hpp",
     "598e054bef21edf94b1fd6bb1447cfa9cfcf5a5907ab370128102448dbb6d530"),
)
BINDINGS = (
    "compiled_probe_sha256", "dependency_manifest_sha256", "generated_sources_sha256",
    "loaded_module_sha256", "compiler_flags_sha256", "tactics_sha256",
    "correction_runner_sha256", "workspace_audit_sha256", "consumer_masks_sha256",
    "launch_correlation_sha256", "capture_encoder_sha256",
)
HOST_FIELDS = (
    "swap_ab", "shape_info", "stride_act", "stride_weight", "ptr_act", "ptr_weight",
    "stride_c", "ptr_c", "stride_d", "ptr_d", "fusion", "alpha_scale_ptr_array",
    "fpX_block_scaling_factors_act", "fpX_block_scaling_factors_weight",
    "fpX_block_scaling_factors_stride_act", "fpX_block_scaling_factors_stride_weight",
    "fpX_block_scaling_type", "int4_groupwise_params", "gemm_workspace", "gemm_workspace_size",
    "precomputed_scheduler_workspace", "precomputed_scheduler_workspace_size",
    "precomputed_scheduler_total_routed_tokens", "enable_pdl",
) + tuple("fused_finalize_epilogue." + name for name in (
    "ptr_final_output", "stride_final_output_transposed", "stride_final_output", "ptr_bias",
    "ptr_router_scales", "ptr_source_token_index", "num_rows_in_final_output",
    "shape_override", "use_reduction"))


def exact(obj, keys, label):
    if type(obj) is not dict or set(obj) != set(keys):
        raise ValueError(label + " has missing/unknown fields")


def integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(label + " exceeds its integer bounds")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha(value):
    if type(value) is not str or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("invalid SHA256")
    return value


def unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def decode_json(data):
    if len(data) > MAX_JSON:
        raise ValueError("JSON exceeds 1 MiB cap")
    def reject_number(_):
        raise ValueError("JSON floating-point numbers unsupported; preserve scalar bits")
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=reject_number, parse_float=reject_number)


def read_file(root, relative, cap):
    """Read once; reject traversal, links and oversized files before parsing/copying."""
    root = Path(root).resolve(strict=True)
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(p in ("..", ".") for p in path.parts):
        raise ValueError("artifact path must be a contained relative file")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("artifact/source symlinks are unsupported")
    if not current.resolve(strict=True).is_relative_to(root):
        raise ValueError("artifact escapes its root")
    status = current.stat(follow_symlinks=False)
    if not stat.S_ISREG(status.st_mode) or status.st_size > cap:
        raise ValueError("artifact is not a bounded regular file")
    descriptor = os.open(current, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        status = os.fstat(stream.fileno())
        if not stat.S_ISREG(status.st_mode) or status.st_size > cap:
            raise ValueError("artifact is not a bounded regular file")
        data = stream.read(cap + 1)
    if len(data) > cap:
        raise ValueError("artifact grew beyond cap")
    return data


def wire(value):
    """Semantic JSON encoding; never serialize opaque/native object padding."""
    if type(value) is bytes:
        return value.hex()
    if is_dataclass(value):
        return {f.name: wire(getattr(value, f.name)) for f in fields(value)}
    if type(value) in (tuple, list):
        return [wire(x) for x in value]
    return value


def same(left, right):
    """JSON equality with exact types; bool must never compare equal to int."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(same(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def source_audit(roots):
    records = []
    for package, path, expected in REFERENCE_FILES:
        data = read_file(roots[package], path, MAX_FILE)
        records.append({"package": package, "path": path, "sha256": digest(data),
                        "bytes": len(data), "matches_reference": digest(data) == expected})
    return {"schema": "megartx.m1-source-audit.v1", "files": records,
            "reference_source_match": all(x["matches_reference"] for x in records),
            "installed_abi_verified": False}


def check_source_audit(audit):
    exact(audit, ("schema", "files", "reference_source_match", "installed_abi_verified"), "source audit")
    if audit["schema"] != "megartx.m1-source-audit.v1" or audit["installed_abi_verified"] is not False:
        raise ValueError("source audit cannot certify installed ABI")
    if type(audit["files"]) is not list or len(audit["files"]) != len(REFERENCE_FILES):
        raise ValueError("source audit must name all four reference inputs")
    matches = []
    for record, (package, path, expected) in zip(audit["files"], REFERENCE_FILES):
        exact(record, ("package", "path", "sha256", "bytes", "matches_reference"), "source record")
        if (record["package"], record["path"]) != (package, path):
            raise ValueError("source record identity/order changed")
        integer(record["bytes"], 1, MAX_FILE, "source bytes")
        match = sha(record["sha256"]) == expected
        if record["matches_reference"] is not match:
            raise ValueError("source-match label disagrees with hashes")
        matches.append(match)
    result = all(matches)
    if audit["reference_source_match"] is not result:
        raise ValueError("aggregate source-match label disagrees with hashes")
    return result


def check_host_probe(probe):
    exact(probe, ("scope", "little_endian", "types", "fields", "scale_layouts",
                  "workspace_layout", "consumer_masks"), "host probe")
    if probe["scope"] != "typed_host_m1_layout_samples" or probe["little_endian"] is not True:
        raise ValueError("unsupported host probe scope/byte order")
    if probe["workspace_layout"] is not None or probe["consumer_masks"] is not None:
        raise ValueError("host-only probe cannot invent runtime workspace/consumer masks")
    exact(probe["types"], ("descriptor", "problem", "stride_a", "stride_b", "sf_layout", "element_sf"), "types")
    for value in probe["types"].values():
        exact(value, ("size", "align"), "typed size/alignment")
        integer(value["size"], 1, 65536, "typed size")
        integer(value["align"], 1, 4096, "typed alignment")
        if value["align"] & (value["align"] - 1) or value["size"] % value["align"]:
            raise ValueError("invalid typed size/alignment")
    if probe["types"]["element_sf"]["size"] != 1:
        raise ValueError("unsupported scale element size")
    exact(probe["fields"], HOST_FIELDS, "typed fields")
    for value in probe["fields"].values():
        exact(value, ("offset", "size", "align"), "typed field")
        integer(value["offset"], 0, 65536, "field offset")
        integer(value["size"], 1, 65536, "field size")
        integer(value["align"], 1, 4096, "field alignment")
        if (value["align"] & (value["align"] - 1)
                or value["offset"] + value["size"] > probe["types"]["descriptor"]["size"]):
            raise ValueError("typed field exceeds descriptor")
    if type(probe["scale_layouts"]) is not list or len(probe["scale_layouts"]) != 4:
        raise ValueError("four independent stage/swap layout samples required")
    for sample, (stage, n, k, swap) in zip(probe["scale_layouts"], (
            ("fc1", 1408, 2816, False), ("fc1", 1408, 2816, True),
            ("fc2", 2816, 704, False), ("fc2", 2816, 704, True))):
        exact(sample, ("stage", "n", "k", "swap_ab", "act_offsets",
                       "weight_tile_offsets", "weight_last_offsets"), "SF layout sample")
        if not same([sample[key] for key in ("stage", "n", "k", "swap_ab")], [stage, n, k, swap]):
            raise ValueError("unsupported or reordered layout geometry")
        blocks = k // 16
        expectations = {
            "act_offsets": [oracle.sf_coordinate(0, b, blocks) for b in range(blocks)],
            "weight_tile_offsets": [oracle.sf_coordinate(r, b, blocks)
                                    for r in range(128) for b in range(blocks)],
            "weight_last_offsets": [oracle.sf_coordinate(n - 1, b, blocks) for b in range(blocks)],
        }
        for name, expected in expectations.items():
            actual = sample[name]
            if type(actual) is not list or len(actual) != len(expected):
                raise ValueError("SF coordinate extent mismatch")
            if any(type(x) is not int for x in actual) or actual != expected:
                raise ValueError("typed SF coordinates disagree with independent oracle")


def context(value):
    exact(value, (f.name for f in fields(oracle.StageContext)), "stage context")
    def raw(text, size):
        if type(text) is not str or not re.fullmatch("[0-9a-f]{" + str(size * 2) + "}", text):
            raise ValueError("stage scalar bits require exact lowercase hex")
        return bytes.fromhex(text)
    if type(value["weight_global_bits"]) is not list or len(value["weight_global_bits"]) not in (1, 2):
        raise ValueError("original global tables missing")
    return oracle.StageContext(value["stage"], value["swap_ab"], value["fusion"],
        raw(value["alpha_bits"], 512), raw(value["activation_global_bits"], 4),
        tuple(raw(x, 512) for x in value["weight_global_bits"]), value["scalar_provenance"])


def role_views(prepared):
    roles = {"sorted_to_slot": 32, "permuted_weight_bits": 32}
    for stage in (prepared.fc1, prepared.fc2):
        for desc in stage.active_descriptors:
            views = [desc.activation, desc.weight, desc.activation_sf, desc.weight_sf, desc.output,
                     desc.alpha.ref, desc.activation_global.ref, desc.finalize_map, desc.finalize_weights]
            views += [binding.ref for binding in desc.weight_globals]
            for view in views:
                if view is not None:
                    roles[view.owner] = view.capacity
    return roles


def check_owners(owners, roles):
    exact(owners, roles, "canonical owner registry")
    for role, value in owners.items():
        exact(value, ("allocation", "offset", "capacity", "allocation_extent", "alignment", "lifetime"), "owner")
        if type(value["allocation"]) is not str or not re.fullmatch("alloc_[0-9]{1,6}", value["allocation"]):
            raise ValueError("anonymous allocation ID required; raw addresses unsupported")
        for name in ("offset", "capacity", "allocation_extent"):
            integer(value[name], 0, 1 << 40, "owner " + name)
        if value["capacity"] != roles[role] or value["offset"] + value["capacity"] > value["allocation_extent"]:
            raise ValueError("owner view extent mismatch/out of bounds")
        integer(value["alignment"], 1, 4096, "owner alignment")
        if value["alignment"] & (value["alignment"] - 1):
            raise ValueError("owner alignment must be a power of two")
        life = value["lifetime"]
        if type(life) is not list or len(life) != 2:
            raise ValueError("explicit inclusive owner lifetime required")
        for phase in life:
            integer(phase, 0, 6, "lifetime phase")
        if life[0] > life[1]:
            raise ValueError("reversed owner lifetime")
    records = list(owners.values())
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            if a["allocation"] != b["allocation"]:
                continue
            if (a["allocation_extent"], a["alignment"]) != (b["allocation_extent"], b["alignment"]):
                raise ValueError("conflicting allocation extent/alignment")
            overlap = max(a["offset"], b["offset"]) < min(a["offset"] + a["capacity"], b["offset"] + b["capacity"])
            live = max(a["lifetime"][0], b["lifetime"][0]) <= min(a["lifetime"][1], b["lifetime"][1])
            if overlap and live:
                raise ValueError("overlapping live aliases unsupported; do not guess scratch ownership")


def validate_bundle(root):
    """Check exact semantic artifacts, retaining unsupported native fields explicitly."""
    payloads = {"capture.json": read_file(root, "capture.json", MAX_JSON)}
    packet = decode_json(payloads["capture.json"])
    exact(packet, ("schema", "geometry", "origin", "source_audit", "host_probe", "bindings", "owners", "cases"), "capture")
    if packet["schema"] != SCHEMA or packet["origin"] not in ("synthetic", "target_incumbent"):
        raise ValueError("unsupported capture schema/origin")
    exact(packet["geometry"], GEOMETRY, "geometry")
    if any(type(packet["geometry"][k]) is not int for k in GEOMETRY) or packet["geometry"] != GEOMETRY:
        raise ValueError("only M1 geometry is supported; small-M needs a separate oracle/version")
    pending = []
    if packet["source_audit"] is None:
        pending.append("installed_source_audit_missing")
    elif not check_source_audit(packet["source_audit"]):
        pending.append("installed_sources_do_not_match_reference_profile")
    if packet["host_probe"] is None:
        pending.append("typed_host_probe_missing")
    else:
        record = packet["host_probe"]
        exact(record, ("path", "sha256", "bytes"), "host probe artifact")
        if record["path"] != "host-abi.json":
            raise ValueError("unexpected host probe filename")
        integer(record["bytes"], 1, MAX_JSON, "host probe bytes")
        data = read_file(root, record["path"], MAX_JSON)
        if len(data) != record["bytes"] or digest(data) != sha(record["sha256"]):
            raise ValueError("host probe digest/size mismatch")
        check_host_probe(decode_json(data))
        payloads[record["path"]] = data
    exact(packet["bindings"], BINDINGS, "binary/runner bindings")
    for name, value in packet["bindings"].items():
        if value is None:
            pending.append(name + "_missing")
        else:
            sha(value)  # A digest assertion is not proof of loaded-module/build equality.
    if type(packet["cases"]) is not list or not 1 <= len(packet["cases"]) <= 3:
        raise ValueError("capture must contain one to three bounded M1 cases")
    routes, all_roles, frozen_contexts, previous_sf = [], {}, None, None
    for index, case in enumerate(packet["cases"]):
        exact(case, ("fc1", "fc2", "descriptors", "files"), "case")
        contexts = [case["fc1"], case["fc2"]]
        if frozen_contexts is not None and not same(contexts, frozen_contexts):
            raise ValueError("same configured layer/stage contexts required across cases")
        frozen_contexts = contexts
        exact(case["files"], FILE_SIZES, "case artifacts")
        raw = {}
        for name, size in FILE_SIZES.items():
            record = case["files"][name]
            exact(record, ("path", "sha256", "bytes"), "artifact record")
            if record["path"] != f"case{index}/{name}.bin" or type(record["bytes"]) is not int or record["bytes"] != size:
                raise ValueError("artifact path/physical byte extent mismatch")
            data = read_file(root, record["path"], size)
            if len(data) != size or digest(data) != sha(record["sha256"]):
                raise ValueError("artifact digest/size mismatch")
            raw[name] = data
            payloads[record["path"]] = data
            if sum(map(len, payloads.values())) > MAX_BUNDLE:
                raise ValueError("capture exceeds 32 MiB byte budget")
        input_row = oracle.InputRow(struct.unpack("<8i", raw["ids"]), raw["weights"], raw["aq"], raw["sf"])
        if raw["input_after"] != raw["ids"] + raw["weights"] + raw["aq"] + raw["sf"]:
            raise ValueError("preparation mutated an input source")
        prepared = oracle.prepare(input_row, context(case["fc1"]), context(case["fc2"]), reference_abi=oracle.REFERENCE_ABI)
        for name in ("slot_to_sorted", "sorted_to_slot", "expert_offsets", "expanded_aq", "permuted_weight_bits"):
            if raw[name] != getattr(prepared, name):
                raise ValueError("byte-exact preparation mismatch: " + name)
        for stage in ("fc1", "fc2"):
            if raw[stage + "_shapes"] != getattr(prepared, stage).problem_shapes:
                raise ValueError("all-128 problem shape mismatch: " + stage)
        exact(case["descriptors"], ("fc1", "fc2"), "descriptors")
        for stage in ("fc1", "fc2"):
            if not same(case["descriptors"][stage], wire(getattr(prepared, stage).active_descriptors)):
                raise ValueError("active semantic descriptor mismatch: " + stage)
        oracle.require_sf_storage(prepared, raw["sf_before"], raw["sf_after"], origin=32)
        if previous_sf is not None and raw["sf_before"] != previous_sf:
            raise ValueError("reused scale scratch must continue from the previous capture")
        previous_sf = raw["sf_after"]
        routes.append(input_row.selected_ids)
        all_roles.update(role_views(prepared))
    if packet["owners"] is None:
        pending.append("workspace_owner_registry_missing")
    else:
        check_owners(packet["owners"], all_roles)
    stress = len(routes) == 3 and routes[0] == routes[1] and set(routes[0]).isdisjoint(routes[2])
    if not stress:
        pending.append("repeat_and_disjoint_route_fixture_missing")
    pending.extend(("consumer_read_masks_and_opaque_workspace_unverified",
                    "installed_probe_build_vs_loaded_module_equivalence_unverified",
                    "gpu_execution_and_corrected_numerical_lane_unqualified"))
    return {"scope": "CPU artifact consistency only", "origin": packet["origin"],
            "cpu_semantic_match": True, "cases": len(routes), "route_stress_present": stress,
            "captured_payload_bytes": sum(map(len, payloads.values())), "unsupported": pending,
            "installed_abi_verified": False, "candidate_selectable": False}, payloads


def write_new(path, data):
    with open(path, "xb") as stream:
        stream.write(data)


def fixture_request():
    """Target handoff specification, with no fabricated capture or ABI values."""
    ids = [127, 0, 82, 42, 126, 7, 89, 12]
    return {"schema": "megartx.m1-preparation-request.v1", "capture_schema": SCHEMA,
            "geometry": dict(GEOMETRY), "scope": "standalone incumbent preparation fixture",
            "routing_intervention": True, "requested_ids": [ids, ids.copy(), list(range(20, 28))],
            "artifact_bytes_per_case": dict(FILE_SIZES), "sf_guard_origin": 32,
            "raw_bytes_per_case": sum(FILE_SIZES.values()), "max_cases": 3,
            "max_bundle_bytes": MAX_BUNDLE, "max_json_bytes": MAX_JSON,
            "max_source_file_bytes": MAX_FILE, "bindings": dict.fromkeys(BINDINGS),
            "source_audit": None, "host_probe": None, "owners": None,
            "candidate_selectable": False, "gpu_run_authorized_by_this_file": False}


def capture(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("capture destination must be outside the producer tree")
    report, payloads = validate_bundle(source)
    destination.mkdir(mode=0o700)  # Exclusive; do not overwrite an existing capture.
    for name, data in payloads.items():
        target = destination / name
        target.parent.mkdir(mode=0o700, exist_ok=True)
        write_new(target, data)
    write_new(destination / "validation.json", (json.dumps(report, indent=2) + "\n").encode())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    request = commands.add_parser("request")
    request.add_argument("--output", required=True)
    audit = commands.add_parser("source-audit")
    audit.add_argument("--flashinfer-root", required=True)
    audit.add_argument("--cutlass-root", required=True)
    audit.add_argument("--output", required=True)
    collect = commands.add_parser("capture")
    collect.add_argument("--input", required=True)
    collect.add_argument("--output", required=True)
    check = commands.add_parser("validate")
    check.add_argument("directory")
    check.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "request":
            report, status = fixture_request(), 0
            write_new(args.output, (json.dumps(report, indent=2) + "\n").encode())
        elif args.command == "source-audit":
            roots = {"flashinfer": args.flashinfer_root, "cutlass": args.cutlass_root}
            output = Path(args.output).resolve()
            if any(output.is_relative_to(Path(root).resolve()) for root in roots.values()):
                raise ValueError("source-audit output must be outside source trees")
            report = source_audit(roots)
            write_new(output, (json.dumps(report, indent=2) + "\n").encode())
            status = 0 if report["reference_source_match"] else 2
        elif args.command == "capture":
            report, status = capture(args.input, args.output), 0
        else:
            report, _ = validate_bundle(args.directory)
            status = 0 if args.cpu_only else 2
        print(json.dumps(report, indent=2))
        return status
    except (ValueError, OSError, TypeError, RecursionError, oracle.PreparationMismatch) as error:
        # Avoid dumping private paths or operand bytes from OS/parser exceptions.
        print(json.dumps({"status": "rejected", "error_type": type(error).__name__,
                          "candidate_selectable": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
