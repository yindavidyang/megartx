"""Independent exact-byte checker for narrowly scoped installed preparation captures."""
import argparse
import json
from pathlib import Path
import struct

import m1_abi_probe
import m1_kernel_fixture as fixture
import m1_preparation_reference as oracle


def metadata(root, name):
    return m1_abi_probe.decode_json(m1_abi_probe.read_file(root, name, 1 << 20))


def check_quantparams(q):
    m1_abi_probe.exact(q, ("scope", "type", "fields"), "QuantParams report")
    if q["scope"] != "installed_typed_quantparams":
        raise ValueError("QuantParams scope")
    m1_abi_probe.exact(q["type"], ("size", "align"), "QuantParams type")
    size, align = q["type"]["size"], q["type"]["align"]
    m1_abi_probe.integer(size, 1, 65536, "QuantParams size")
    m1_abi_probe.integer(align, 1, 4096, "QuantParams alignment")
    if align & (align-1) or size % align:
        raise ValueError("QuantParams alignment")
    names = ("fp4", "fp4.fc1", "fp4.fc2") + tuple(
        f"fp4.{stage}.{field}" for stage in ("fc1", "fc2") for field in (
            "use_per_expert_act_scale", "act_global_scale", "weight_block_scale", "global_scale"))
    m1_abi_probe.exact(q["fields"], names, "QuantParams fields")
    for name, field in q["fields"].items():
        m1_abi_probe.exact(field, ("offset", "size", "align"), "QuantParams field")
        for key, low in (("offset", 0), ("size", 1), ("align", 1)):
            m1_abi_probe.integer(field[key], low, 65536, "QuantParams field " + key)
        if field["offset"] + field["size"] > size or field["align"] & (field["align"]-1):
            raise ValueError("QuantParams field bounds/alignment")
        if "." in name:
            parent = q["fields"][name.rsplit(".", 1)[0]]
            if (field["offset"] < parent["offset"] or
                    field["offset"] + field["size"] > parent["offset"] + parent["size"]):
                raise ValueError("QuantParams nested field bounds")


def check_owners(owners):
    common = {"scope", "scratch_bytes", "allocations", "stream", "reused_scratch_cases", "tma_or_gemm_calls"}
    native = {"native_opt_in_tested", "stock_fallback_cases", "candidate_cases", "rejection_checks", "quantparams_unchanged"}
    if type(owners) is not dict or set(owners) not in (common, common | native):
        raise ValueError("fixture ownership fields")
    if (owners["scope"] != "fixture_owned_disjoint_allocations" or
            owners["stream"] != "one_owned_nonblocking_stream" or
            type(owners["reused_scratch_cases"]) is not int or owners["reused_scratch_cases"] != 3 or
            type(owners["tma_or_gemm_calls"]) is not int or owners["tma_or_gemm_calls"] != 0):
        raise ValueError("fixture ownership scope")
    sizes = (32, 32, 1408, 22528, 4) + (32, 32, 1032, 11264, 32, 2883584) * 2
    if type(owners["allocations"]) is not list or len(owners["allocations"]) != len(sizes):
        raise ValueError("fixture allocation count")
    for i, (p, capacity) in enumerate(zip(owners["allocations"], sizes)):
        expected = {"owner": f"alloc_{i}", "bytes": capacity+64, "origin": 32,
                    "capacity": capacity, "alignment": 32, "alignment_verified": True}
        if not m1_abi_probe.same(p, expected):
            raise ValueError("fixture allocation extent/origin/alignment")
    m1_abi_probe.integer(owners["scratch_bytes"], 1, 8 << 20, "fixture scratch")
    if owners["scratch_bytes"] != sum(n+64 for n in sizes):
        raise ValueError("fixture scratch total")
    if native.issubset(owners):
        if (owners["native_opt_in_tested"] is not True or owners["quantparams_unchanged"] is not True or
                type(owners["stock_fallback_cases"]) is not int or owners["stock_fallback_cases"] != 3 or
                type(owners["candidate_cases"]) is not int or owners["candidate_cases"] != 3 or
                type(owners["rejection_checks"]) is not int or owners["rejection_checks"] != 22):
            raise ValueError("native opt-in observation fields")


def compare(fixtures, captures, abi):
    fixtures, captures, abi = Path(fixtures), Path(captures), Path(abi)
    m1_abi_probe.check_host_probe(metadata(abi, "host-abi.json"))
    check_quantparams(metadata(abi, "quantparams.json"))
    initial = b"\xA7" * 32 + b"\xD3" * 2883584 + b"\xB6" * 32
    before = initial
    cases = []
    for i, ids in enumerate(fixture.ROUTES):
        name = f"case{i}"
        raw = fixture.read_exact(fixtures / (name + ".input"), 24000)
        if struct.unpack("<8i", raw[:32]) != ids:
            raise ValueError("fixture route changed")
        row = oracle.InputRow(ids, raw[32:64], raw[64:1472], raw[1472:])
        expected, after = fixture.expected(row, before)
        if fixture.read_exact(captures / (name + ".stock-before"), len(before)) != before:
            raise ValueError("stock SF scratch continuity mismatch")
        roles = {"slot_to_sorted": (0,32), "sorted_to_slot": (32,64), "offsets": (64,1096),
                 "expanded_aq": (1096,12360), "weight_bits": (12360,12392),
                 "guarded_sf": (12392, fixture.OUTPUT_BYTES),
                 "input_after": (fixture.OUTPUT_BYTES, fixture.OUTPUT_BYTES+24000)}
        results = {}
        for backend in ("stock", "fused"):
            blob = fixture.read_exact(captures / (name + "." + backend), fixture.OUTPUT_BYTES+24000)
            expected_blob = expected + raw
            results[backend] = {role: blob[start:end] == expected_blob[start:end]
                                for role, (start,end) in roles.items()}
            if blob != expected_blob:
                bad = [role for role, good in results[backend].items() if not good]
                raise ValueError(backend + " differs from independent oracle: " + ", ".join(bad))
        prepared = oracle.prepare(row, *fixture.contexts(), reference_abi=oracle.REFERENCE_ABI)
        cases.append({"case": i, "roles": results,
                      "useful_sf_bytes": sum(len(w.data) for w in prepared.expanded_sf_writes)})
        before = after
    owners = metadata(captures, "owners.json")
    check_owners(owners)
    return {"scope": "installed maps/packed expansion exact bytes; no TMA or GEMM execution",
            "cases": cases, "byte_equal": True, "typed_host_layouts_equal": True,
            "fixture_scratch_bytes": owners["scratch_bytes"], "fixture_owners_verified": True,
            "native_opt_in_tested": owners.get("native_opt_in_tested", False),
            "logical_sf_consumer_mask_bytes_per_active_expert": 176,
            "conservative_sf_consumer_mask_bytes": 2883584,
            "all_conservative_mask_bytes_compared": True,
            "producer_identity_verified": False,
            "runtime_workspace_owners_verified": False, "physical_gemm_consumer_masks_verified": False,
            "candidate_selectable": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures"); parser.add_argument("captures"); parser.add_argument("abi")
    args = parser.parse_args()
    print(json.dumps(compare(args.fixtures, args.captures, args.abi), indent=2))
