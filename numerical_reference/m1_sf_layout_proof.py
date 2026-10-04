"""Check host-only installed-layout output against the independent M1 oracle."""
import hashlib
import json
import struct

from m1_preparation_reference import BufferRef, grouped_sf_base, sf_coordinate


def verify_host_layouts(proof):
    if (proof.get("scope") != "host_only_pinned_cutlass_m1_sf_layout"
            or proof.get("dynamic_leaf_drift_controls") != 20
            or proof.get("stage_drift_controls") != 6 or len(proof.get("stages", [])) != 2):
        raise ValueError("host proof identity/control drift")
    stages, active_cases = [], 0
    for stage, k in zip(proof["stages"], (2816, 704)):
        count = 128 * (k // 16)
        actual = stage["offsets"]
        if (stage["k"] != k or stage["rows"] != 128 or stage["bytes"] != count
                or stage["inactive_rows"] != 0 or type(stage["inactive_bytes"]) is not int
                or len(actual) != count or any(type(v) is not int for v in actual)):
            raise ValueError("host carrier geometry drift")
        expected = [sf_coordinate(row, block, k // 16)
                    for row in range(128) for block in range(k // 16)]
        if actual != expected or sorted(actual) != list(range(count)):
            raise ValueError("installed physical offsets differ from independent exhaustive oracle")
        # Every physically possible selected (expert, rank) for eight distinct
        # routes. Prefix/rank is intentionally recomputed; it is never cached.
        cases = 0
        capacity = grouped_sf_base(128, 8, 2816)  # Shared FC1/FC2 SF owner.
        for expert in range(128):
            for rank in range(8):
                if rank > expert or 7-rank > 127-expert:
                    continue
                base = ((rank+127*expert+127)//128)*128*(k//16)
                if base != grouped_sf_base(expert, rank, k) or base+count > capacity:
                    raise ValueError("active expert/prefix carrier escapes owner")
                # Dense carrier proof covers all row-zero writes and physical
                # padding reads, including the last byte. A one-byte-short
                # retained subview must fail the existing extent inequality.
                if max(actual)+1 != count:
                    raise ValueError("physical padding/boundary proof failed")
                BufferRef("sf",base,count,base+count)
                try:
                    BufferRef("sf",base,count,base+count-1)
                except ValueError:
                    pass
                else:
                    raise ValueError("short carrier accepted by independent extent contract")
                cases += 1
        active_cases += cases
        stages.append({"k": k, "carrier_bytes": count,
                       "physical_coordinates": count, "padding_coordinates": 127*(k//16),
                       "active_expert_prefix_cases": cases,
                       "inactive_rows": stage["inactive_rows"], "inactive_cosize": stage["inactive_bytes"],
                       "offsets_sha256": hashlib.sha256(struct.pack(f"<{count}I", *actual)).hexdigest()})
    return {"schema": "megartx-m1-sf-layout-cpu-proof-v1", "stages": stages,
            "active_expert_prefix_cases": active_cases,
            "initial_proof_coordinates": sum(s["physical_coordinates"] for s in stages),
            "source_coordinates_avoided_per_preparation": 8*sum(s["physical_coordinates"] for s in stages),
            "source_coordinates_avoided_per_120_call_lane": 120*8*sum(s["physical_coordinates"] for s in stages),
            "dynamic_leaf_drift_controls": 20, "stage_drift_controls": 6,
            "cuda_calls": 0, "extra_device_allocation_bytes": 0,
            "native_bridge_compiled": False, "gpu_execution_verified": False,
            "performance_qualified": False, "graphs_qualified": False}


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host_proof", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = verify_host_layouts(json.loads(args.host_proof.read_text()))
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
