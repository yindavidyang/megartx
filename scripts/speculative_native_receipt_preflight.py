"""Source-only freeze/validation. No runtime imports, device query or GPU job."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from megartx.speculative_native_plan import (LIMITS, freeze, installed_preflight,
    checkpoint_preflight, read_json, validate_plan)
from megartx.speculative_native_evidence import PrivateEvidence
from megartx.speculative_native_preparation import runtime_preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--plan", type=Path)
    parser.add_argument("--runner-lane", choices=("v1-legacy", "v2"),
                        help="Required for freeze; v1-legacy retains its original guard and schema")
    parser.add_argument("--client-mode", choices=("sync", "async"), default="async")
    parser.add_argument("--runtime-root", type=Path,
                        help="V2 freeze: short fresh task-owned runtime path; no directories are created")
    parser.add_argument("--entrypoint-root", type=Path,
                        help="V2 freeze: existing exact source wheel --target installation")
    parser.add_argument("--installed-root", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path,
                        help="CPU full streaming file rehash, no model parse/download")
    parser.add_argument("--private-directory", type=Path,
                        help="Existing owned mode-0700 directory, new immutable output only")
    args = parser.parse_args(argv)
    if args.freeze != (args.runner_lane is not None):
        parser.error("--freeze requires explicit --runner-lane; existing plans bind their own lane")
    if args.freeze and (args.installed_root or args.checkpoint_manifest):
        parser.error("Freeze and installed/checkpoint inspection are separate phases")
    if not args.freeze and (args.runtime_root or args.entrypoint_root):
        parser.error("Runtime and entrypoint paths must be bound at freeze time")
    if args.freeze:
        result = freeze(PROJECT, client_mode=args.client_mode, runner_lane=args.runner_lane,
                        runtime_root=args.runtime_root, entrypoint_root=args.entrypoint_root)
        name = "native-v2-receipt-plan.private.json" if args.runner_lane == "v2" else "native-receipt-plan.private.json"
    else:
        plan = validate_plan(read_json(args.plan, LIMITS["plan_bytes"]), PROJECT)
        result = {"schema": ("megartx-native-v2-receipt-preflight-v1" if plan.get("runner_lane") == "v2"
                             else "megartx-native-receipt-preflight-v1"), "plan_sha256": plan["plan_sha256"],
                  "source_head": plan["source_head"], "runtime_imports": False, "device_queries": False,
                  "gpu_authorized": False, "preflight_blockers": plan["preflight_blockers"]}
        if args.installed_root:
            result["installed"] = installed_preflight(args.installed_root, runner_lane=plan.get("runner_lane", "v1-legacy"))
        if args.checkpoint_manifest:
            result["checkpoint"] = checkpoint_preflight(args.checkpoint_manifest)
        if plan.get("runtime_binding") is not None:
            # Existing output directory is a separate immutable evidence store.
            if args.private_directory is None:
                parser.error("Bound V2 preparation requires --private-directory for path isolation checks")
            result["runtime"] = runtime_preflight(plan["runtime_binding"], args.private_directory,
                                                  installed_root=args.installed_root)
        name = "native-receipt-preflight.private.json"
    if args.private_directory:
        with PrivateEvidence(args.private_directory) as evidence:
            evidence.write(name, result, cap=LIMITS["plan_bytes"])
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
