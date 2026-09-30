"""Read-only CPU utilities except the explicit schema export command."""

import argparse
import json
import sys
from pathlib import Path

from .contracts import ContractError, load_json, readiness, validate
from .inventory import collect_inventory
from .memory import estimate_memory
from .schema import SCHEMAS
from .stats import summarize


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU-only WP0/WP1 contracts; no model execution")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate", help="Check schema and semantic constraints, not GPU gates")
    check.add_argument("path", type=Path)
    ready = commands.add_parser("readiness", help="Report unresolved experiment freeze decisions")
    ready.add_argument("path", type=Path)
    inventory = commands.add_parser("inventory", help="Collect allowlisted host/package metadata")
    inventory.add_argument("--probe-nvidia-smi", action="store_true", help="Opt in to a read-only GPU metadata query")
    memory = commands.add_parser("memory", help="Estimate bytes from explicit local metadata, no downloads")
    memory.add_argument("--tensors", type=Path, required=True)
    memory.add_argument("--budget", type=Path, required=True)
    stats = commands.add_parser("summarize", help="Summarize paired requests, never a gate decision")
    stats.add_argument("path", type=Path)
    stats.add_argument("--bootstrap-samples", type=int, default=2000)
    stats.add_argument("--seed", type=int, default=0)
    stats.add_argument("--minimum-pairs", type=int, default=30)
    export = commands.add_parser("schemas", help="Export canonical JSON Schemas to a directory")
    export.add_argument("--directory", type=Path, default=Path("schemas"))
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            paths = sorted(args.path.glob("*.json")) if args.path.is_dir() else [args.path]
            if not paths:
                raise ContractError("no JSON contracts found")
            checked = []
            for path in paths:
                doc = validate(load_json(path))
                checked.append({"file": path.name, "kind": doc["kind"]})
            result = {"structurally_valid": checked, "note": "Validation is not evidence of a frozen contract or passed GPU gate"}
        elif args.command == "readiness":
            result = readiness(load_json(args.path))
        elif args.command == "inventory":
            result = collect_inventory(args.probe_nvidia_smi)
        elif args.command == "memory":
            result = estimate_memory(load_json(args.tensors), load_json(args.budget))
        elif args.command == "summarize":
            if args.minimum_pairs < 2:
                raise ContractError("minimum request pair count must be at least two")
            result = summarize(load_json(args.path), args.bootstrap_samples, args.seed, args.minimum_pairs)
        else:
            args.directory.mkdir(parents=True, exist_ok=True)
            for kind, schema in SCHEMAS.items():
                (args.directory / f"{kind}.schema.json").write_text(
                    json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = {"exported_schema_count": len(SCHEMAS)}
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ContractError, OSError, ValueError, TypeError) as error:
        print(f"megartx: {error}", file=sys.stderr)
        return 2
