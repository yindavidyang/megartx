"""Verify private first-receipt evidence and emit scalar-only comparison."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from megartx.speculative_native_plan import LIMITS, read_json, validate_plan
from megartx.speculative_native_compare import compare_files
from megartx.speculative_native_evidence import PrivateEvidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--private-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = validate_plan(read_json(args.plan, LIMITS["plan_bytes"]), PROJECT)
    result = compare_files(args.private_directory, plan)
    with PrivateEvidence(args.private_directory) as evidence:
        evidence.write("native-receipt-comparison.scalars.json", result, cap=64 << 10)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
