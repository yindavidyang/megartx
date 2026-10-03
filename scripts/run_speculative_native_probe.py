"""CPU source/protocol freeze only; GPU launch stays with reviewed owned lifecycle."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from megartx.speculative_native_probe import ADAPTER_FILES, digest, inspect_sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-root", type=Path,
                        help="Read/hash installed Python source; no runtime imports")
    parser.add_argument("--output", type=Path, help="New private freeze receipt path")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    protocol = root / "docs/design/speculative-native-protocol.json"
    result = {"schema": "megartx-speculative-native-cpu-freeze-v1",
              "protocol_sha256": digest(protocol), "gpu_enabled": False,
              "implementation_sha256": digest(root / "src/megartx/speculative_native_probe.py"),
              "driver_sha256": digest(Path(__file__)),
              "adapter_source_sha256": {name: digest(root / "src/megartx" / name) for name in ADAPTER_FILES},
              "sources": inspect_sources(args.installed_root) if args.installed_root else None,
              "blocking_extension": "reviewed owned lifecycle EngineCore utility + Worker extension",
              "native_executed": False}
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(payload)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
