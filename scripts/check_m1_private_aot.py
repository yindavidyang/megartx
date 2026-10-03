"""CUDA-hidden CPU dlopen dry-run through pinned AOT fast path; no rebuild."""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from m1_private_aot import MODULE_PINS, sha, validate_cache


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("private AOT CPU dry-run requires CUDA_VISIBLE_DEVICES empty")
    root = Path(os.environ["MEGARTX_M1_PRIVATE_AOT"]).resolve()
    manifest = validate_cache(root)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized during AOT CPU dry-run")
    from flashinfer.jit.core import JitSpecNvcc
    results = []
    before = {n: sha(Path(manifest["cache"]) / n / "build.ninja") for n in MODULE_PINS}
    with patch.object(JitSpecNvcc, "build", side_effect=RuntimeError("CPU dry-run rebuild forbidden")) as build:
        modules = []
        for name in MODULE_PINS:
            spec = JitSpecNvcc(name, [], None, None, None, None)
            modules.append(spec.build_and_load())
            results.append({"name": name, "aot": spec.is_aot,
                            "module_sha256": sha(spec.aot_path),
                            "resolved_module": str(spec.aot_path.resolve())})
        if build.call_count or not all(r["aot"] for r in results):
            raise RuntimeError("private AOT fast path not used")
    after = {n: sha(Path(manifest["cache"]) / n / "build.ninja") for n in MODULE_PINS}
    if before != after or torch.cuda.is_initialized():
        raise RuntimeError("CPU AOT check modified Ninja or initialized CUDA")
    report = {"schema": "megartx-private-aot-cpu-dry-run-v1", "source_head": manifest["source_head"],
              "manifest_sha256": sha(root / "manifest.json"), "build_calls": 0,
              "cuda_initialized": False, "module_hashes": MODULE_PINS,
              "ninja_hashes_unchanged": before, "modules": results, "passed": True}
    (root / "cpu-dry-run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
