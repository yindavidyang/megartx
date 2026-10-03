"""Explicit private reuse of eight hash-pinned FlashInfer AOT artifacts.

Uses the unmodified 0.6.18.post1 AOT fast path. Its ordinary JIT path remains
available for other specs; rebuilding a promoted incumbent fails closed.
"""
import hashlib
import json
import os
from pathlib import Path

LOADER_PINS = {
    "jit/core.py": "454316d67f22ebe3035cf17a13e9d01ffe5eed392d4e96adb952f3955bcb132b",
    "jit/cpp_ext.py": "c2ebe90fa1ddc3896553c51a4799e43ee8e2cd506740d3e3b1493280bfd7262b",
    "jit/env.py": "c3955fd0b83154356942840c1217e7c64a3602766feddbd701f814bae25ff47e",
    "jit/fused_moe.py": "b41f213cba1be67d7367e0f01b5612992f38ca06e29de1d9e97c8194fa9bc58d",
}
MODULE_PINS = {
    "batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_True_use_logits_cap_False_f16qk_False": "5dc7fc27cce23c3859551aeb10020477f67c32b141772be3b2eb847e29b4b401",
    "batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_512_head_dim_vo_512_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False": "e82f1ef88888879ce2116d151f2ff19ae390ef48c707992975ae723c8772bf2d",
    "fp4_gemm_cutlass_sm120": "3b8557cf008c54ffcc23b07666cd027d718bf13ef4111e447a6110c36f3056d1",
    "fp4_quantization_120f": "3201b425718a1046cb8e25e8e310e7d99b97f16a12ba967c853bb564ae1f1834",
    "fused_moe_120": "dc26a85431c946b0f636ddd2e08aa676f6340fd085361b21e8b0fe00617d28f9",
    "sampling": "9fb99580e484ecb95652c2f5e5fde06c7450f74225564dabe4d10091a54b5eaf",
    "trtllm_utils": "51efa73e008ef82b528962d6e834e8a41b80d9d3e7beaf7612b25f116b72968a",
    "xqa_input_bf16_kv_cache_bf16_output_bf16_page_size_16_head_dim_256_head_group_ratio_2_use_sliding_window_True_use_spec_dec_False_spec_q_seq_len_1": "4073c2ad18d68d01e6e302aae5460362073664acfdd074d5c1219dad9a884f10",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_cache(root, source_head=None):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != "megartx-private-aot-v1"
            or manifest.get("loader_hashes") != LOADER_PINS
            or manifest.get("module_hashes") != MODULE_PINS
            or source_head is not None and manifest.get("source_head") != source_head):
        raise RuntimeError("private AOT manifest/source scope differs")
    fi = Path(manifest["flashinfer_root"])
    if any(sha(fi / name) != expected for name, expected in LOADER_PINS.items()):
        raise RuntimeError("installed FlashInfer AOT loader differs from exact release")
    cache = Path(manifest["cache"])
    if root.is_relative_to(cache) or cache.is_relative_to(fi) or not cache.is_relative_to(root.parent):
        raise RuntimeError("private AOT must use a separate task-local cache")
    if manifest.get("helper_sha256") != sha(__file__):
        raise RuntimeError("private AOT helper differs from frozen source")
    for name, expected in manifest["shim_hashes"].items():
        if sha(root / "python" / name) != expected:
            raise RuntimeError("private AOT Python copy differs")
    for name, expected in MODULE_PINS.items():
        module = cache / name / (name + ".so")
        link = root / "aot" / name / (name + ".so")
        if (not module.resolve().is_relative_to(cache) or link.resolve() != module.resolve()
                or sha(link) != expected):
            raise RuntimeError("private AOT incumbent module identity differs: " + name)
    return manifest


def install_guard():
    root = Path(os.environ["MEGARTX_M1_PRIVATE_AOT"])
    validate_cache(root)
    import flashinfer.jit.core as core
    if Path(core.__file__).resolve() != (Path(validate_cache(root)["flashinfer_root"]) / "jit/core.py").resolve():
        raise RuntimeError("private AOT imported a different loader")
    manifest = validate_cache(root)
    if (core.jit_env.FLASHINFER_AOT_DIR.resolve() != (root / "aot").resolve()
            or core.jit_env.FLASHINFER_JIT_DIR.resolve() != Path(manifest["cache"]).resolve()):
        raise RuntimeError("private AOT/JIT workspace discovery differs")
    original = core.JitSpecNvcc.build
    if getattr(original, "_megartx_aot_guard", False):
        return

    def guarded_build(spec, *args, **kwargs):
        if spec.name in MODULE_PINS:
            raise RuntimeError("private pinned AOT load failed; incumbent rebuild forbidden: " + spec.name)
        return original(spec, *args, **kwargs)

    guarded_build._megartx_aot_guard = True
    core.JitSpecNvcc.build = guarded_build


def prepare(cache, flashinfer_root, output, source_head):
    import shutil
    cache, flashinfer_root, output = map(lambda p: Path(p).resolve(), (cache, flashinfer_root, output))
    if any(sha(flashinfer_root / n) != h for n, h in LOADER_PINS.items()):
        raise RuntimeError("installed loader pin mismatch")
    if any(sha(cache / n / (n + ".so")) != h for n, h in MODULE_PINS.items()):
        raise RuntimeError("incumbent module pin mismatch")
    output.mkdir(mode=0o700)
    for name in MODULE_PINS:
        directory = output / "aot" / name
        directory.mkdir(parents=True)
        (directory / (name + ".so")).symlink_to(cache / name / (name + ".so"))
    source = Path(__file__).resolve().parent
    shutil.copytree(source / "m1_aot_cache", output / "python", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(__file__, output / "python/m1_private_aot.py")
    manifest = {"schema": "megartx-private-aot-v1", "source_head": source_head,
                "cache": str(cache), "flashinfer_root": str(flashinfer_root),
                "loader_hashes": LOADER_PINS, "module_hashes": MODULE_PINS,
                "helper_sha256": sha(__file__),
                "shim_hashes": {str(p.relative_to(output / "python")): sha(p)
                                for p in (output / "python").rglob("*.py")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validate_cache(output, source_head)
    return manifest


if __name__ == "__main__":
    import argparse
    import subprocess
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--flashinfer-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip():
        raise RuntimeError("freeze source before preparing private AOT")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    print(json.dumps(prepare(args.cache, args.flashinfer_root, args.output, head), indent=2))
