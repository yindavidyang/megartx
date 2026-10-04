"""CPU-only task-local preparation. No runtime import, install, or GPU query.

An existing wheel --target installation supplies real distribution metadata.
The execution supervisor alone creates the fresh writable runtime root. Source
freeze and metadata checks do not create caches or grant execution authority.
"""
import configparser
import os
from pathlib import Path
import stat
import re

from .speculative_native_probe import ProbeError

IPC_PATH_LIMIT_BYTES = 107  # Linux sockaddr_un.sun_path, excluding trailing NUL.
IPC_UUID_SUFFIX_BYTES = 37  # slash plus str(uuid.uuid4()).
IPC_MARGIN_BYTES = 16  # Reserved for additional runtime naming; not consumed.
RUNTIME_BLOCKER = "task_local_runtime_and_entrypoint_binding_required"
ENTRYPOINT = {"megartx_scale_adapter": "megartx.vllm_scale_plugin:install"}


def canonical_path(value):
    """Reject aliases rather than silently resolving paths into shared state."""
    path = Path(value)
    if (not path.is_absolute() or str(path) != str(value)
            or path.resolve() != path or any(p in (".", "..") for p in path.parts)
            or any(char in str(path) for char in ("\x00", "\n", ":"))):
        raise ProbeError("Preparation paths must be canonical absolute nonsymlink paths")
    return path


def _separate(left, right):
    if left == right or left in right.parents or right in left.parents:
        raise ProbeError("Task-local paths overlap shared/source/evidence paths")


def ipc_preflight(tmp):
    size = len(str(canonical_path(tmp)).encode("utf-8"))
    required = size + IPC_UUID_SUFFIX_BYTES + IPC_MARGIN_BYTES
    if required > IPC_PATH_LIMIT_BYTES:
        raise ProbeError("Runtime IPC path exceeds UTF-8 socket budget including margin")
    return {"root_utf8_bytes": size, "uuid_suffix_bytes": IPC_UUID_SUFFIX_BYTES,
            "reserved_margin_bytes": IPC_MARGIN_BYTES,
            "sun_path_payload_limit_bytes": IPC_PATH_LIMIT_BYTES}


def directory(path):
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ProbeError("Preparation directory must be a nonsymlink owned directory")
    return info


def validate_entrypoint_discovery(binding, search_path=None):
    """Model importlib's name-deduplicated discovery without loading a plugin."""
    from importlib.metadata import distributions
    import sys
    seen, matches = set(), []
    for dist in distributions(path=sys.path if search_path is None else search_path):
        name = re.sub(r"[-_.]+", "_", dist.metadata.get("Name", "")).lower()
        if name in seen:
            continue
        seen.add(name)
        for entry in dist.entry_points:
            if entry.group == "vllm.general_plugins" and entry.name in ENTRYPOINT:
                matches.append((name, entry.value, str(Path(dist.locate_file("")).resolve())))
    expected = [("megartx", ENTRYPOINT["megartx_scale_adapter"], binding["root"])]
    if matches != expected:
        raise ProbeError("Task-local wheel entrypoint is shadowed, duplicated or unavailable")


def entrypoint_binding(root, project):
    """Read a real task-local wheel installation, never manufacture dist-info."""
    from .speculative_native_plan import BASE, hash_file, object_digest
    root, project = canonical_path(root), canonical_path(project)
    _separate(root, BASE)
    directory(root)
    entries = list(root.glob("*.dist-info"))
    if len(entries) != 1 or entries[0].name != "megartx-0.0.1.dist-info":
        raise ProbeError("Exactly one task-local megartx wheel distribution required")
    if any(root.glob("*.pth")) or any(root.glob("*.egg-link")):
        raise ProbeError("Task-local entrypoint path may not inject search paths")
    dist = entries[0]
    directory(dist)
    files = {}
    for path in dist.iterdir():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ProbeError("Wheel metadata must be owned regular files")
        files[path.name] = hash_file(path, 1 << 20)
    if not {"METADATA", "WHEEL", "entry_points.txt", "RECORD"} <= set(files):
        raise ProbeError("Complete wheel distribution metadata required")
    from email.parser import Parser
    metadata = Parser().parsestr((dist / "METADATA").read_text())
    if metadata.get_all("Name") != ["megartx"] or metadata.get_all("Version") != ["0.0.1"]:
        raise ProbeError("Wheel distribution identity differs")
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    try:
        parser.read_string((dist / "entry_points.txt").read_text())
        if dict(parser["vllm.general_plugins"]) != ENTRYPOINT:
            raise ProbeError("Wheel plugin entrypoint differs from committed package")
    except (configparser.Error, KeyError) as error:
        raise ProbeError("Malformed wheel plugin entrypoint") from error
    # PYTHONPATH puts the committed source first. Also reject a stale wheel
    # copy so spawn/import search ordering cannot silently select old code.
    def package_files(base):
        package = base / "megartx"
        canonical_path(package)
        directory(package)
        result = {}
        for path in package.rglob("*.py"):
            canonical_path(path)
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ProbeError("Wheel source must be nonsymlink regular files")
            result[str(path.relative_to(base))] = hash_file(path, 1 << 20)
        return result
    source_files, installed_files = package_files(project / "src"), package_files(root)
    if not source_files or source_files != installed_files:
        raise ProbeError("Task-local wheel source differs from committed checkout")
    result = {"root": str(root), "distribution": dist.name,
            "metadata_sha256": files, "package_source_sha256": object_digest(source_files),
            "package_source_file_count": len(source_files),
            "entrypoint": dict(ENTRYPOINT), "pyproject_sha256": hash_file(project / "pyproject.toml", 1 << 20)}
    validate_entrypoint_discovery(result, [str(project / "src"), str(project / "numerical_reference"), str(root)])
    return result


def runtime_binding(project, runtime_root, entrypoint_root):
    from .speculative_native_plan import BASE
    project, root = canonical_path(project), canonical_path(runtime_root)
    entrypoint = entrypoint_binding(entrypoint_root, project)
    for other in (BASE, project, Path(entrypoint["root"])):
        _separate(root, other)
    paths = {"root": str(root), "cache": str(root / "cache"),
             "tmp": str(root / "tmp"), "home": str(root / "home"),
             "config": str(root / "config")}
    return {"paths": paths, "project_root": str(project), "entrypoint": entrypoint,
            "ipc": ipc_preflight(paths["tmp"]), "creation": "exclusive_fresh_owned_0700",
            "shared_cache_fallback": False}


def runtime_environment(binding):
    paths = binding["paths"]
    cache = Path(paths["cache"])
    return {"HOME": paths["home"], "XDG_CACHE_HOME": str(cache),
            "XDG_CONFIG_HOME": paths["config"], "HF_HOME": str(cache / "huggingface"),
            "VLLM_CACHE_ROOT": str(cache / "vllm"), "VLLM_CONFIG_ROOT": str(Path(paths["config"]) / "vllm"),
            "TMPDIR": paths["tmp"], "TMP": paths["tmp"], "TEMP": paths["tmp"],
            "VLLM_RPC_BASE_PATH": paths["tmp"],
            "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer-workspace"),
            "TRITON_CACHE_DIR": str(cache / "triton"), "CUDA_CACHE_PATH": str(cache / "cuda"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"), "TORCH_EXTENSIONS_DIR": str(cache / "torch-extensions"),
            "PYTHONPATH": ":".join((str(Path(binding["project_root"]) / "src"),
                                    str(Path(binding["project_root"]) / "numerical_reference"),
                                    binding["entrypoint"]["root"]))}


def evidence_path(private, project):
    from .speculative_native_plan import BASE
    private = canonical_path(private)
    for other in (BASE, canonical_path(project)):
        _separate(private, other)
    return private


def runtime_preflight(binding, private, *, created=False, installed_root=None):
    """Read-only path/ownership validation; fresh means no prior run artifacts."""
    from .speculative_native_plan import canonical
    expected = runtime_binding(binding["project_root"], binding["paths"]["root"], binding["entrypoint"]["root"])
    if canonical(binding) != canonical(expected):
        raise ProbeError("Frozen runtime/entrypoint metadata changed")
    if installed_root is not None:
        validate_entrypoint_discovery(binding["entrypoint"],
            runtime_environment(binding)["PYTHONPATH"].split(":") + [str(installed_root)])
    private = evidence_path(private, binding["project_root"])
    _separate(Path(binding["paths"]["root"]), private)
    _separate(Path(binding["entrypoint"]["root"]), private)
    for key, value in binding["paths"].items():
        path = canonical_path(value)
        if created:
            if stat.S_IMODE(directory(path).st_mode) != 0o700:
                raise ProbeError("Runtime directories must remain private mode 0700")
        elif path.exists():
            raise ProbeError("Runtime root must be exclusively fresh; reuse forbidden")
    return expected


def create_runtime(binding, private):
    runtime_preflight(binding, private)
    paths = binding["paths"]
    Path(paths["root"]).mkdir(mode=0o700)  # exclusive, no parents or exist_ok
    for key, value in paths.items():
        if key != "root":
            Path(value).mkdir(mode=0o700)
    runtime_preflight(binding, private, created=True)
