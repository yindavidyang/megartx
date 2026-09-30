"""Redacted metadata collection without importing CUDA/PyTorch or changing a host."""

import csv
import importlib.metadata
import io
import platform
import re
import shutil
import subprocess


def collect_inventory(probe_gpu=False):
    packages = {}
    for name in ("torch", "flashinfer-python", "nvidia-cuda-runtime-cu12"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "schema_version": 1, "kind": "inventory", "evidence_scope": "metadata_only",
        "os": platform.system(), "architecture": platform.machine(),
        "python_version": platform.python_version(), "packages": packages,
        "tools_available": {tool: shutil.which(tool) is not None for tool in ("nvidia-smi", "nvcc")},
        "gpu_probe": "not_requested", "gpus": [],
        "warning": "No hostname, usernames, paths, serials, UUIDs or environment variables are collected. No kernels run."
    }
    if not probe_gpu:
        return report
    if not report["tools_available"]["nvidia-smi"]:
        report["gpu_probe"] = "unavailable"
        return report
    try:
        process = subprocess.run([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False)
        if process.returncode:
            report["gpu_probe"] = "unavailable_or_unsupported_query"
            return report
        for row in csv.reader(io.StringIO(process.stdout)):
            if len(row) != 5:
                raise ValueError("unexpected GPU query columns")
            name, driver, total, free, compute = [item.strip() for item in row]
            # Do not echo arbitrary stdout/stderr from a failed or malformed probe.
            if not re.fullmatch(r"[A-Za-z0-9 ._-]{1,100}", name):
                raise ValueError("invalid GPU name")
            if not re.fullmatch(r"[0-9.]+", driver) or not re.fullmatch(r"[0-9]+\.[0-9]+", compute):
                raise ValueError("invalid version")
            if not total.isdigit() or not free.isdigit():
                raise ValueError("invalid memory size")
            report["gpus"].append({"name": name, "driver_version": driver,
                                    "total_bytes": int(total) * 1024**2,
                                    "free_bytes": int(free) * 1024**2,
                                    "compute_capability": compute})
        report["gpu_probe"] = "observed" if report["gpus"] else "no_devices"
    except (OSError, ValueError, subprocess.SubprocessError):
        report["gpus"] = []
        report["gpu_probe"] = "unavailable_or_unparseable"
    return report
