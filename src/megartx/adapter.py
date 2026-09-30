"""Fail-closed boundary for a future, version-selected FlashInfer host adapter.

Installed package names, requested backend flags and unsupported paths are not
dispatch proof. This scaffold implements no inference backend, even when an
evidence packet is structurally valid and its local artifact hash matches.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .contracts import ContractError, validate


class BackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class EvidenceReceipt:
    artifact_sha256: str
    checkpoint_revision: str
    environment_sha256: str
    numerical_lane: str


def check_dispatch_packet(packet, artifact_root, checkpoint_revision, environment_sha256,
                          numerical_lane, expected_providers):
    """Validate a reviewed local evidence packet; does not independently trace GPU dispatch."""
    validate(packet, "dispatch")
    if packet["classification"] != "verified_flashinfer":
        raise ContractError("actual FlashInfer dispatch remains unverified/unsupported/fallback")
    for key, expected in (("checkpoint_revision", checkpoint_revision),
                          ("environment_sha256", environment_sha256), ("numerical_lane", numerical_lane)):
        if not expected or packet[key] != expected:
            raise ContractError(f"dispatch {key} does not match the intended experiment")
    required = ("host_runtime", "host_revision", "flashinfer_revision", "artifact_relative_path",
                "artifact_sha256", "reviewer_reference")
    if any(not packet[key] for key in required):
        raise ContractError("dispatch packet lacks pinned runtime or reviewed trace provenance")
    observations = packet["observations"]
    if len(observations) != 2 or {x["component"] for x in observations} != {"attention", "moe"}:
        raise ContractError("dispatch packet needs exact attention and MoE observations")
    if set(expected_providers) != {"attention", "moe"} or "flashinfer" not in expected_providers.values():
        raise ContractError("freeze intended attention/MoE providers with at least one FlashInfer component")
    for item in observations:
        if item["provider"] != expected_providers[item["component"]]:
            raise ContractError("actual component provider differs from the intended provider")
        if item["provider"] != "flashinfer" and item["component"] not in packet["non_flashinfer_components"]:
            raise ContractError("non-FlashInfer component must be explicitly declared")
        if numerical_lane == "nvfp4_w4a4" and item["component"] == "moe" and not item["native_nvfp4"]:
            raise ContractError("native W4A4 MoE dispatch is not established")
    root = Path(artifact_root).resolve()
    relative = Path(packet["artifact_relative_path"])
    artifact = (root / relative).resolve()
    if relative.is_absolute() or not artifact.is_relative_to(root):
        raise ContractError("dispatch trace must be contained in the supplied artifact directory")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != packet["artifact_sha256"]:
        raise ContractError("dispatch trace hash mismatch")
    return EvidenceReceipt(digest, checkpoint_revision, environment_sha256, numerical_lane)


class FlashInferAdapter:
    """An unimplemented integration boundary, never a synthetic runtime."""

    def run(self, *args, evidence=None, **kwargs):
        if not isinstance(evidence, EvidenceReceipt):
            raise BackendUnavailable("FlashInfer execution blocked: verified dispatch packet required")
        raise BackendUnavailable("No FlashInfer runtime adapter is implemented; GPU validation is pending")
