"""Experiment provenance: resolved package versions and model/tokenizer commits.

Every training/evaluation run writes this manifest next to its outputs so a
result can be reproduced or rejected later. No patient text or identifiers are
included.
"""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import sys
from typing import Any

PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "scikit-learn",
    "pandas",
    "numpy",
    "peft",
    "trl",
    "tokenizers",
)


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
    }
    try:
        import torch

        environment["torch_cuda_available"] = bool(torch.cuda.is_available())
        environment["torch_mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if torch.cuda.is_available():
            environment["cuda_device"] = torch.cuda.get_device_name(0)
            environment["cuda_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    except Exception as exc:  # pragma: no cover - defensive provenance only
        environment["torch_error"] = str(exc)
    return environment


def model_provenance(model: Any | None, tokenizer: Any | None, model_id: str, revision: str | None) -> dict[str, Any]:
    """Capture the resolved commits that actually loaded, not just the request."""
    model_commit = getattr(getattr(model, "config", None), "_commit_hash", None)
    tokenizer_commit = getattr(tokenizer, "_commit_hash", None)
    if tokenizer_commit is None:
        tokenizer_commit = (getattr(tokenizer, "init_kwargs", None) or {}).get("_commit_hash")
    return {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_model_commit": str(model_commit) if model_commit else None,
        "resolved_tokenizer_commit": str(tokenizer_commit) if tokenizer_commit else None,
    }
