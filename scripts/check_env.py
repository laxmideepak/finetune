#!/usr/bin/env python3
"""Verify the local environment before running any experiment.

Checks the required encoder stack, optional generative stack, compute devices,
the canonical label policy, and the shared data contract against the smoke
fixture. No network access is required.

Exit code 0 means the encoder path is ready; the generative stack is reported
separately because it intentionally lives in a second environment (risk R5).

    python scripts/check_env.py
    python scripts/check_env.py --json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRED_ENCODER = (
    "torch",
    "transformers",
    "tokenizers",
    "datasets",
    "accelerate",
    "numpy",
    "pandas",
    "sklearn",
)
OPTIONAL_GENERATIVE = ("peft", "trl")
PINNED = {
    "torch": "2.8.0",
    "transformers": "4.57.0",
    "tokenizers": "0.22.0",
    "datasets": "4.8.5",
    "accelerate": "1.15.0",
    "numpy": "2.1.3",
    "pandas": "2.3.1",
    "scikit-learn": "1.7.0",
    "peft": "0.21.2",
    "trl": "0.29.1",
}
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "smoke_splits.csv"


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _import(name: str) -> tuple[bool, str | None]:
    try:
        importlib.import_module(name)
        return True, None
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    args = parser.parse_args(argv)

    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "repo_root": str(REPO_ROOT),
        "required_encoder": {},
        "optional_generative": {},
        "devices": {},
        "policy": {},
        "data_contract": {},
        "warnings": [],
        "ok": True,
    }

    for module in REQUIRED_ENCODER:
        package = "scikit-learn" if module == "sklearn" else module
        version = _version(package)
        imported, error = _import(module)
        entry = {"package": package, "version": version, "pinned": PINNED.get(package), "imported": imported}
        if error:
            entry["error"] = error
        if not imported:
            report["ok"] = False  # type: ignore[index]
        report["required_encoder"][module] = entry  # type: ignore[index]
        if version and PINNED.get(package) and version != PINNED[package]:
            report["warnings"].append(  # type: ignore[union-attr]
                f"{package}=={version} differs from pinned {PINNED[package]}"
            )

    generative_ready = True
    for module in OPTIONAL_GENERATIVE:
        version = _version(module)
        imported, error = _import(module)
        entry = {"version": version, "pinned": PINNED.get(module), "imported": imported}
        if error:
            entry["error"] = error
        if not imported or (version and PINNED.get(module) and version != PINNED[module]):
            generative_ready = False
            if not imported:
                report["warnings"].append(f"generative package missing: {module}")  # type: ignore[union-attr]
            else:
                report["warnings"].append(
                    f"{module}=={version} differs from pinned {PINNED[module]}"
                )
        report["optional_generative"][module] = entry  # type: ignore[index]
    report["generative_ready"] = generative_ready

    try:
        import torch

        mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        report["devices"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "mps_available": mps,
            "cpu_threads": torch.get_num_threads(),
        }
        if not (torch.cuda.is_available() or mps):
            report["warnings"].append("No CUDA/MPS device; experiments will run on CPU")  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover - environment dependent
        report["devices"] = {"error": str(exc)}
        report["ok"] = False  # type: ignore[index]

    try:
        from src.label_policy import LABELS, POLICY_VERSION

        report["policy"] = {"version": POLICY_VERSION, "labels": list(LABELS)}
        if POLICY_VERSION != "smoking-status-v1":
            report["ok"] = False  # type: ignore[index]
            report["warnings"].append(f"unexpected policy version {POLICY_VERSION}")  # type: ignore[union-attr]
    except Exception as exc:
        report["policy"] = {"error": str(exc)}
        report["ok"] = False  # type: ignore[index]

    try:
        from src.data_contract import read_and_validate_csv

        _, manifest = read_and_validate_csv(str(FIXTURE))
        report["data_contract"] = {
            "fixture": str(FIXTURE),
            "n_rows": manifest["n_rows"],
            "data_fingerprint": manifest["data_fingerprint"],
            "split_counts": manifest["split_counts"],
        }
    except Exception as exc:
        report["data_contract"] = {"fixture": str(FIXTURE), "error": f"{type(exc).__name__}: {exc}"}
        report["ok"] = False  # type: ignore[index]

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Python {report['python']} on {report['machine']} ({report['platform']})")
        print("\nEncoder stack:")
        for module, entry in report["required_encoder"].items():  # type: ignore[union-attr]
            state = "ok" if entry["imported"] else "MISSING"
            pin = f" (pin {entry['pinned']})" if entry["pinned"] else ""
            print(f"  {module:<14} {entry['version'] or '-':<12} {state}{pin}")
        print("\nGenerative stack (separate environment):")
        for module, entry in report["optional_generative"].items():  # type: ignore[union-attr]
            pin = f" (pin {entry['pinned']})" if entry["pinned"] else ""
            print(f"  {module:<14} {entry['version'] or '-':<12} {'ok' if entry['imported'] else 'MISSING'}{pin}")
        print("\nDevices:", json.dumps(report["devices"]))
        print("Policy:", json.dumps(report["policy"]))
        print("Data contract:", json.dumps(report["data_contract"]))
        if report["warnings"]:
            print("\nWarnings:")
            for warning in report["warnings"]:  # type: ignore[union-attr]
                print(f"  - {warning}")
        print("\nRESULT:", "READY" if report["ok"] else "NOT READY")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
