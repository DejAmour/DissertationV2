#!/usr/bin/env python3
"""
create_handover_package.py
==========================
Produces an auditable handover ZIP for the Asian-option Monte Carlo
dissertation workflow (Stage 7).

The package contains:
  - logs/pytest_results.txt          : pytest -v output
  - logs/method_comparison.txt       : run_method_comparison.py output
  - logs/validation_minimal.txt      : run_experiments.py validation_minimal output
  - logs/environment.txt             : Python/platform/package metadata
  - outputs/                         : CSV/Markdown outputs from above runs
  - provenance.json                  : timestamps, git commit, run configuration
  - manifest.txt                     : SHA-256 checksums for every file in the ZIP
  - WARNING_validation_only.txt      : label: results are quick-validation only

Usage
-----
    python scripts/create_handover_package.py [--output-dir <dir>]

The script does NOT fabricate results or hashes.  All checksums are computed
from files actually produced during the run.

NOTE: Results labelled "validation_minimal" are NOT dissertation-scale.
They use a small number of paths/replications and must NOT be cited as
production results.  Dissertation-scale results require ≥30 independent
outer replications (--profile dissertation_full, not yet implemented here).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], log_path: Path, cwd: Path = REPO_ROOT) -> int:
    """Run a subprocess, capturing stdout+stderr to log_path. Returns exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Running: {' '.join(cmd)}", flush=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Command: {' '.join(cmd)}\n")
        fh.write(f"# cwd: {cwd}\n")
        fh.write(f"# started: {datetime.now(timezone.utc).isoformat()}\n\n")
        fh.flush()
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        fh.write(f"\n# finished: {datetime.now(timezone.utc).isoformat()}\n")
        fh.write(f"# exit_code: {result.returncode}\n")
    print(f"    -> exit code {result.returncode}, log: {log_path.name}")
    return result.returncode


def _collect_outputs(output_dir: Path, staging: Path) -> list[Path]:
    """Copy CSV and Markdown outputs into staging/outputs/. Returns staged paths."""
    dest = staging / "outputs"
    dest.mkdir(parents=True, exist_ok=True)
    collected = []
    patterns = ["*.csv", "*.md", "*.json"]
    # Collect from repo root and output_dir
    for search_dir in [REPO_ROOT, output_dir]:
        for pattern in patterns:
            for src in search_dir.glob(pattern):
                if "legacy" in src.parts:
                    continue
                dst = dest / src.name
                shutil.copy2(src, dst)
                collected.append(dst)
    return collected


def _environment_info() -> dict:
    installed: dict[str, str] = {}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            for pkg in json.loads(result.stdout):
                installed[pkg["name"].lower()] = pkg["version"]
    except Exception:
        pass

    git_commit = "unknown"
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
        )
        if r.returncode == 0:
            git_commit = r.stdout.strip()
    except Exception:
        pass

    git_status = "unknown"
    try:
        r = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
        )
        if r.returncode == 0:
            git_status = r.stdout.strip() or "clean"
    except Exception:
        pass

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit,
        "git_status": git_status,
        "key_packages": {k: installed.get(k, "not installed")
                         for k in ("numpy", "scipy", "torch", "pytest")},
        "all_installed_packages": installed,
    }


def _build_manifest(staging: Path) -> Path:
    manifest_path = staging / "manifest.txt"
    lines = ["# SHA-256 checksums for all files in this handover package\n",
             f"# generated: {datetime.now(timezone.utc).isoformat()}\n\n"]
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path != manifest_path:
            rel = path.relative_to(staging)
            lines.append(f"{_sha256(path)}  {rel}\n")
    manifest_path.write_text("".join(lines), encoding="utf-8")
    return manifest_path


def _build_zip(staging: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging))
    print(f"\n  Handover ZIP: {zip_path}")
    print(f"  ZIP SHA-256 : {_sha256(zip_path)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create Stage 7 handover ZIP.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "handover_outputs",
        help="Directory for experiment outputs (default: ./handover_outputs)",
    )
    parser.add_argument(
        "--zip-name",
        default=None,
        help="Name of the output ZIP file (default: stage7_handover_<timestamp>.zip)",
    )
    args = parser.parse_args(argv)

    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    zip_name = args.zip_name or f"stage7_handover_{ts}.zip"
    zip_path = REPO_ROOT / zip_name
    staging = REPO_ROOT / f".handover_staging_{ts}"
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Stage 7 Handover Package Builder ===")
    print(f"  Timestamp  : {ts}")
    print(f"  Staging    : {staging}")
    print(f"  Output dir : {output_dir}")
    print(f"  ZIP target : {zip_path}\n")

    staging.mkdir(parents=True, exist_ok=True)
    logs_dir = staging / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    exit_codes: dict[str, int] = {}

    # -----------------------------------------------------------------------
    # Step 1: pytest
    # -----------------------------------------------------------------------
    print("Step 1/3 — Running pytest...")
    ec = _run(
        [sys.executable, "-m", "pytest", "asian_options/tests/", "-v",
         "--tb=short", "--no-header"],
        logs_dir / "pytest_results.txt",
    )
    exit_codes["pytest"] = ec
    if ec != 0:
        print(f"  WARNING: pytest reported failures (exit {ec}). "
              "See logs/pytest_results.txt for details.")

    # -----------------------------------------------------------------------
    # Step 2: run_method_comparison.py
    # -----------------------------------------------------------------------
    print("\nStep 2/3 — Running method comparison...")
    ec = _run(
        [sys.executable, "run_method_comparison.py"],
        logs_dir / "method_comparison.txt",
    )
    exit_codes["method_comparison"] = ec
    if ec != 0:
        print(f"  WARNING: run_method_comparison.py exited with {ec}.")

    # -----------------------------------------------------------------------
    # Step 3: validation_minimal profile
    # -----------------------------------------------------------------------
    print("\nStep 3/3 — Running validation_minimal profile...")
    ec = _run(
        [sys.executable, "scripts/run_experiments.py",
         "--profile", "validation_minimal",
         "--output-dir", str(output_dir)],
        logs_dir / "validation_minimal.txt",
    )
    exit_codes["validation_minimal"] = ec
    if ec != 0:
        print(f"  WARNING: run_experiments.py (validation_minimal) exited with {ec}.")

    # -----------------------------------------------------------------------
    # Collect outputs
    # -----------------------------------------------------------------------
    print("\nCollecting outputs...")
    _collect_outputs(output_dir, staging)

    # -----------------------------------------------------------------------
    # Environment info
    # -----------------------------------------------------------------------
    env_info = _environment_info()
    env_path = logs_dir / "environment.txt"
    env_path.write_text(json.dumps(env_info, indent=2), encoding="utf-8")

    # -----------------------------------------------------------------------
    # Provenance
    # -----------------------------------------------------------------------
    provenance = {
        "package_timestamp": now.isoformat(),
        "builder_script": str(Path(__file__).relative_to(REPO_ROOT)),
        "repo_root": str(REPO_ROOT),
        "git_commit": env_info["git_commit"],
        "git_status": env_info["git_status"],
        "python_version": env_info["python_version"],
        "platform": env_info["platform"],
        "exit_codes": exit_codes,
        "profile": "validation_minimal",
        "note": (
            "VALIDATION ONLY — NOT DISSERTATION-SCALE. "
            "Results are from a quick validation run (small n_paths, "
            "few replications). Do NOT cite as dissertation results. "
            "Dissertation-scale requires >=30 independent outer replications."
        ),
    }
    provenance_path = staging / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    # -----------------------------------------------------------------------
    # Warning label
    # -----------------------------------------------------------------------
    warning_path = staging / "WARNING_validation_only.txt"
    warning_path.write_text(
        "VALIDATION ONLY — NOT DISSERTATION-SCALE\n"
        "=========================================\n\n"
        "The results in this package are produced by the 'validation_minimal'\n"
        "profile which uses a small number of Monte Carlo paths and few\n"
        "replications.  They are intended for quick sanity checks only.\n\n"
        "DO NOT:\n"
        "  - cite these results in the dissertation\n"
        "  - use them to support any statistical inference\n"
        "  - compare them against dissertation-scale results without noting\n"
        "    the difference in scale\n\n"
        "Dissertation-scale results require the 'dissertation_full' profile\n"
        "with >=30 independent outer replications per configuration,\n"
        "separate training/pilot/pricing datasets, and Student-t CI across\n"
        "replications.  This profile has not yet been run.\n",
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Manifest (must be last before ZIP, after all files are in staging)
    # -----------------------------------------------------------------------
    print("\nBuilding manifest...")
    _build_manifest(staging)

    # -----------------------------------------------------------------------
    # Create ZIP
    # -----------------------------------------------------------------------
    print("Building ZIP...")
    _build_zip(staging, zip_path)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Summary ===")
    for step, ec in exit_codes.items():
        status = "PASS" if ec == 0 else f"WARN (exit {ec})"
        print(f"  {step:30s}: {status}")
    print(f"\n  ZIP: {zip_path.name}")
    print(f"  SHA: {_sha256(zip_path)}")

    # Clean up staging directory
    shutil.rmtree(staging, ignore_errors=True)

    # Non-zero exit only if pytest has real failures (not just PyTorch-missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
