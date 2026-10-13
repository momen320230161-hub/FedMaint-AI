#!/usr/bin/env python3
"""Validate and locally prepare the team's Azure and NASA dataset ZIP archives."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.json"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def row_count(path: Path, dataset: str) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        if dataset == "azure":
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for line in reader if line)
        return sum(1 for line in handle if line.strip())

def extract_zip(archive: Path, destination: Path, expected: dict):
    if not archive.is_file():
        raise FileNotFoundError(archive)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad:
            raise ValueError(f"Corrupted ZIP member: {bad}")
        entries = {Path(item.filename).name: item for item in zf.infolist() if not item.is_dir()}
        for name in expected:
            if name not in entries:
                raise ValueError(f"{archive.name}: missing {name}")
            with zf.open(entries[name]) as src, (destination / name).open("wb") as dst:
                while block := src.read(1024 * 1024):
                    dst.write(block)
            print(f"Extracted {destination.name}/{name}")

def validate(destination: Path, expected: dict, dataset: str) -> bool:
    success = True
    for name, details in expected.items():
        path = destination / name
        if not path.is_file():
            print(f"MISSING {dataset}/{name}", file=sys.stderr)
            success = False
            continue
        n = row_count(path, dataset)
        digest = sha256(path)
        ok = n == details["rows"] and digest == details["sha256"]
        print(f'{"OK" if ok else "MISMATCH"} {dataset}/{name}: rows={n}, sha256={digest}')
        success = success and ok
    return success

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--azure-zip", type=Path)
    parser.add_argument("--nasa-zip", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not (args.azure_zip or args.nasa_zip or args.verify_only):
        parser.error("Provide a ZIP path or --verify-only.")
    meta = json.loads(MANIFEST.read_text(encoding="utf-8"))
    supplied = {"azure": args.azure_zip, "nasa": args.nasa_zip}
    selected = ("azure", "nasa") if args.verify_only else tuple(
        name for name, file in supplied.items() if file is not None
    )
    if not args.verify_only:
        for dataset in selected:
            archive = supplied[dataset]
            if not archive.is_file():
                raise FileNotFoundError(archive)
            if sha256(archive) != meta[dataset]["archive_sha256"]:
                raise ValueError(
                    f"{dataset}: archive SHA-256 does not match the supplied reference ZIP. "
                    "Check the source/version before editing the manifest."
                )
            extract_zip(archive, ROOT / "data" / "raw" / dataset, meta[dataset]["files"])
    ok = all(validate(ROOT / "data" / "raw" / dataset, meta[dataset]["files"], dataset)
             for dataset in selected)
    print("Dataset validation passed." if ok else "Dataset validation failed.")
    return 0 if ok else 1

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
