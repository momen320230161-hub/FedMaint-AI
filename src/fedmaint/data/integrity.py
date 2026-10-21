"""Stream exact hashes and logical row counts, without modifying the manifest."""

import csv
import hashlib
import json
from pathlib import Path

from .contracts import DataError


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path) -> list[dict]:
    root = root.resolve()
    manifest = json.loads((root / "data/manifest.json").read_text())
    records, problems = [], []
    for dataset in ("azure", "nasa"):
        section = manifest[dataset]
        for name, expected in section["files"].items():
            path = (root / section["repository_path"] / name).resolve()
            if not path.is_relative_to(root / "data"):
                raise DataError("Manifest path escapes project data directory")
            try:
                before = path.stat()
                digest = sha256(path)
                with path.open(newline="", encoding="utf-8") as stream:
                    if dataset == "azure":
                        reader = csv.reader(stream)
                        next(reader)
                        rows = sum(1 for _ in reader)
                    else:
                        rows = sum(1 for line in stream if line.strip())
                after = path.stat()
                accepted = {expected["sha256"], expected.get("repository_sha256", expected["sha256"])}
                if digest not in accepted or rows != expected["rows"]:
                    problems.append(f"{name}: hash={digest}, rows={rows}; expected one of {sorted(accepted)}, {expected['rows']}")
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    problems.append(f"{name}: changed during verification")
                records.append({"dataset": dataset, "name": name, "path": str(path.relative_to(root)),
                                "rows": rows, "sha256": digest, "bytes": after.st_size,
                                "hash_variant": "original" if digest == expected["sha256"] else "repository",
                                "mtime_ns": after.st_mtime_ns})
            except (OSError, StopIteration, csv.Error) as exc:
                problems.append(f"{name}: {exc}")
    if problems:
        raise DataError("Dataset integrity failed: " + "; ".join(problems))
    return records
