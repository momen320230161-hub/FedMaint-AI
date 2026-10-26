"""Provenance diagnostics and preservation checks; no held-out selection information."""

import json

import numpy as np

from fedmaint.data.preprocessing import LocalPreprocessor
from .common import DOMAINS, digest, write_json
from .nasa import NasaStore


def preserve_before(root, output):
    target = output / "preservation.json"
    if target.exists():
        return
    original = json.loads((root / "outputs/phase021-alignment/preservation-before.json").read_text())
    # The already-recorded digests protect FD004 bytes without opening them before freeze.
    original.pop(".gitignore", None)  # One new environment ignore rule is an intentional engineering change.
    original["docs/PHASE02_1_RESEARCH_PLAN.md"] = digest(root / "docs/PHASE02_1_RESEARCH_PLAN.md")
    write_json(target, {"expected": original, "deferred_heldout_hash_reads": True})


def verify_preservation(root, output):
    original = json.loads((output / "preservation.json").read_text())["expected"]
    changed = [p for p, h in original.items() if not (root / p).is_file() or digest(root / p) != h]
    if changed:
        raise ValueError(f"Original files changed: {changed}")
    result = {"verified_files": len(original), "changed": changed}
    write_json(output / "preservation-verified.json", result)
    return result


def diagnostics(root, output, config):
    preserve_before(root, output)
    store = NasaStore(root, config)
    frames, states = {}, {}
    for domain in DOMAINS:
        frames[domain], _ = store.partition(domain)
        states[domain] = LocalPreprocessor.load(root / f"outputs/phase01/nasa-{domain}-preprocessor.json")
    historical = json.loads((root / "outputs/phase021-alignment/source-diagnostics.json").read_text())
    results = []
    for source in DOMAINS:
        for target in DOMAINS:
            if source == target:
                continue
            z = np.abs(states[source].transform(frames[target]))
            values = {"source": source, "target": target, "abs_z_p99_all_coordinates": float(np.quantile(z, .99)),
                      "fraction_coordinates_above_10": float((z > 10).mean()), "largest_abs_z": float(z.max())}
            previous = next(r for r in historical["source_to_source"] if r["source"] == source and r["target"] == target)
            for name in ("abs_z_p99_all_coordinates", "fraction_coordinates_above_10", "largest_abs_z"):
                if not np.isclose(values[name], previous[name], rtol=1e-12, atol=1e-12):
                    raise ValueError("Source diagnostic provenance/reproduction mismatch")
            results.append(values)
    report = {"status": "reproduced", "access": store.access, "source_to_source": results,
              "prior_diagnostic_sha256": digest(root / "outputs/phase021-alignment/source-diagnostics.json"),
              "local_states_sha256": {d: digest(root / f"outputs/phase01/nasa-{d}-preprocessor.json") for d in DOMAINS}}
    write_json(output / "diagnostics.json", report)
    return report
