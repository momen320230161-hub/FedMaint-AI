import hashlib
import json
from pathlib import Path

import pytest

from fedmaint.data.contracts import DataError, PipelineConfig
from fedmaint.data.integrity import verify_manifest
from fedmaint.data.prepare import safe_output

ROOT = Path(__file__).resolve().parents[1]


def fixture_manifest(tmp_path):
    (tmp_path / "data/azure").mkdir(parents=True)
    (tmp_path / "data/nasa").mkdir()
    a = b'id,value\n1,2\n2,3\n'
    n = b'1 2\n3 4\n'
    (tmp_path / "data/azure/a.csv").write_bytes(a)
    (tmp_path / "data/nasa/n.txt").write_bytes(n)
    manifest = {"azure": {"repository_path": "data/azure", "files": {"a.csv": {
        "rows": 2, "sha256": "archive-byte-hash", "repository_sha256": hashlib.sha256(a).hexdigest()}}},
        "nasa": {"repository_path": "data/nasa", "files": {"n.txt": {"rows": 2, "sha256": hashlib.sha256(n).hexdigest()}}}}
    (tmp_path / "data/manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_accepted_repository_hashes_and_logical_row_counts(tmp_path):
    fixture_manifest(tmp_path)
    assert [r["rows"] for r in verify_manifest(tmp_path)] == [2, 2]


@pytest.mark.parametrize("defect", ["hash", "row_count", "missing"])
def test_manifest_failure_does_not_rewrite_manifest(tmp_path, defect):
    manifest = fixture_manifest(tmp_path)
    path = tmp_path / "data/manifest.json"
    raw = tmp_path / "data/nasa/n.txt"
    if defect == "hash":
        raw.write_bytes(b"3 4\n1 2\n")
    elif defect == "row_count":
        manifest["nasa"]["files"]["n.txt"]["rows"] = 3
        path.write_text(json.dumps(manifest))
    else:
        raw.unlink()
    before = path.read_bytes()
    with pytest.raises(DataError, match="integrity failed"):
        verify_manifest(tmp_path)
    assert path.read_bytes() == before


def test_real_inputs_match_all_17_manifest_entries():
    records = verify_manifest(ROOT)
    assert len(records) == 17
    assert sum(r["rows"] for r in records if r["dataset"] == "azure") == 884166
    assert sum(r["rows"] for r in records if r["dataset"] == "nasa") == 265963


@pytest.mark.parametrize("path", ["data/processed", "data/Azure Data", "data/CMAPSSData", "docs", "src", ".git", "."])
def test_output_paths_protect_raw_inputs_and_source(path):
    with pytest.raises(DataError):
        safe_output(ROOT, ROOT / path)


def test_config_rejects_ambiguous_boundaries_or_invalid_defaults():
    from fedmaint.data.contracts import AzureConfig
    with pytest.raises(ValueError):
        AzureConfig(train_end="2015-12-01", validation_end="2015-08-01")
    with pytest.raises(ValueError):
        AzureConfig(history_hours=0)
    with pytest.raises(ValueError):
        AzureConfig(feature_groups=["unknown"])
    assert PipelineConfig.load(ROOT / "configs/v3_data.json").seed == 42
