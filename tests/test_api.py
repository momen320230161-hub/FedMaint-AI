import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from fedmaint.api.main import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_real_metadata_endpoints_and_structured_404(prepared):
    output, _ = prepared
    with TestClient(create_app(ROOT, output / "metadata.json")) as client:
        assert client.get("/health").json() == {"status": "ok", "metadata_ready": True}
        datasets = client.get("/api/datasets")
        assert datasets.status_code == 200
        assert sum(len(d["files"]) for d in datasets.json()) == 17
        assert len(client.get("/api/factories").json()) == 8
        factory = client.get("/api/factories/azure-model4/summary").json()
        assert factory["units"] == 32
        assert factory["role"] == "held_out"
        unknown = client.get("/api/factories/unknown/summary")
        assert unknown.status_code == 404
        assert unknown.json()["error"]["code"] == "not_found"
        assert client.get("/nonexistent").json()["error"]["code"] == "not_found"
        assert client.post("/api/datasets").status_code == 405
        routes = client.get("/openapi.json").json()["paths"]
        assert set(routes) == {"/health", "/api/datasets", "/api/factories", "/api/factories/{factory_id}/summary"}


@pytest.mark.parametrize("defect", ["missing", "invalid_json", "manifest_hash", "source_identity", "missing_files"])
def test_unavailable_or_stale_metadata_returns_503(prepared, tmp_path, defect):
    output, _ = prepared
    path = tmp_path / "metadata.json"
    content = json.loads((output / "metadata.json").read_text())
    if defect == "invalid_json":
        path.write_text("{")
    elif defect != "missing":
        if defect == "manifest_hash":
            content["manifest_sha256"] = "0" * 64
        elif defect == "source_identity":
            content["datasets"][0]["files"][0]["mtime_ns"] = 0
        else:
            content["datasets"][0]["files"] = []
        path.write_text(json.dumps(content))
    with TestClient(create_app(ROOT, path)) as client:
        assert client.get("/health").json()["metadata_ready"] is False
        response = client.get("/api/datasets")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "metadata_unavailable"


def test_import_and_app_creation_do_not_load_raw_telemetry(monkeypatch, tmp_path):
    import pandas as pd
    def forbidden(*args, **kwargs):
        raise AssertionError("API must not read raw tabular data")
    monkeypatch.setattr(pd, "read_csv", forbidden)
    with TestClient(create_app(ROOT, tmp_path / "missing.json")) as client:
        assert client.get("/health").status_code == 200
