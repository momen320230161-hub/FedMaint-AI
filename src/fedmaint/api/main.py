"""Lazy catalog reads: no raw telemetry is loaded at import or server startup."""

import os
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from fedmaint.data.contracts import DataError
from fedmaint.data.integrity import sha256
from .schemas import Catalog, Dataset, ErrorResponse, Factory, Health


def read_catalog(root: Path, metadata: Path) -> Catalog:
    try:
        catalog = Catalog.model_validate_json(metadata.read_text(encoding="utf-8"))
        if sha256(root / "data/manifest.json") != catalog.manifest_sha256:
            raise DataError("Manifest changed")
        manifest = json.loads((root / "data/manifest.json").read_text())
        if sorted(d.dataset_id for d in catalog.datasets) != ["azure", "nasa"]:
            raise DataError("Incomplete dataset catalog")
        for dataset in catalog.datasets:
            section = manifest[dataset.dataset_id]
            if sorted(r.name for r in dataset.files) != sorted(section["files"]):
                raise DataError("Incomplete source catalog")
            for record in dataset.files:
                expected = section["files"][record.name]
                if (record.dataset != dataset.dataset_id or record.rows != expected["rows"]
                        or record.sha256 not in {expected["sha256"], expected.get("repository_sha256", expected["sha256"])}
                        or Path(record.path).as_posix() != f"{section['repository_path']}/{record.name}"):
                    raise DataError("Catalog differs from accepted manifest")
                path = (root / record.path).resolve()
                if not path.is_relative_to((root / "data").resolve()):
                    raise DataError("Invalid catalog source path")
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != (record.bytes, record.mtime_ns):
                    raise DataError("Raw source identity changed")
        return catalog
    except (OSError, ValueError, ValidationError) as exc:
        raise DataError("Validated metadata unavailable or stale; run scripts/prepare_data.py") from exc


def create_app(root: Path | None = None, metadata: Path | None = None) -> FastAPI:
    project_root = (root or Path(os.environ.get("FEDMAINT_ROOT", Path(__file__).resolve().parents[3]))).resolve()
    catalog_path = metadata or Path(os.environ.get("FEDMAINT_METADATA", project_root / "outputs/phase01/metadata.json"))
    app = FastAPI(title="FedKG-Maint data foundation", version="0.1.0")

    def catalog():
        return read_catalog(project_root, catalog_path)

    @app.exception_handler(DataError)
    async def data_error(request: Request, exc: DataError):
        return JSONResponse(status_code=503, content={"error": {"code": "metadata_unavailable", "message": str(exc)}})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code,
                            content={"error": {"code": "not_found" if exc.status_code == 404 else "http_error",
                                               "message": str(exc.detail)}}, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request", "message": "Invalid request parameters"}})

    errors = {503: {"model": ErrorResponse}, 404: {"model": ErrorResponse}}

    @app.get("/health", response_model=Health)
    def health():
        try:
            catalog()
            ready = True
        except DataError:
            ready = False
        return Health(metadata_ready=ready)

    @app.get("/api/datasets", response_model=list[Dataset], responses=errors)
    def datasets():
        return catalog().datasets

    @app.get("/api/factories", response_model=list[Factory], responses=errors)
    def factories():
        return catalog().factories

    @app.get("/api/factories/{factory_id}/summary", response_model=Factory, responses=errors)
    def factory_summary(factory_id: str):
        for factory in catalog().factories:
            if factory.factory_id == factory_id:
                return factory
        raise HTTPException(404, "Unknown factory")

    return app


app = create_app()
