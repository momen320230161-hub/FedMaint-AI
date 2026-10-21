# FedKG-Maint

Privacy-preserving predictive maintenance using federated learning, factory-local knowledge graphs, and evidence-grounded LLM explanations and maintenance recommendations.

Phase 01 supplies validated Azure and NASA pipelines, reproducible simulated factories, client-local preprocessing, and a read-only FastAPI metadata backend. Predictive models, performance results, federation, privacy mechanisms, KGs, explanations, and the UI belong to later phases. Previous implementation and research artifacts remain in the separately verified Phase 00 backup and the separately preserved legacy repository.

Read the [original academic brief](docs/Project2_Description.pdf), [project overview](docs/PROJECT_OVERVIEW.md), [development roadmap](docs/ROADMAP.md), and [agent workflow](AGENTS.md).

The retained inputs are:

- [Microsoft Azure Predictive Maintenance](data/Azure%20Data/): five original CSV files.
- [NASA C-MAPSS](data/CMAPSSData/): twelve numeric data files and its original source documentation.
- [Verified manifest](data/manifest.json): accepted exact SHA-256 hashes and row counts. Preserve these files byte-for-byte, including existing line endings; do not rewrite hashes to accept changed data.

Development proceeds through data/backend, baselines, federated learning, explicit privacy, local KGs/LLM explanations, and scientific/backend validation. React implementation is exclusively Phase 07, after the backend and models pass Phase 06 acceptance. The final deliverable is a complete full-stack project.

Codex directly implements, reviews, and tests this project. Use the existing main branch. See the [Phase 01 data contracts](docs/PHASE01_PROTOCOL.md) and [acceptance report](docs/PHASE01_ACCEPTANCE.md) for definitions, limitations, and executed checks.

## Reproduce Phase 01

Run from this repository in PowerShell with CPython 3.12 (validated on Windows, Python 3.12.10). The exact environment is pinned in `requirements-lock.txt`; `pyproject.toml` declares package and direct dependencies. No global settings or global package installs are needed.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/verify_data.py
.\.venv\Scripts\python.exe scripts/prepare_data.py --config configs/phase01.json
.\.venv\Scripts\python.exe -m pytest -q --basetemp outputs/pytest-phase01
.\.venv\Scripts\python.exe scripts/smoke_backend.py
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Preparation verifies all 17 manifest entries, validates schemas, creates real Azure tabular samples and NASA masked sequences, fits six training-client preprocessors, and publishes `outputs/phase01/metadata.json` last. Generated files are ignored by Git and never written under `data/`. Tests include two full real-data preparation runs, so allow time and disk space for those outputs. `--basetemp` is disposable pytest scratch space, recreated by pytest on subsequent runs; do not point it at valuable files.

For a changed configuration use a separate output directory:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py --config configs/phase01.json --output outputs/experiment-01
```

Generated NPZ files load with `numpy.load(path, allow_pickle=False)`. Their feature order, source/split names, transformed status, and numeric-array SHA-256 digests appear in the catalog. Preprocessor JSON files are reloadable with `LocalPreprocessor.load(path)`. Held-out samples are explicitly **untransformed**; do not fit a held-out scaler. Later zero-shot experiments must choose and freeze a training-client transformation using training/validation evidence only.

## Start the backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn fedmaint.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for typed contracts, or request:

- `GET /health`: liveness plus `metadata_ready`.
- `GET /api/datasets`: verified source hashes and row counts for both task definitions.
- `GET /api/factories`: eight real simulated-factory summaries.
- `GET /api/factories/azure-model4/summary` (or another listed ID): one factory's provenance and descriptive statistics.

The app starts without reading telemetry. Missing, malformed, or stale metadata yields a structured 503 on dataset/factory endpoints; liveness remains 200 with `metadata_ready: false`. Unknown factories return structured 404 errors. No training/prediction endpoints exist. The catalog checks the manifest hash and each source's size/mtime on requests; this is a lightweight freshness check, not a substitute for the full hash audit. Re-run preparation after moving or touching raw files.

Optional process-local variables are `FEDMAINT_ROOT` (repository root) and `FEDMAINT_METADATA` (catalog path, useful for a different output directory). They do not change global Codex settings.

## Project layout

| Location | Responsibility |
| --- | --- |
| `src/fedmaint/data/` | Config validation, hash/row audit, schema loaders, targets/windows, partitions, local transformations, preparation |
| `src/fedmaint/api/` | Pydantic response contracts and lazy metadata endpoints |
| `configs/phase01.json` | Frozen initial window/horizon/split/holdout defaults |
| `scripts/` | Integrity audit, real-data preparation, live HTTP smoke test |
| `tests/` | Synthetic adversarial tests and full real-data/API/reproducibility checks |
| `outputs/` | Ignored generated samples, preprocessing states, catalog, pytest scratch |

The [protocol](docs/PHASE01_PROTOCOL.md) documents the scientific assumptions that must stay fixed before later model comparisons. [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) supports split-before-fit preprocessing, and [FastAPI's testing documentation](https://fastapi.tiangolo.com/tutorial/testing/) describes the HTTP test client used here.
