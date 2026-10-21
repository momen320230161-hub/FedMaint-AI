# Phase 01 acceptance evidence

Date: 2026-10-21. Repository: `FedMaint-AI`, existing `main` branch. Status: **complete locally**. Codex implemented, reviewed, and tested directly. No external implementer, branch creation, commit, push, global configuration change, or Phase 02 work occurred.

## Delivered architecture

`configs/phase01.json` freezes initial experimental defaults. The `fedmaint.data` package reads and verifies the retained inputs, validates each source schema, constructs separate Azure classification/NASA RUL samples, enforces domain/temporal/engine splits, and fits only permitted client-local preprocessing. A preparation command writes 21 sample artifacts, six reusable preprocessing JSON states, and a typed provenance catalog under ignored `outputs/phase01/`. The FastAPI package serves that catalog lazily through the four requested endpoint paths, with structured errors. It does not load telemetry at startup or implement model/prediction endpoints.

Default preparation produced **51,132 real samples**: 36,128 Azure and 15,004 NASA, including explicitly separated evaluation pools. Two full preparation runs in pytest produced identical catalogs (including numeric-array digests, configuration/environment, split membership, and source identities) and byte-identical fitted states. NPZ ZIP-container bytes themselves are not used as the reproducibility metric.

## Dataset validation

All **17/17** numeric source files passed exact SHA-256 and logical row-count checks. All parsed Azure/NASA schemas passed. Azure files match the manifest's original hashes, not its alternate LF repository hashes. Neither raw files nor the manifest were rewritten.

| Azure file | Rows |
| --- | ---: |
| PdM_telemetry.csv | 876,100 |
| PdM_machines.csv | 100 |
| PdM_errors.csv | 3,919 |
| PdM_maint.csv | 3,286 |
| PdM_failures.csv | 761 |
| Total | 884,166 |

| NASA domain | Train rows / engines | Official test rows / engines | RUL rows |
| --- | ---: | ---: | ---: |
| FD001 | 20,631 / 100 | 13,096 / 100 | 100 |
| FD002 | 53,759 / 260 | 33,991 / 259 | 259 |
| FD003 | 24,720 / 100 | 16,596 / 100 | 100 |
| FD004 | 61,249 / 249 | 41,214 / 248 | 248 |

NASA numeric rows total 265,963, including RUL vectors. The supplied NASA README reverses FD004 engine counts; observed IDs and official RUL-vector length establish the counts above. Source documentation remains unchanged.

`git diff --exit-code -- data docs/Project2_Description.pdf` exited zero, confirming no tracked input/manifest/PDF changes. Archive ZIP hashes were not rechecked because those archives are not retained pipeline inputs. Both numeric datasets were fully checked; no required dataset or acceptance check was skipped.

## Factory strategy and emitted sample counts

| Factory | Role | Units in source training/telemetry | Train | Validation | Temporal/official test | Held-out evaluation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| azure-model1 | Training | 16 | 3,360 | 944 | 1,456 | — |
| azure-model2 | Training | 17 | 3,570 | 1,003 | 1,547 | — |
| azure-model3 | Training | 35 | 7,350 | 2,065 | 3,185 | — |
| azure-model4 | Held out | 32 | — | — | — | 11,648 |
| nasa-FD001 | Training | 100 | 1,481 | 336 | 100 | — |
| nasa-FD002 | Training | 260 | 3,797 | 937 | 259 | — |
| nasa-FD003 | Training | 100 | 1,812 | 418 | 100 | — |
| nasa-FD004 | Held out | 249 | — | — | — | 5,516 run-to-failure + 248 official test |

Azure factories contain disjoint machine-model groups; training factories use purged chronological intervals and held-out model4 is never fitted/tuned. NASA factories are distinct domains; training/validation split whole official training engines with seed 42 and a 20% validation fraction. Official test trajectories never enter fitting. FD004's run-to-failure and official-test evaluations remain separate. The [protocol](PHASE01_PROTOCOL.md) specifies exact boundaries, masks, source-scoped engine identity, RUL/capping conventions, and fit guards.

Descriptive heterogeneity: Azure mean ages are 12.25, 12.7647, 12.0286, and 9.34375 years in model order; positive-window fractions are approximately 2.8646%, 2.5327%, 1.6429%, and 1.4595%. These describe all eligible windows, not a performance score or train-only prevalence estimate. NASA mean run-to-failure trajectory lengths are 206.31, 206.7654, 247.20, and 245.9799 cycles; mean first operating settings are approximately 0, 23.9984, 0, and 23.9998. Full sensor/setting means and split IDs are in the catalog. Holdout defaults were fixed before these descriptive audits; these summaries are not authorized model-selection input.

Training preprocessors retain 21 features for each Azure client and 17/24/18 features for FD001/FD002/FD003. Fit-row counts are 3,360/3,570/7,350 Azure samples and 16,779/43,083/20,074 unique NASA training-engine rows. Removed constant/all-missing features are recorded in each state. With the daily Azure sampling grid, maintenance recency takes a constant eligible value per component after median imputation and is removed locally; counts remain available. This cadence is configurable, not an optimized choice. No preprocessing state exists for either held-out factory.

## Exact final gate commands and results

Commands ran from the repository root in PowerShell, using project-local CPython **3.12.10**. Dependency/runtime versions are in `requirements-lock.txt` and the generated catalog.

| Command | Executed result |
| --- | --- |
| `.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt` | Passed; locked runtime/test/build dependencies installed |
| `.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .` | Passed; local editable package installed |
| `.\.venv\Scripts\python.exe -m pip check` | Passed: `No broken requirements found.` |
| `.\.venv\Scripts\python.exe -m pytest -q --basetemp outputs/pytest-phase01` | **57 passed in 31.85s; 0 failed, 0 skipped, 0 warnings** |
| `.\.venv\Scripts\python.exe scripts/verify_data.py` | Passed: 17 exact hashes/row counts |
| `.\.venv\Scripts\python.exe -W error scripts/prepare_data.py` | Passed: both complete real-data pipelines, 8 factories, 21 sample artifacts; no warnings |
| `.\.venv\Scripts\python.exe scripts/smoke_backend.py` | Passed: live Uvicorn HTTP health/readiness, dataset/factory metadata, held-out summary; owned server stopped afterward |
| `.\.venv\Scripts\python.exe -m compileall -q src scripts` | Passed |
| `.\.venv\Scripts\python.exe -c "from fedmaint.api.main import app; print('Import OK:', app.title)"` | Passed: import/app initialization |
| `git diff --check` | Passed |
| `git diff --exit-code -- data docs/Project2_Description.pdf` | Passed; original tracked inputs unchanged |
| `git branch --show-current` | `main` |

Coverage includes manifest tampering/missing files/row mismatches, Azure target endpoint semantics and censoring, timestamp gaps/duplicates, delayed error/maintenance availability, adversarial future-data mutations, feature ablations, chronological purging, NASA train/official-test RUL, left-padding/masks, capping, engine-wise validation, held-out exclusion, local-only medians/constants/scaling, serialization, seed determinism, full real-data reproducibility, typed API responses, 404/405/503 handling, and lazy startup.

Development checks initially exposed an overly restrictive verifier that accepted only the alternate Azure hashes, an invalid test-fixture dtype assignment, integer-inferred NASA normalization warnings, and an AnyIO/Starlette deprecation. The verifier now recognizes both manifest-recorded variants, numeric feature dtypes are explicit, and AnyIO is pinned compatibly. Sandbox restrictions required the package-install/full-test/server-process checks to run with approved access. One early live probe ran during catalog rebuilding and correctly observed `metadata_ready=false`; the final sequential live probe passed after preparation. No final gate failure remains. Pip printed an informational update notice; no runtime/test warnings remain.

## Start command

After the documented install and preparation steps:

```powershell
.\.venv\Scripts\python.exe -m uvicorn fedmaint.api.main:app --host 127.0.0.1 --port 8000
```

The four requested paths are `/health`, `/api/datasets`, `/api/factories`, and `/api/factories/{factory_id}/summary`. `/docs` exposes OpenAPI documentation. See [README](../README.md) for exact setup/check commands and optional process-local root/catalog settings.

## Files created or modified

Modified: `.gitignore`, `AGENTS.md`, `README.md`, `docs/PROJECT_OVERVIEW.md`, `docs/ROADMAP.md`.

Created:

- `pyproject.toml`, `requirements-lock.txt`, `configs/phase01.json`.
- `src/fedmaint/__init__.py`.
- `src/fedmaint/data/__init__.py`, `contracts.py`, `integrity.py`, `azure.py`, `nasa.py`, `preprocessing.py`, `prepare.py`.
- `src/fedmaint/api/__init__.py`, `schemas.py`, `main.py`.
- `scripts/verify_data.py`, `scripts/prepare_data.py`, `scripts/smoke_backend.py`.
- `tests/conftest.py`, `test_integrity.py`, `test_azure.py`, `test_nasa.py`, `test_preprocessing.py`, `test_real_pipeline.py`, `test_api.py`.
- `docs/PHASE01_PROTOCOL.md`, `docs/PHASE01_ACCEPTANCE.md`.

The local `.venv/`, editable-install metadata, caches, prepared samples/catalog, and pytest scratch are generated/ignored, not project source deliverables. All changes remain uncommitted on `main` for review. Legacy repositories and backups were untouched.

## Remaining limitations and acceptance blockers

No blocker prevents Phase 01 acceptance. Simulated domains are not independent real factories. Azure event availability and complete end-of-observation failure surveillance remain assumptions, and static age is disabled by default because historical availability is unknown. Defaults are not tuned/optimal. NASA local feature sets differ after legitimate train-only constant removal; a compatible zero-shot/source preprocessing strategy and federated feature contract must be frozen from permitted data in later phases. Held-out artifacts deliberately remain untransformed. Metadata freshness uses size/mtime plus manifest hash, while full tamper detection requires the integrity command. This environment was validated on Windows/Python 3.12, not every platform.

No predictive performance, privacy guarantee, federation, KG, explanation, or frontend result is claimed. Phase 02 remains unstarted and requires further user approval.
