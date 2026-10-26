# FedMaint-AI

A predictive-maintenance research foundation for Azure failure classification and
NASA C-MAPSS RUL regression. Architecture V3 consolidates Phase 01/02/02.1 into
one PyTorch training/inference system. FedAvg/FedProx, explicit privacy, local KGs,
LLM explanations and React remain later phases; Phase 03 requires authorization.

Read [Architecture V3](docs/ARCHITECTURE_V3.md), [current data protocol](docs/V3_DATA_PROTOCOL.md),
[experiment protocol](docs/V3_EXPERIMENT_PROTOCOL.md), [measured results](docs/V3_RESULTS.md),
[acceptance evidence](docs/V3_ACCEPTANCE.md), [project overview](docs/PROJECT_OVERVIEW.md)
and [roadmap](docs/ROADMAP.md).

## Reproduce

Use CPython 3.12 and a project-local environment. Existing `.venv-phase021` has the
pinned CPU dependencies and can be reused. For a fresh local environment:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-v3.lock
.venv/Scripts/python.exe -m pip install --no-build-isolation --no-deps -e .
.venv/Scripts/python.exe -m pip check
```

Commands below use the existing validated environment. Run from this repository.
Choose a **fresh** experiment directory; accepted outputs are never overwritten.
Run metadata/full-suite tests after experimental freeze, since they inspect holdouts.

```powershell
.venv-phase021/Scripts/python.exe scripts/verify_data.py
.venv-phase021/Scripts/python.exe -W error scripts/run_v3.py run --output outputs/v3/reference-final
.venv-phase021/Scripts/python.exe -W error scripts/run_v3.py verify --output outputs/v3/reference-final
.venv-phase021/Scripts/python.exe scripts/prepare_data.py
.venv-phase021/Scripts/python.exe -m pytest -q -o cache_dir=outputs/pytest-v3-cache --basetemp outputs/pytest-v3-final
.venv-phase021/Scripts/python.exe scripts/smoke_backend.py
.venv-phase021/Scripts/python.exe -m compileall -q src scripts
```

On this Windows host pytest temporary-directory permission operations require a
normal host process rather than the restricted agent sandbox. Warnings remain
errors; no warning filters are weakened. `--basetemp` is disposable test scratch.

`configs/v3.json` freezes all experiment settings. Azure uses 16 telemetry summary
coordinates, MLP(16,32,1), BCE/logits, validation AP selection and validation F1
threshold selection. NASA uses the fixed sorted 24-coordinate schema, train-only
N2 candidate normalization, GRU(24,32,1) plus masked MLP control, uncapped RUL,
raw linear predictions and macro cycle 50/100 validation RMSE. Both local-only
and centralized references use identical per-task models and budgets. N0/N1 are
supported alternatives; changing candidates requires a fresh declared experiment,
not selecting a strategy after looking at FD004. Prior held-out exposure means
V3 held-out results are retrospective.

All raw data, the [verified manifest](data/manifest.json), original academic PDF,
and historical ignored outputs are preserved. Historical Phase 01/02/02.1 reports
remain records of their original code at commit `f6a4e8e4cdc276c65f76a42286428626beb2c0f2`.
Their commands/configs describe that revision; they are not current entry points
or new V3 measurements. Current dependencies use `requirements-v3.lock`; earlier
locks remain provenance. No sklearn estimators remain in the training path;
scikit-learn is retained for well-defined classification metrics.

## Backend and inference

```powershell
.venv-phase021/Scripts/python.exe -m uvicorn fedmaint.api.main:app --host 127.0.0.1 --port 8000
```

`GET /health`, `/api/datasets`, `/api/factories`, and
`/api/factories/{factory_id}/summary` use validated metadata at
`outputs/v3/data/metadata.json`. Missing/stale metadata gives structured errors.
`FEDMAINT_ROOT` and `FEDMAINT_METADATA` are optional process-local overrides.
No predictor HTTP endpoint or frontend scaffold is introduced.

```python
from pathlib import Path
from fedmaint.api.artifacts import load_model_artifact
model = load_model_artifact(
    Path("outputs/v3/reference-final/nasa-gru-pooled-seed 42"), expected_dataset="nasa"
)
# raw_windows: [N,30,24], feature_names in model.metadata["features"] order
# masks: [N,30] boolean, chronological with left padding
prediction = model.predict_raw(raw_windows, feature_names, masks)
```

Artifacts contain numeric NPZ weights, a model contract, training history,
separately hashed **private** normalizers, and auditable predictions. Loading uses
`allow_pickle=False`, validates names/shapes/dtypes/finiteness and refuses escaping
paths or hash mismatches. Bundles are trusted offline research artifacts. Only
`export_state`/`import_state` neural tensors are potential future FL messages;
private scalers and raw evaluation NPZs must not leave a client in an FL deployment.
Central pooling and weight exchange alone provide no privacy guarantee.

## Layout

| Path | Responsibility |
| --- | --- |
| `src/fedmaint/data` | Integrity, raw schemas, labels/splits, private normalization, fixed batches/windows, offline metadata preparation |
| `src/fedmaint/models` | PyTorch MLP/GRU, one trainer, validation metrics, safe artifacts, frozen experiment runner |
| `src/fedmaint/api` | Metadata API and validated internal inference adapter |
| `configs/v3.json` | Frozen local/centralized experiment |
| `configs/v3_data.json` | Offline metadata preparation with the same data protocol |
| `outputs/v3` | Ignored V3 models, predictions, freezes, metrics and metadata |
| `tests` | Adversarial data/model tests, real preparation, inference and API checks |

Local coordinate meanings still differ across factories. Unknown N2 regimes fall
back to the source-global robust state and are reported. A strict zero-shot serving
policy compatible with private preprocessing remains a deployment research issue.
