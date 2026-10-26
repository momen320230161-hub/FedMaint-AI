# V3 technical audit, implementation and acceptance

Executed 2026-10-26. Final status: implemented and tested within Phase 01/02/02.1. No Phase 03 implementation, remote push, Git history rewrite, global environment/security change or backup mutation.

## 1. Initial audit

- Reusable: exact17-file integrity audit and manifest variants; Azure schema validation, causal feature windows and fully purged target support; NASA source-scoped engine IDs, official offsets, engine splitting and chronological masks; lazy typed FastAPI metadata endpoints.
- Phase 01 private constant-column removal yielded different NASA dimensions per factory. Phase 02 added a union adapter, while Phase 02.1 separately used 24 fixed coordinates. V3 preserves public dimensions and neutralizes private constant coordinates.
- sklearn and optimization packages duplicated models, fit loops, selection, budgets, checkpointing and runner configuration. V3 replaces them with a single PyTorch trainer and artifact contract. Classification metrics still use sklearn; no sklearn estimator training remains.
- Historical Azure event-enriched scores lacked evidence of posting-time availability. The primary baseline is telemetry only; causal event-window ablations remain opt-in research, without fabricated timestamps or causal assertions.
- Audited Phase 02.1 GRU padding logic and N2 fallback were reusable. N2 still has unsupported-regime/source-scale and coordinate-meaning limitations. Prior held-out exposure cannot be erased by a new freeze. New fit guards additionally reject same-engine train/validation leakage at different cycles.

## 2. Architecture and decisions

The design was written before implementation in [ARCHITECTURE_V3.md](ARCHITECTURE_V3.md), including Mermaid data/privacy diagrams. Shared raw validators and partition rules feed private source-only normalization and fixed tabular/masked batches. One trainer selects checkpoints/thresholds on validation only; safe numeric bundles feed offline evaluation and internal API inference. Neural tensor export/import, shape/name/dtype contracts, configurable local epochs and sample counts are implemented. No aggregator, proximal loss, DP or secure aggregation is implemented.

Current protocols: [V3 data](V3_DATA_PROTOCOL.md) and [V3 experiments](V3_EXPERIMENT_PROTOCOL.md). Target definitions, time cuts, engine split seed 42/fraction.2 and FD004/model4 holdouts are unchanged. Azure telemetry baseline, fixed coordinates and the unified model system are authorized rebuild decisions. NASA prefix selection preserves Phase 02.1 policy; historical Phase 02 window-validation scores remain a separate protocol.

## 3. Implementation inventory

15 created, 28 modified and 24 removed files relative to initial HEAD (working-tree status, before commit). Reusable optimization primitives were consolidated under data/models; this inventory counts Git delete/add paths, not semantic module lineage.

Created:

- `configs/v3.json`
- `configs/v3_data.json`
- `docs/ARCHITECTURE_V3.md`
- `docs/V3_ACCEPTANCE.md`
- `docs/V3_DATA_PROTOCOL.md`
- `docs/V3_EXPERIMENT_PROTOCOL.md`
- `docs/V3_RESULTS.md`
- `requirements-v3.lock`
- `scripts/run_v3.py`
- `src/fedmaint/common.py`
- `src/fedmaint/data/batches.py`
- `src/fedmaint/data/clients.py`
- `src/fedmaint/data/nasa_clients.py`
- `src/fedmaint/data/normalization.py`
- `tests/test_v3.py`

Modified:

- `README.md`
- `docs/PHASE01_ACCEPTANCE.md`
- `docs/PHASE01_PROTOCOL.md`
- `docs/PHASE021_ACCEPTANCE.md`
- `docs/PHASE021_RESULTS.md`
- `docs/PHASE02_1_RESEARCH_PLAN.md`
- `docs/PHASE02_ACCEPTANCE.md`
- `docs/PHASE02_PROTOCOL.md`
- `docs/PHASE02_RESULTS.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/ROADMAP.md`
- `pyproject.toml`
- `scripts/prepare_data.py`
- `src/fedmaint/api/artifacts.py`
- `src/fedmaint/api/main.py`
- `src/fedmaint/data/contracts.py`
- `src/fedmaint/data/preprocessing.py`
- `src/fedmaint/models/artifacts.py`
- `src/fedmaint/models/contracts.py`
- `src/fedmaint/models/experiments.py`
- `src/fedmaint/models/metrics.py`
- `src/fedmaint/models/network.py`
- `src/fedmaint/models/training.py`
- `tests/conftest.py`
- `tests/test_integrity.py`
- `tests/test_model_metrics.py`
- `tests/test_preprocessing.py`
- `tests/test_real_pipeline.py`

Removed as superseded implementation/config/tests:

- `configs/phase01.json`
- `configs/phase02.json`
- `configs/phase021.json`
- `scripts/audit_phase021_availability.py`
- `scripts/phase021.py`
- `scripts/report_baselines.py`
- `scripts/report_phase021.py`
- `scripts/run_baselines.py`
- `scripts/verify_baselines.py`
- `scripts/verify_phase021.py`
- `src/fedmaint/models/data.py`
- `src/fedmaint/optimization/__init__.py`
- `src/fedmaint/optimization/audit.py`
- `src/fedmaint/optimization/azure.py`
- `src/fedmaint/optimization/common.py`
- `src/fedmaint/optimization/nasa.py`
- `src/fedmaint/optimization/normalization.py`
- `src/fedmaint/optimization/selection.py`
- `src/fedmaint/optimization/torch_models.py`
- `src/fedmaint/optimization/training.py`
- `src/fedmaint/optimization/workflow.py`
- `tests/test_model_contracts.py`
- `tests/test_model_training.py`
- `tests/test_optimization.py`

Historical reports retain their findings beneath a provenance banner; an obsolete configuration link points to its original Git revision. Historical locks, datasets, original PDF, existing ignored outputs and Git history remain intact; old code/configs can be inspected at the original revision. No promise of backward-compatible loading of historical sklearn bundles is made.

## 4. Data integrity

All 17 raw manifest files passed exact accepted SHA-256 and row-count checks before/after real experiments and in the full test suite. Independent before/after snapshot: **742 files checked; 0 changed**. Includes the original academic PDF, manifest, raw data/source documentation and historical Phase 01/02/02.1 outputs. Evidence: `outputs/v3-preservation/before.json`, `after.json`, `remote.json` and final-review.json. External backups/legacy directories were never modified.

## 5. Models

| Track | Shared input | Model | Parameters | Neural float32 bytes |
| --- | --- | --- | --- | --- |
| Azure primary |16 telemetry summaries; private constants neutral | MLP16->32 ReLU->1 logit |577 |2308 |
| NASA candidate |30x24 plus explicit masks | GRU24->32, scalar linear head | 5601 |22404 |
| NASA matched control |720 observed coordinates+30 masks | MLP750->32 ReLU->1 |24065 |96260 |

Both references use the same architecture per task, seeded Adam, maximum 40 epochs/patience 6, batch 128 and recorded loss histories. Binary imbalance, validation-only thresholds, uncapped RUL and raw output policy are explicit. GPU is supported by configuration but untested; the available build is CPU. Parameter bytes are not measured network communication.

## 6. Real experiments

Accepted run: `outputs/v3/reference-final/`. **24 local/centralized fits and 76 evaluation files**, seeds 42/43. Each track has 3 participating source factories and a strict held-out group. All metrics, sample/engine memberships, selected epochs, routing sources and unsupported-regime diagnostics appear in [V3_RESULTS.md](V3_RESULTS.md).

Selected examples (both seeds, no rounding-based model selection): pooled NASA GRU FD004 RMSE 33.67810/33.10999 cycles, MAE 25.21372/25.59112; pooled masked MLP RMSE 47.44544/53.71024. Pooled Azure model4 AP 0.16688/0.17587, PR-AUC 0.16405/0.17329, F1 0.26074/0.26136 versus no-skill AP 0.01459. Local and per-factory official-test results are fully reported, including severe source failures.

The first development panel completed but independent inference differed by up to 5.96e-8 from CPU thread defaults. A real inference-policy bug was fixed and regression-tested; no tolerances were relaxed. The full unchanged panel was rerun into a fresh directory. Both development and accepted directories are retained. A premature verify command before final results.json existed failed harmlessly and was rerun after completion.

## 7. Validation commands and outcomes

Environment: Windows, CPython 3.12.10, torch 2.6.0+cpu, NumPy 2.2.6, pandas 2.2.3, scikit-learn 1.6.1, FastAPI 0.115.8. Exact dependencies: requirements-v3.lock; model plan records versions. Reused the project-local .venv-phase021 environment; no global install.

From repository root, successful final gates:

```powershell
.venv-phase021/Scripts/python.exe -W error scripts/run_v3.py run --output outputs/v3/reference-final
.venv-phase021/Scripts/python.exe -W error scripts/run_v3.py verify --output outputs/v3/reference-final
.venv-phase021/Scripts/python.exe scripts/prepare_data.py
.venv-phase021/Scripts/python.exe -m pytest -q -o cache_dir=outputs/pytest-v3-accepted-cache --basetemp outputs/pytest-v3-accepted --tb=short --junitxml=outputs/v3/tests-final.xml
.venv-phase021/Scripts/python.exe scripts/smoke_backend.py
.venv-phase021/Scripts/python.exe -m compileall -q src scripts
.venv-phase021/Scripts/python.exe -m pip check
git diff --check
```

- **80 passed,0 failed** in 40.43 s; earlier76-test suite also passed before4 added regressions. Tests cover original input hashes, schemas, label boundaries, causal features, temporal/engine splits, held-out gates, fixed dimensions, local fits, constant/missing sensors, N2 unseen-condition fallback, masks, deterministic initialization/training, selection boundaries, same-engine leakage, safe weights/private hashes/path rejection, restored predictions, metrics, real pipelines and API metadata/errors.
- **24 model bundles /76 prediction files** independently reloaded: predictions and all metrics recomputed exactly under the recorded single-thread CPU policy. No allclose relaxation was used in final independent verification.
- Real metadata preparation completed, publishing 8 factories and 17 verified source records. Two full test preparations were reproducible. A live Uvicorn process passed HTTP health/datasets/factories/held-out-summary checks and was terminated by the smoke helper.
- Compileall, pip check and diff whitespace check passed. Warnings remain errors. Initial sandbox pytest attempts failed Windows directory/cache permissions; normal host test execution succeeded without modifying security settings.

Numeric inference reproducibility assumes the recorded CPU versions, thread policy and batch policy. GPU/different hardware/platform bitwise equality is unverified. The complete24-fit panel was repeated for a bug fix, not statistical replication; synthetic training reproducibility is tested explicitly. The final accepted metrics are not a significance claim.

## 8. Scientific limitations

Held-out results are retrospective. Synthetic factories confound domain/fault/operating-regime differences. Two optimization seeds do not provide significance testing. Unsupported N2 regimes use source-global robust fallback and remain out-of-support; local scalings differ semantically. Event posting availability and end-of-telemetry failure surveillance are unverified assumptions. Negative raw NASA predictions remain visible, and run-to-failure windows are correlated. Centralization is non-private; model weights can leak and private scaler export is not an approved FL deployment message.

## 9. Repository status

Branch **main**, unchanged HEAD `f6a4e8e4cdc276c65f76a42286428626beb2c0f2`. Initial working tree was clean. GitHub connector fetch_commit(main) independently confirmed the same remote HEAD after git transport failed. All V3 changes remain **uncommitted and unstaged** for review. No branch was created, no push performed, no history rewritten. Generated experiments/test scratch are ignored. The final inventory and per-file hashes are recorded under ignored preservation evidence.

## 10. Phase 03 readiness

Completed/tested: separate data tracks, immutable inputs, source-only fit contracts, fixed shared dimensions, Azure MLP/NASA GRU and MLP control, deterministic local/pooled training, model state import/export and parameter/sample accounting, saved raw inference and FastAPI foundation. The code is mechanically ready for later federated orchestration.

Before a meaningful FedAvg/FedProx study, authorize Phase 03 and freeze aggregation participation/weighting, matched local-epoch budgets, communication measurement and the client/server threat model. Resolve/evaluate heterogeneous local feature semantics and strict zero-shot preprocessing deployment. Explicit privacy, KG/LLM and React remain future work. These are research/deployment limitations, not implemented guarantees. No remaining Phase 01/02 code/test failure is known from the executed gates.
