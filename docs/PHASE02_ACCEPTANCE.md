> Historical research record. Commands, code paths and measurements below refer to the pre-V3 revision `f6a4e8e4cdc276c65f76a42286428626beb2c0f2`. Current implementation and protocols: [Architecture V3](ARCHITECTURE_V3.md), [data](V3_DATA_PROTOCOL.md), [experiments](V3_EXPERIMENT_PROTOCOL.md). Historical outputs remain preserved.

# Phase 02 acceptance evidence

Date: 2026-10-21. Status: **complete locally on the existing main branch**. Codex implemented, reviewed, and tested directly. No external implementer, branch creation, global configuration change, raw-data/manifest change, commit, push, or Phase 03 implementation occurred.

## Implemented models and Phase 01 audit

- Azure: L2 logistic regression and a one-hidden-layer ReLU MLP for binary failure prediction.
- NASA: a one-hidden-layer ReLU MLP using flattened 30-cycle windows and explicit masks for RUL regression.
- Local-only and pooled centralized models for every participating client/task/architecture, with seeds 42, 43, and 44: **36 completed runs**.
- Numeric model bundles with checkpoint/preprocessing SHA-256 checks, strict task/dimension/source validation, pickle-free loading, and an internal backend `load_model_artifact` interface.

Phase 01 data code, tests, input bytes, manifest, and prepared artifacts were not rewritten. An adapter constructs a training-only sorted feature union: **21 Azure features; 24 NASA features/cycle, producing 750 encoded inputs with masks**. Locally dropped coordinates remain zero. Existing client-local train-only statistics are reused. Local zero-shot inference freezes its own source transform; centralized zero-shot inference uses the maximum-training-feature-coverage rule, fixing Azure model1/NASA FD002 before evaluation. No held-out record determined the schema or transform. [PHASE02_PROTOCOL.md](PHASE02_PROTOCOL.md) specifies the full strategy and heterogeneous-scaling limitations.

## Exact training configuration and results

The frozen config is [configs/phase02.json](https://github.com/momen320230161-hub/FedMaint-AI/blob/f6a4e8e4cdc276c65f76a42286428626beb2c0f2/configs/phase02.json): hidden width 32, batch size 128, MLP epoch cap 40 per candidate, patience 6, learning rates 0.001/0.003, alpha 0.0001, logistic C 0.1/1/10, Newton–Cholesky maximum 2,000 iterations, and fixed 100-cycle RUL optimization scale. No class reweighting or resampling was used. Candidate/checkpoint selection uses validation AP (Azure) or negative uncapped validation-window RMSE (NASA). Azure thresholds use validation maximum F1 with a deterministic tie-break. All model choices finish before opening test/held-out arrays. NASA capped training `y` remains supported separately from uncapped evaluation `raw_y`; this experiment used uncapped training inputs.

All planned runs completed, with **0 failed or skipped full-budget runs**. Wall runtime was **71.34 s**, with **51.99 s** summed candidate fitting/selection. Exact selected candidates, epochs, thresholds, per-run timings, every metric, and mean ± sample SD are in [PHASE02_RESULTS.md](PHASE02_RESULTS.md), generated directly from the saved plan. Local/centralized models use matched per-record exposure/selection budgets; centralized models visit more total records and are not claimed to have identical FLOPs or wall time.

Selected measured results below are arithmetic means across three seeds, not confidence intervals:

| Task / evaluation | Local-only source | Local result | Centralized result |
| --- | --- | --- | --- |
| Azure MLP, model1 temporal test | model1 | PR-AUC 0.9765; F1 0.9315 | PR-AUC 0.9938; F1 0.9762 |
| Azure MLP, model2 temporal test | model2 | PR-AUC 0.9845; F1 0.9177 | PR-AUC 0.9922; F1 0.9691 |
| Azure MLP, model3 temporal test | model3 | PR-AUC 0.9951; F1 0.9546 | PR-AUC 0.9979; F1 0.9857 |
| NASA MLP, FD001 official terminal test | FD001 | RMSE 37.71 cycles | RMSE 35.27 cycles |
| NASA MLP, FD002 official terminal test | FD002 | RMSE 48.84 cycles | RMSE 52.05 cycles |
| NASA MLP, FD003 official terminal test | FD003 | RMSE 45.51 cycles | RMSE 46.15 cycles |

Held-out Azure model4: centralized logistic PR-AUC **0.7937**, F1 **0.8022**; centralized MLP PR-AUC **0.9850 ± 0.0043**, F1 **0.9504 ± 0.0115**. Local MLP source1/source2/source3 held-out PR-AUC values were **0.9344/0.9443/0.9884**, respectively; no source was chosen after observing those values.

Held-out NASA FD004 official test: centralized RMSE **57.79 ± 2.08** cycles, MAE **46.44 ± 1.48** cycles, mean NASA score **2.871e7**. Local FD002-source RMSE was **59.42 ± 3.07**. FD001/FD003 source models failed severely under frozen scaling: mean RMSE **53,312.64 / 199,614.86** cycles; NASA score sums exceeded ordinary float range in all three seeds for each source, so exact scoring conventions are represented by log/scientific scores and explicit overflow flags. These failures are retained, not hidden or retrospectively corrected. FD004 run-to-failure window diagnostics remain separate from its 248-engine official terminal evaluation.

Centralized pooling is a research reference, not a privacy-preserving method. These measurements show neither universal centralized superiority nor robust cross-factory generalization for all source domains.

## Executed commands and gates

Commands ran from the repository root using the existing pinned project-local CPython 3.12.10 environment.

| Command | Verified result |
| --- | --- |
| `.\.venv\Scripts\python.exe -m pytest -q --basetemp outputs/pytest-phase02-final` | **78 passed in 39.37 s; 0 failed, 0 skipped, 0 warnings**. All 57 Phase 01 tests remain included. |
| `.\.venv\Scripts\python.exe -W error scripts/run_baselines.py --config configs/phase02.json --prepared outputs/phase01 --output outputs/phase02` | **36 completed runs**, 71.34 s; no warnings/errors. Both tasks, local/centralized comparisons, and both held-out factories evaluated. |
| `.\.venv\Scripts\python.exe -W error scripts/verify_baselines.py` | **36 saved models and 138 prediction files verified**; all predictions reproduced bit-for-bit after CSV round-trip parsing, all metrics/validation-only thresholds reproduced, frozen source statistics matched Phase 01, and 17 raw files reverified. |
| `.\.venv\Scripts\python.exe scripts/report_baselines.py` | Generated measured results directly from completed plan/checkpoints. |
| `.\.venv\Scripts\python.exe scripts/smoke_backend.py` | Live HTTP health/readiness, dataset/factory metadata, and held-out summary passed; owned Uvicorn process stopped afterward. |
| `.\.venv\Scripts\python.exe -m pip check` | `No broken requirements found.` |
| `.\.venv\Scripts\python.exe -m compileall -q src scripts` | Passed. |
| `git diff --check` | Passed. |
| `git diff --exit-code -- data docs/Project2_Description.pdf` | Passed; original tracked data/manifest/PDF unchanged. |
| `git branch --show-current` | `main`. |

The full suite includes two real-data preparation runs with identical catalogs/digests/states, real cross-client shape/source/engine-identity checks, mask/padding invariance, train-only union/frozen source tests, contamination guards, independent metric examples, threshold selection boundaries, seeded checkpoint reproduction, exported inference agreement with scikit-learn, malformed/tampered artifact rejection, capped-fit versus uncapped-evaluation targets, and a real 12-run two-epoch training/evaluation smoke plan. That smoke plan is distinct from the full-budget 36-run result set.

Early development tests exposed a scikit-learn/SciPy L-BFGS option incompatibility; the Newton–Cholesky solver avoids it without changing the environment or suppressing warnings. A test-edit placement error was corrected before the final suite. No final gate failed or was skipped. CPU/package/server checks required approved execution outside sandbox restrictions; no security/global permission settings were modified.

## Saved artifact locations and loading

- `outputs/phase02/plan.json`: authoritative statuses, config, provenance, source-file hashes, feature contracts, selected runs, and all metrics.
- `outputs/phase02/phase01-catalog.json`: exact preparation-catalog snapshot.
- `outputs/phase02/metrics.csv`: flat index of all 138 validation/test/held-out evaluation records.
- `outputs/phase02/verification.json`: completed reload/prediction/metric audit results.
- `outputs/phase02/<run-id>/model.json`, `weights.npz`, `transform-*.json`, `predictions-*.csv`: real per-model artifacts. Example: `outputs/phase02/nasa-mlp-pooled-seed42/`.

Artifacts are generated/ignored and remain locally available. The [README](../README.md) gives exact reproduction/startup/loading commands. `fedmaint.api.artifacts.load_model_artifact` offers validated `predict_raw` and `predict_encoded` methods; no new HTTP prediction/training endpoints or full serving system was added.

## Changed files

Modified documentation: `AGENTS.md`, `README.md`, `docs/ROADMAP.md`, `docs/PROJECT_OVERVIEW.md`.

Created:

- `configs/phase02.json`.
- `src/fedmaint/models/__init__.py`, `contracts.py`, `data.py`, `metrics.py`, `network.py`, `training.py`, `artifacts.py`, `experiments.py`.
- `src/fedmaint/api/artifacts.py`.
- `scripts/run_baselines.py`, `scripts/verify_baselines.py`, `scripts/report_baselines.py`.
- `tests/test_model_contracts.py`, `tests/test_model_metrics.py`, `tests/test_model_training.py`.
- `docs/PHASE02_PROTOCOL.md`, `docs/PHASE02_RESULTS.md`, `docs/PHASE02_ACCEPTANCE.md`.

Changes remain uncommitted on `main` for review. Phase 01 implementation/dependency files and its historical acceptance evidence remain intact.

## Scientific limitations and next steps

No blocker prevents Phase 02 acceptance. Predictive performance has no imposed minimum acceptance threshold; severe zero-shot failures remain valid measured outcomes and important limitations. Factory domains are simulated, local standardized coordinates are heterogeneous, source variance can amplify unseen operating regimes, and zero-fill can reveal domain membership. Azure source/event-availability and surveillance assumptions remain. High Azure scores on this synthetic dataset do not establish industrial generalization. NASA window-wise validation does not reproduce official-test terminal censoring, and correlated run-to-failure windows are not independent engines. Small budgets/three optimization seeds do not establish optimality or statistical significance. Logistic seeds repeat a deterministic fit. No formal privacy mechanism or guarantee is implemented.

Phase 03 requires explicit user approval. Its work should freeze federated feature/missing-coordinate semantics and matched experimental budgets before implementing FedAvg/FedProx. No federated, privacy, KG, LLM, or frontend implementation has begun.
