> Historical research record. Commands, code paths and measurements below refer to the pre-V3 revision `f6a4e8e4cdc276c65f76a42286428626beb2c0f2`. Current implementation and protocols: [Architecture V3](ARCHITECTURE_V3.md), [data](V3_DATA_PROTOCOL.md), [experiments](V3_EXPERIMENT_PROTOCOL.md). Historical outputs remain preserved.

# Phase 02.1 execution and acceptance evidence

Date: 2026-10-23. Engineering implementation and the bounded experiments are complete locally on the existing `main` branch. Codex implemented, reviewed and tested directly. Changes remain uncommitted; nothing was pushed. Phase 03, FedAvg/FedProx, privacy mechanisms, KG, LLM and UI were not started.

**Scientific acceptance qualification:** the strict prohibition on *all* held-out access before freeze was not fully met. The two historical full-suite regression runs included real-data preparation and a small Phase 02 smoke experiment that read FD004/Azure model4. Only pass/fail test output was inspected; none of their target metrics/predictions were imported into research selection. The new Phase 02.1 loader and selection stages remained source-only. This is a disclosed execution deviation, not a pristine protocol-compliance claim. The predeclared new candidate panel was evaluated exactly once after its freeze. The exposure is recorded in `outputs/phase021/protocol-deviations.json`, sealed before that evaluation. Future clean executions should defer the legacy real-data regression suite until after final evaluation. Original Phase 02 FD004 observations also already existed, so these results are not fully blind.

The full generated [measured-results report](PHASE021_RESULTS.md) contains every source fold/endpoint, final local/centralized pool, per-seed metric, floor diagnostic, Azure confusion matrix and checkpoint source. The approved [research plan](PHASE02_1_RESEARCH_PLAN.md), original [academic PDF](Project2_Description.pdf), and Phase 01/02 protocols remain unchanged. Wording about pre-freeze gating in the results report refers to the **new candidate evaluation path**; the legacy-test exception above applies to the execution as a whole.

## Completed bounded experiment

| Stage | Actual research fits | Outcome |
| --- | ---: | --- |
| N0/N1/N2 x matched MLP/GRU x three source folds, seed42 | 18 | All completed |
| N0-MLP and shortlisted N2-MLP/N2-GRU x three source folds, seed43 | 9 | All completed |
| Three final candidates x three local clients/pooled x seeds42/43 | 24 | All completed before target scoring |
| Six Azure feature/lag arms x three held source-model folds, seed42 | 18 | All completed; no model4 selection |
| Total | **69** | **0 failed, skipped, incomplete or retried research fits** |

Charged fitting/inner-selection/artifact time was **488.2474 seconds**, below the 2700-second cap. This excludes preprocessing, audit, inference and regression-test runs; it is not an end-to-end runtime claim. Tests have their own scratch fits, including the two legacy 12-fit smoke plans, and do not add research candidates. No post-FD004 fitting, routing change, threshold change, feature selection or hyperparameter search occurred.

The freeze digest is `1b3592442f9548ab99b6a732498b0b606bdc5b967b3e09d5434cbdbaea7f90aa`. It covers **323 artifacts** plus implementation/configuration/lock/test/generator hashes. The freeze includes the selection result, all 24 final checkpoints, separate private transforms, source-routing identities, Azure audit arms, budget, test XML and deviation record. `evaluation-started.json` was created exclusively after the freeze; a repeat marker is rejected, even following an interruption. No second candidate evaluation command was run. Subsequent audits recomputed metrics from saved predictions and reloaded weights without another held-out model forward pass.

## Supported findings and limits

Source-only diagnostics reproduced all six prior comparisons with relative/absolute tolerance1e-12. They verify the near-constant-scale/operating-condition amplification mechanism: source FD001/FD003 transforms produce absolute standardized-coordinate p99 near19214/19173 on FD002. This is evidence of a contributor, not proof that scaling is the only RUL failure mechanism.

The selection criterion is the equally weighted mean RMSE over cycles50/100 and three outer domains. Each outer domain is absent from fitted transformations, constant masks, condition dictionaries, inner checkpoint choice and source routing. Inner training/validation engines are disjoint. The saved key digests, exact engine membership, transform fit-engine IDs and input hashes support the provenance audit. The outer folds are exploratory selection data, not an independent unbiased test estimate.

| Candidate | Source macro RMSE seed42 | Confirmation seed43 |
| --- | ---: | ---: |
| N0-MLP | 24545.94 | 32673.75 |
| N0-GRU | 81.00 | Not predeclared for confirmation |
| N1-MLP | 329.96 | Not shortlisted |
| N1-GRU | 70.87 | Not shortlisted |
| N2-MLP | 321.38 | 314.81 |
| **N2-GRU** | **64.00** | **69.46** |

N2-GRU and N2-MLP each pass the predeclared macro/worst-domain/two-of-three-domain promotion gates versus N0-MLP in both seeds. N2-GRU is selected because its combined source macro error is lower. Matched N2-GRU beats matched N2-MLP in all three domains in both seeds. N2 versus N1 under GRU has only one screening seed; the incremental benefit of condition bins is not independently confirmed. GRU robustness across every factory is not established. In the FD002-held fold,45715/53759 rows use unsupported-bin fallback, exposing absent operating-regime support rather than borrowing held-domain statistics.

All new comparisons share raw linear-head predictions with no floor or upper cap; zero-floor results are separate diagnostics. Ground truth is uncapped. Raw/floored source macro results are both tabulated and show that changed output handling does not explain the large source gains. Historical Phase 02 remains zero-floored and follows a different optimizer/search/validation protocol.

Final FD004 official terminal results below average **the same seeds42/43**,248 engines each. Standard deviations and per-seed NASA sums remain in the measured-results report. These final outcomes did not select the winner.

| Protocol / pooled model | RMSE | MAE | NASA asymmetric sum |
| --- | ---: | ---: | ---: |
| Historical Phase02, zero-floored, different training/selection protocol | 58.9865 | 47.2912 | 4.08850e7 |
| New matched N0-MLP, raw | 52.7280 | 43.6562 | 5.87202e6 |
| New matched N2-MLP, raw | 49.2815 | 37.0151 | 2.83724e9 |
| **Selected N2-GRU, raw** | **34.7756** | **26.6972** | **8.42398e5** |

N2-GRU pooled per-seed RMSE is36.8911/32.6601, MAE27.3583/26.0361, NASA sum1.66126e6/2.35402e4. Sample SD of RMSE is2.9918; two seeds are not significance evidence. The historical all-three-seed mean57.79 remains in the unchanged original report. Historical-versus-new differences cannot be attributed solely to architecture or normalization; the new matched controls provide the relevant evidence.

Failures and trade-offs remain visible: N2-GRU local FD003→FD004 averages142.05 RMSE, and local FD002→FD004 averages67.05 (worse than the historical same-seed58.09). Pooled known-FD003 NASA score averages1.2102e8 versus historical9.8965e7 despite better RMSE. N2-MLP's pooled FD004 NASA sum is worse than the new baseline despite lower RMSE. No universal improvement or elimination of tail risk is claimed. Overflow is preserved as null numeric score plus log/scientific values and explicit overflow flags; no capped substitute is used. FD004 run-to-failure metrics remain separate correlated-window diagnostics.

## Azure availability findings

Every arm retains the same24480 source-model machine/timestamp samples before fold selection (14280 train,4012 inner-validation,6188 temporal-test rows), using models1/2/3 only. Fit/validation pools exclude the pseudo-held model; local preprocessing and thresholds use only the permitted source training/validation records. Arm/fold input dimensions and parameter counts, AP, trapezoidal PR-AUC, F1, precision/recall, prevalence and confusion matrices are in the measured report. Azure uses seed42 only.

| Arm | Source-fold mean AP | Trapezoidal PR-AUC | F1 |
| --- | ---: | ---: | ---: |
| Telemetry only | 0.2300 | 0.2217 | 0.2278 |
| Telemetry + errors | 0.9941 | 0.9941 | 0.9756 |
| Telemetry + maintenance | 0.2285 | 0.2164 | 0.2594 |
| Full, lag1h | 0.9910 | 0.9909 | 0.9642 |
| Full, lag6h | 0.9891 | 0.9890 | 0.9601 |
| Full, lag12h | 0.9908 | 0.9907 | 0.9608 |

The very high score is principally associated with **error-event features**, not maintenance. Longer fixed lags preserve high source scores, but do not establish event availability. Source-fold no-skill AP/prevalence ranges0.01476–0.02885.

The predeclared descriptive timing audit was completed after final scoring without any model fit or selection change, using source-model records only. `outputs/phase021/availability-audit.json` records its code/freeze hashes. All24480 prediction timestamps fall at05:00;562/578 source failure events and all2247 maintenance events occur at06:00. Error events peak at06:00 (822/2702). All527 positive sampled future-failure windows have an eligible past error; none of22190 error-free windows is positive. This strong synthetic-grid association helps explain the ablation result; it is not proof of causal leakage or actual posting-time availability. A prior-day06:00 record can remain eligible at05:00 even under a12-hour lag, so lag stress tests do not remove this structure.

The source contains **no posting/ingestion timestamp**. Exact prediction-time events are excluded with lag1; duplicate events are rejected; future-event mutation tests pass. A backdated-posting test explicitly demonstrates the remaining vulnerability: an event posted after prediction but timestamped before prediction cannot be recognized as unavailable. Failure features are never inputs; future failures define labels only. Maintenance logging time cannot be distinguished from action time. **Telemetry-only is the conservative Phase03 reference until real event availability is established.** Source associations and synthetic scores cannot certify industrial deployment performance.

## Backbone, feature contract and academic compatibility

Recommended NASA Phase03 reference: **N2 with a one-layer32-unit GRU**,24 inputs and a scalar linear head, **5601 parameters**. The matched MLP is750→32 ReLU→1, **24065 parameters**. Keep the fixed sorted schema of three `feat_setting_*` and21 `feat_sensor_*` fields,30 chronological steps, neutral locally constant/missing coordinates, and an explicit boolean validity mask. Strip left padding before recurrent packing and use the final valid hidden state. Exact column order is stored in every private state and generated by `optimization.common.FEATURES`.

For Azure, use a separate conservative telemetry baseline: the12 sensor mean/min/max features (four sensors × three statistics), a32-unit ReLU hidden layer and sigmoid output (**449 parameters**). The audit also records17-input errors controls (**609**),16-input maintenance controls (**577**) and21-input full controls (**737**). Telemetry count fields were locally constant under complete24-hour windows and excluded in these fits. A future shared feature contract must freeze the common names/order before aggregation; event features require a documented availability policy.

Keep medians, scales, imputation values, constant masks, engine membership and regime dictionaries **private and separate** from neural weight tensors. Aggregate/proximal-penalize only the shared neural parameters in any later approved FedAvg/FedProx implementation. Common tensor shapes enable arithmetic aggregation but do not establish aligned local semantics or federated accuracy.

The design is compatible with local-only **fitting** and mechanically suitable for future federated backbones. It **does not yet satisfy the complete academic privacy/FL requirements**: no FL algorithm, explicit privacy mechanism, threat model, accounting or residual-leakage analysis was implemented. Centralized pooling is non-private. Strict unseen-site zero-shot deployment remains unresolved: shipping frozen source scalers exposes source distribution information, shipping raw target data violates the intended boundary, and fitting target scalers is adaptation. A later approved threat-model/deployment protocol must resolve this distinction.

## Validation and preservation

| Executed command / gate | Exact result |
| --- | --- |
| `python -m pytest -q --basetemp outputs/phase021/pytest-full` | 90 passed in60.28s;0 failures/errors/skips; before three added boundary tests |
| `python -m pytest -q --basetemp outputs/phase021/pytest-final --junitxml outputs/phase021/tests-final.xml` | **93 passed in102.13s;0 failures,0 errors,0 skipped; no warnings reported** |
| `python -m pytest tests/test_optimization.py -q --basetemp outputs/phase021/pytest-preseal` | **15 passed in21.45s** after the freeze-readiness gate; no real held-out data read |
| `python -W error scripts/phase021.py diagnostics` | All six source diagnostics reproduced; input/state provenance recorded |
| `python -W error scripts/phase021.py source`, `azure`, `fit` | 27 source,18 Azure,24 final fits completed;69 total |
| `python -W error scripts/phase021.py freeze` | Tests, stage completion, model panel and exposure record passed readiness checks;323 artifacts sealed |
| `python -W error scripts/phase021.py evaluate` | One new-candidate evaluation invocation;24 final models,84 NASA prediction pools |
| `python -W error scripts/verify_phase021.py` | **129 prediction CSVs** independently rescored (27 source +84 final NASA +18 Azure);24 final NASA models reloaded;323 frozen artifacts checked;**344 original files byte-identical** |
| `python -W error scripts/audit_phase021_availability.py` | Descriptive source-only timing/grid audit; no additional fit or model evaluation |
| `python scripts/report_phase021.py` | Generated measured results from verified artifacts, with report/source/freeze provenance |
| `python -m pip check` in `.venv-phase021` | No broken requirements found |
| `python -m compileall -q ...` for new modules/scripts/tests | Passed |
| `git diff --check` | Passed |
| `git branch --show-current` | `main` |

All Python commands above used `.\.venv-phase021\Scripts\python.exe`. The initial new-unit-test invocation had5 passes and7 setup errors because the requested pytest scratch parent did not yet exist; creating `outputs/phase021` fixed the setup. Subsequent unit/full suites passed. No research fit failed or was retried.

New tests cover source-only fitting guards, N0 parity with the historical scaler, constant neutrality, N2 unknown-bin fallback, fixed model counts, mixed-length/pad-value invariance, pickle-free weight round-trips, historical sklearn→PyTorch forward parity on synthetic inputs, pre-freeze loader rejection before any read, freeze-tamper rejection, exclusive-evaluation repeat rejection, uncapped official terminal RUL alignment/padding on synthetic files, source promotion gates, deterministic CPU fitting, budget repeat rejection, and the backdated-posting limitation. Existing Phase01/02 regression tests remain unmodified.

The original environment `.venv`, `requirements-lock.txt`, raw datasets/manifest, academic PDF, Phase01/02 source/tests/configs, all original Phase02 checkpoints/predictions/reports, Git history and backups were preserved. The344-file audit includes the approved alignment document and permits only the intentional `.gitignore` environment entry. No global Codex/MCP configuration was changed during this implementation.

## Changed files, saved artifacts and reproduction

Modified: `.gitignore` adds `.venv-phase021/`.

Created:

- `configs/phase021.json`: bounded, reproducible research settings.
- `requirements-phase021.lock`: separate full pinned CPython3.12/Windows CPU environment, including `torch==2.6.0+cpu`; all original dependency pins retained.
- `src/fedmaint/optimization/__init__.py`, `common.py`, `normalization.py`, `nasa.py`, `torch_models.py`, `training.py`, `audit.py`, `selection.py`, `azure.py`, `workflow.py`.
- `scripts/phase021.py`, `scripts/verify_phase021.py`, `scripts/report_phase021.py`, `scripts/audit_phase021_availability.py`.
- `tests/test_optimization.py`.
- `docs/PHASE021_RESULTS.md`, `docs/PHASE021_ACCEPTANCE.md`.

The approved `docs/PHASE02_1_RESEARCH_PLAN.md` was already present/untracked at implementation start and remains byte-identical. All generated experiments, audit records and test scratch artifacts are separately ignored under **`outputs/phase021/`**. The new environment is `.venv-phase021/`; old artifacts were not overwritten.

The measured report contains complete environment/stage reproduction commands. Do not rerun research stages into the accepted output directory: guards reject duplicate fit IDs, existing stage records and the exclusive evaluation marker. Use a separate fresh checkout/output location retaining the original Phase01/02/alignment inputs for a deliberate reproduction; do not remove accepted artifacts or backups. Safe inspection commands are:

```powershell
.\.venv-phase021\Scripts\python.exe -W error scripts/verify_phase021.py
.\.venv-phase021\Scripts\python.exe scripts/report_phase021.py
.\.venv-phase021\Scripts\python.exe -W error scripts/audit_phase021_availability.py
```

The full regression command is safe **after** freeze/evaluation, but its legacy real-data test deliberately accesses held-out datasets. Future source-only workflows must account for that behavior explicitly.
