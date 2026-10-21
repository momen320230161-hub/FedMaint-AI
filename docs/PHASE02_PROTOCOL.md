# Phase 02 models, selection, and evaluation protocol

This is a fixed, bounded baseline experiment, not a search for state-of-the-art results. The academic brief requires local-only and pooled references before federation, separate classification/RUL tasks, and an unseen factory. All Phase 01 temporal, engine, raw-byte, and held-out rules remain in force. No model-selection rule below uses test/held-out observations. Defaults were fixed before the full experiment's evaluation stage.

## Audit and shared feature coordinates

Phase 01's exact source hashes, rows, array digests, feature orders, normalization states, and split identities are audited when loading prepared inputs. Its preprocessing removed constants independently: all Azure clients retained 21 features; NASA FD001/FD002/FD003 retained 17/24/18. Directly pooling those NASA arrays would be invalid.

Phase 02 forms a **sorted union of retained features from participating training clients only**. Each already-normalized client array is placed into those coordinates, with zero in coordinates the client dropped. Held-out arrays/states/distributions never define the union. Source medians, means, scales, and local retained/dropped decisions are reused unchanged. No preprocessing fitting is introduced in Phase 02, and no Phase 01 source code or prepared artifact is rewritten.

Result: 21 common Azure inputs; 24 common NASA features per cycle. A NASA input is its chronological 30-cycle flattened feature window (720 coordinates) followed by its boolean mask converted to 30 numeric entries: **750 input dimensions**. Padded rows are forcibly zeroed before encoding, including during raw inference, so arbitrary padding values cannot affect predictions. The mask distinguishes true zero observations from padding. No factory/engine/machine ID, timestamp, failure target, or RUL label enters the feature vector.

Union construction exchanges schema/retention metadata, not pooled means/variances. Local scalings still represent different physical coordinate systems: a standardized value of one need not mean the same physical measurement across factories. Zero-filled dropped coordinates also vary by factory and can indirectly identify domains. This is a documented baseline compromise, not a proof of domain invariance or a final federated representation design.

## Frozen zero-shot source strategy

For each local-only model, held-out inputs always use that model's own training-factory preprocessor. For a centralized model, select the training factory retaining the most features; break ties by ascending factory ID. This rule uses training-only schema coverage, not validation/test/held-out performance. It freezes **azure-model1** for centralized Azure and **nasa-FD002** for centralized NASA.

Raw held-out features are imputed/scaled only with the frozen source's retained statistics. Features dropped by that source stay zero in shared coordinates even if held-out data vary in them; no held-out min/max, median, constant detection, clipping, renormalization, batch adaptation, or source switching occurs. The same transform is embedded in the saved bundle. This can fail severely under operating-condition shift, especially when a single-condition NASA source has tiny training variances. Such failures are measured and retained, not repaired after inspecting held-out scores.

For participating clients, both local and centralized evaluation uses each target client's original train-fitted transform, as during training. The centralized reference pools **locally standardized training samples** from the three participating clients. It is an experimental pooling reference, not privacy-preserving training or a globally standardized raw-data benchmark. No validation, local test, official test, or held-out record is pooled into fitting.

## Architectures and budgets

All models run on CPU with BLAS thread pools constrained to one for bounded resource use and reproducibility. The pinned NumPy/SciPy/scikit-learn environment is reused; adding PyTorch/GRU would not be necessary for these deliberately small MLP baselines.

| Setting | Frozen default |
| --- | --- |
| Random seeds | 42, 43, 44 |
| Azure logistic regression | Binary L2 logistic regression, Newton–Cholesky solver, intercept, no class weighting; C candidates 0.1, 1, 10; max 2,000 Newton iterations, tolerance 1e-6 |
| Azure MLP | 21 → 32 ReLU → 1 sigmoid; binary cross-entropy, Adam |
| NASA MLP | 750 → 32 ReLU → 1 linear; squared-error training, Adam |
| MLP L2 alpha | 0.0001 |
| MLP learning-rate candidates | 0.001, 0.003 |
| MLP batch size | 128, reduced only if a training set has fewer rows |
| MLP epoch cap / patience | 40 per candidate / 6 nonimproving validation epochs |
| Data order | Seeded per-estimator shuffle; no resampling/oversampling |
| RUL units during optimization | Prepared training `y` divided by fixed 100 cycles; predictions multiplied by 100 and floored at zero; no maximum prediction cap |

The supplied Phase 01 config is uncapped. If another audited Phase 01 preparation uses a RUL cap, training consumes its prepared capped `y`, while validation/checkpoint selection and every reported evaluation still consume uncapped `raw_y`. The scaling factor 100 is fixed, not learned from validation/held-out labels. Raw labels remain saved in prediction CSVs.

Each seed runs both Azure architectures for three local clients plus one centralized reference, and the NASA MLP for three local domains plus one centralized reference: **12 runs per seed, 36 runs total**. Local and centralized models use matched architecture, candidate grids, batch-size rule, epoch cap, and stopping rule. An epoch visits every permitted sample; the centralized model consequently performs more total updates than an individual local model. These are matched per-record exposure/selection budgets, not identical FLOP or wall-time budgets. Actual candidate histories, optimizer iterations, selected epoch/hyperparameter, fit/validation sample counts, and seconds are recorded.

Logistic regression is deterministic here; repeating its seed label does not create independent data/optimization replicates or evidence of statistical significance. MLP seeds vary initialization/shuffle while the Phase 01 data split remains fixed. Three seeds quantify optimization sensitivity only; no significance or confidence-interval claim is made.

## Selection boundaries

Every fit batch must be explicitly `train`, from the declared participating factory/factories; every selection batch must be `validation` from those same factories. Cross-split duplicate sample identities are rejected. NASA identities include domain and official source, so equal numeric engine IDs in different source files are not falsely merged. Phase 01 engine-disjoint validation and purged Azure support intervals are unchanged.

Azure candidate/checkpoint selection maximizes validation **average precision (AP)**. NASA maximizes negative uncapped validation-window RMSE. Every candidate receives the same cap; early stopping is candidate-local. Strict-score ties retain the first candidate/earliest epoch. After selecting an Azure checkpoint, select the classification threshold maximizing validation F1, breaking exact ties in favor of the higher threshold. No internal random validation split, test threshold tuning, or held-out choice occurs. Central selection pools permitted validation samples, so larger clients have more weight; it is not an equal-factory objective. NASA validation uses windows from held-out validation engines, not official test engines.

The runner trains and saves **every** planned model/threshold first. Only then does it load any test/held-out sample arrays. A test instruments this boundary during a real 12-model smoke run. Evaluation never invokes fitting or changes a selected checkpoint. Failed runs remain explicitly failed in `plan.json`; incomplete plans cannot be marked complete. Existing experiment plans are not overwritten: use a new output directory for a new run.

## Metrics and independent units

Azure reports both trapezoidal precision–recall area (`pr_auc`) and noninterpolated average precision (`average_precision`), explicitly distinguished. It also records F1, precision, recall, ROC-AUC where defined, positive prevalence, the frozen threshold, and a confusion matrix in `[[TN, FP], [FN, TP]]` order. Predictions use `probability >= threshold`. Zero predicted positives gives precision/F1 zero. Without positive truth PR area/AP are explicitly undefined (`null`), and a single true class gives undefined ROC-AUC. Threshold/model selection requires both validation classes.

NASA evaluates uncapped truth with RMSE, MAE, and the **sum** of asymmetric NASA penalties. For `d = predicted_RUL - raw_true_RUL`, each penalty is `exp(-d/13)-1` when `d < 0`, otherwise `exp(d/10)-1`. The sum follows [Saxena et al., equation 9](https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf). It is not divided by engine count. Stable log-sum computation retains `nasa_score_log` and `nasa_score_scientific`; an unrepresentable ordinary float becomes `null` with `nasa_score_overflow=true`, never infinity or a clipped score. Exact prediction has score zero.

Official NASA local tests contain exactly one final observed window per official test engine (100/259/100 for FD001/FD002/FD003). FD004 `held_out_official_test` contains exactly 248 engines/terminal windows. Its separate 5,516-window run-to-failure evaluation is labeled as a window-level diagnostic with correlated observations, **not** 5,516 independent engines or a comparable official terminal score. Validation likewise has multiple windows per engine. Scores across pools with different numbers of units are not directly comparable, and seed variations are not independent engine resamples.

AP and trapezoidal PR area use the [scikit-learn metric definitions](https://scikit-learn.org/1.6/modules/model_evaluation.html#precision-recall-f-measure-metrics). All numeric results come from newly executed runs; historical implementation results are not imported.

## Saved artifacts and backend boundary

`outputs/phase02/plan.json` records frozen config, status, source revision/file hashes, environment, feature contracts, every selected run, metrics, prediction checksums, and timings. `phase01-catalog.json` preserves the exact audited preparation catalog. Each run directory contains `model.json`, numeric `weights.npz`, copied train-fitted `transform-*.json` states, and per-factory/split prediction CSVs with source-scoped sample identity, truth, prediction, and Azure decision. `metrics.csv` is a flat, convenient result index; JSON is authoritative for null/overflow/structured values. Generated artifacts are ignored by Git.

`fedmaint.api.artifacts.load_model_artifact(directory, expected_dataset=...)` validates metadata, task, source membership, dimensions, array keys/finite floating-point weights, preprocessing hashes and checkpoint SHA-256. Loading uses `allow_pickle=False`. It returns `predict_encoded(...)` and `predict_raw(...)` with frozen transforms and mask handling; these are internal interfaces for later APIs, not new HTTP serving endpoints. Unknown source transforms and mismatched tasks/shapes are rejected.

## Limits and next-phase boundary

This phase does not establish clinical/industrial reliability, optimal hyperparameters, calibration under site shift, domain-independent factory generalization, formal privacy, or state-of-the-art accuracy. Azure label/availability assumptions and simulated factory limits remain as in Phase 01. The daily Azure grid limits temporal information; no unsupported GRU is added to its tabular summaries. NASA flattened windows are a small MLP baseline, not a recurrent temporal architecture. Selection on full run-to-failure validation windows differs from official terminal censoring. Severe held-out extrapolation and zero-fill/local-scaling limitations must remain visible in reports.

A later phase must explicitly authorize and evaluate FedAvg/FedProx under compatible feature/missing-coordinate semantics and comparable budgets. No federation, privacy mechanism, KG, LLM, frontend, or Phase 03 implementation is authorized here.
