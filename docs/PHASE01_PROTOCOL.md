# Phase 01 data contracts and experimental protocol

This phase implements data/backend foundations only. The academic brief motivates separate task definitions, non-IID simulated clients, client-local statistics, and an unseen factory. The exact defaults below are engineering/experimental choices for this phase, not prescriptions from the PDF or validated optimal hyperparameters. No model is trained, selected, or scored.

## Integrity and provenance

`data/manifest.json` is read-only. Every preparation run streams SHA-256 and logical row counts for all five Azure CSVs and twelve NASA numeric files before loading them. The Azure manifest explicitly records original CRLF file hashes and alternate Git LF hashes; either recorded exact variant is valid, and `hash_variant` identifies it. No normalization, repair, or manifest rewrite occurs. Any unlisted hash, row mismatch, missing file, or source change during preparation fails the run. Generated output is rejected under source/input trees and their parent directories.

The metadata catalog records configuration values and SHA-256, the manifest SHA-256, actual source hashes/row counts/size/mtime, package versions, partitions, numeric-array digests, feature order, and transformed status. The manifest and original PDF remain unchanged. This repository's files match the original-file hashes. Archive hashes describe archives not retained as pipeline inputs; they are not claimed reverified. The NASA source README/research PDF are preserved but are not numeric-file entries in this manifest.

## Azure: recurrent any-component failure classification

Each telemetry row is an hourly prediction point `t` for a particular machine. `target = 1` means at least one recorded component failure has timestamp in **(t, t + horizon]**. Defaults: 24-hour history, 24-hour horizon, 24-hour sampling stride. Failures at `t` are excluded; failures at the right horizon endpoint are included. Multiple failure records within the horizon remain one positive label. This is a recurrent occurrence label, not a mutually exclusive component class or a survival probability. We do not invent downtime, replacement resets, or an at-risk recovery state not supported by these files.

Telemetry features use exactly W consecutive hourly rows from `t-(W-1)` through `t`, equivalent to `(t-W hours, t]`. They contain mean, min, max, and observed-value count per sensor. The current row is included: prediction is made after that hourly observation is available. Full windows are required. Timestamp gaps/duplicate machine-hour records fail validation; incomplete boundary windows are dropped. Missing numeric measurements remain NaN for window aggregation, accompanied by counts, and are imputed only by the permitted client's fitted training transform. Timestamps/IDs/category keys cannot be missing. Infinite/nonnumeric values and unknown event codes fail validation. No interpolation or backward fill from future observations is performed.

Error and maintenance features count each code within `(t-W, t-event_lag]`. Maintenance recency is hours since the latest eligible event **within that window**, otherwise NaN. Default lag: one hour, excluding events recorded at the prediction timestamp. This assumes historical event records are available after that lag; the source lacks true ingestion/availability timestamps, so deployment must verify or increase it. Changing later telemetry, errors, maintenance, or failure outcomes cannot change earlier features. Failure events are used only for targets. Target counts, future records, component failure labels, timestamps, machine IDs, and factory IDs do not enter the numerical feature matrix.

All five tables load, validate, and sort chronologically where applicable. Machine metadata defines model groups and descriptive age summaries. Optional `metadata` features expose static `age` only; they are **disabled by default** because the source has no metadata-as-of timestamps. Enabling age requires the explicit assumption that the recorded static value was known at prediction time. The feature-group switch supports telemetry/error/maintenance/metadata ablations later, without tuning against held-out outcomes.

## Azure temporal partitions and simulated factories

Factories are entire machine-model groups: model1 (16 machines), model2 (17), model3 (35) are training clients; model4 (32) is held out. Machine IDs never span factory groups. Model labels are simulated equipment domains, not independent observed organizations.

For each training factory and each machine:

| Split | Permitted interval |
| --- | --- |
| Train | Feature history starts within observations; target endpoint is strictly before 2015-08-01 00:00 |
| Validation | Feature history starts on/after 2015-08-01; target endpoint is strictly before 2015-10-01 00:00 |
| Test | Feature history starts on/after 2015-10-01; target endpoint is no later than that machine's final observed hour |

Purging considers the **entire feature and target support**, not only the prediction timestamp. Validation/test history cannot borrow rows/events from an earlier partition. Nontraining final hours whose horizon is not fully observable are right-censored and omitted. The end of telemetry is assumed to mark complete failure surveillance; that completeness cannot be proven from the supplied event log alone.

The same machines recur in train/validation/test within each training client: this is temporal generalization on known model groups. The separate, completely machine-disjoint model4 experiment measures an unseen simulated equipment domain. Its full eligible time span is labeled `held_out`, never train/validation/test; it contributes no fitted preprocessing. Shared calendar effects and synthetic generation mean these factory groups are not proven independent samples from real factory populations.

## NASA: cycle-level RUL regression

All FD001–FD004 files are parsed as 26 columns: engine ID, cycle, three operating settings, and **21** sensors. Within each domain/source, engine IDs and cycles are positive integers, cycles start at 1 and are contiguous, and engine/cycle keys are unique. All supplied numeric observations must be finite; missing NASA source measurements fail validation instead of silently changing a trajectory. The reusable local transform still supports missing values for subsequent inference. Engine identity is `(domain, official source, engine ID)`; identical numeric IDs in official train and test files are different engines.

Training targets: `max_observed_training_cycle(engine) - cycle`. These files are run-to-failure, so their final targets are zero. Official test targets at any observed cycle: `last_observed_test_cycle(engine) + official_RUL(engine) - cycle`. The official RUL vector is mapped to ascending contiguous test engine IDs, with a strict length check. Final-test evaluation uses one terminal window per official test engine. Earlier-cycle derived test targets are supported but not emitted as independent official evaluation samples by default.

The default sequence window is 30 cycles and training/validation stride is 10. Windows remain inside an engine. Training/validation windows need 30 observations. Terminal test trajectories shorter than 30 receive zero left-padding and a boolean mask; observed rows are transformed **before** padding, leaving padding zero in normalized space. Consumers must honor the mask. Raw held-out windows remain explicitly untransformed. No component failures, errors, replacements, or maintenance events are inferred for NASA.

Optional `rul_cap` caps emitted `y` but never overwrites `raw_y`. Default: uncapped. Later NASA RMSE/scoring must use the documented uncapped official terminal truth (`raw_y`) unless an explicitly named alternative evaluation is reported. A training cap does not silently redefine official evaluation truth.

The supplied NASA README reverses the FD004 train/test engine counts and has a sensor-index typo. The parsed files and manifest show **249 official training engines and 248 official test engines** in FD004, and 26 total columns including 21 sensors. These source-document inconsistencies are recorded rather than repaired.

## NASA domain partitions and strict held-out protocol

FD001, FD002, FD003 are three training factories; FD004 is the held-out domain. In each training domain a sorted list of official training engine IDs is permuted with NumPy `default_rng(seed=42)`, and the first `ceil(20% of engines)` become validation engines. Remaining engines are training only. No validation engine contributes a training window or fitted statistic. Every official test engine is reserved for official-test evaluation, regardless of its numeric ID matching an official train ID.

FD004's official train file supplies a separate `held_out_run_to_failure` evaluation pool; its official test file supplies `held_out_official_test`. Neither supplies training/validation data or preprocessing. These two evaluation pools must not be conflated with the official tests in the three training domains. Domain partitioning induces differences in operating conditions/fault modes and trajectory distributions; it also confounds those factors and is not evidence of factory identity or real intercompany privacy boundaries.

## Local fitting and reuse

Six transformations are fitted: three Azure clients on **training samples only**, three NASA clients on **all unique raw rows of training engines only**. NASA fitting does not overweight cycles because of overlapping sequence windows. Fit guards require one permitted factory ID and `split=train` on every row, and accept only explicitly named `feat_` columns. All-missing training columns and exact constants after median imputation are removed. Medians, standard-scaler mean and population scale, retained/dropped feature order, fit-row count, and factory ID are serialized as numeric JSON. Transforms reuse those values on validation/test/inference and never refit. No pickle loading is required.

Every validation/test sample is transformed using its training factory's state. Held-out samples get **no locally fitted transform**, remain raw, and retain masks/missingness. A later zero-shot experiment must freeze an appropriate source-client preprocessing strategy, compatible feature schema, and source predictor before held-out evaluation, using training/validation evidence only. This phase does not silently pool client statistics or claim that independent local scalers already define a shared federated input space.

Client summaries quantify unit counts, ages/operating-setting means, sensor means, trajectory lengths, and Azure positive-window prevalence. They are audit descriptions, not input to tuning decisions. Holdout identity and defaults were fixed before running preparation. Do not use held-out summaries, labels, feature variation, or performance to select windows, features, scalers, models, checkpoints, thresholds, or adaptation. Neither validation nor official tests nor held-out data enter preprocessing fit.

## API and operational scope

`/health` is liveness plus metadata readiness. Dataset and factory endpoints serve typed metadata derived from a completed full validation/preparation run. The catalog is published last, and an existing catalog is withdrawn when preparation starts writing replacement artifacts. A changed config requires a new output directory. Run one preparation writer per output directory; concurrent writers are unsupported. Requests check the manifest hash and source sizes/mtimes, without loading telemetry or hashing the 80 MB CSV on each request. A deliberate same-size edit with restored mtime would require the full integrity command to detect; this is not an adversarial tamper-proof storage service. Generated catalogs and outputs are trusted local research artifacts, not user-uploadable API inputs.

The phase provides no predictor, score/performance claim, FedAvg/FedProx, privacy mechanism/guarantee, KG, LLM, frontend, job scheduler, database, or production deployment. It establishes contracts and reproducible samples for later phases. Ordinary partitioning or client-local normalization is not differential privacy or secure aggregation.
