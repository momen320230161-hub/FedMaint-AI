# V3 Phase 01 data protocol

Frozen before V3 training, 2026-10-26. Original targets, support intervals, engine
split membership and holdouts from PHASE01_PROTOCOL.md are retained. That document
is a historical detailed specification; the following supersedes implementation
paths and feature-dimension behavior without silently redefining evaluation truth.

## Azure

All five source tables are validated and exact accepted manifest variants checked.
Primary inputs are telemetry only: four sensors times mean/min/max/observed count
in `(t-24h,t]`, including the current available hourly measurement. Features require
24 consecutive hourly records and use a fixed 24-hour stride. A missing measurement
stays missing until train-only median imputation; there is no future interpolation.

Label is any component failure in `(t,t+24h]`, open left and closed right. Prediction
history and target support are purged across train cut 2015-08-01 and validation
cut 2015-10-01: train target ends strictly before train cut; validation history starts
on/after train cut and target ends strictly before validation cut; test history
starts on/after validation cut and target ends within observation surveillance.
Machine-model groups1/2/3 (16/17/35 machines) train; model4 (32) is strictly held out.
Held-out fitting, source selection, checkpoint/threshold tuning and adaptation are
forbidden. End-of-observation failure surveillance completeness is an assumption.

All 16 public coordinates remain in order across clients. Missing/constant local
coordinates are neutral zeros after training-only statistics; no held-out variation
selects columns. Optional explicit error/maintenance/metadata groups use the causal
original window/lag contract and require a new output/config, with research-ablation
labels. Event-time lag does not prove posting availability; event-enriched models
are not asserted deployment-safe. No posting times or causal failure claims are invented.

## NASA

FD001-FD004 retain the 26-column source schema: engine, cycle, 3 settings, 21 sensors.
Run-to-failure truth is last training cycle minus current cycle. Official test truth
adds the official engine's RUL offset to last observed test cycle minus cycle.
No truth cap applies to this reference. FD004 has 249 train and 248 test engines;
source README discrepancies remain preserved.

Source domains FD001/2/3 use the existing seed 42/20% validation-engine partition.
Official test engine identity includes the source file. Windows remain within each
engine, chronological, length 30/stride 10. Official test uses one final window per
engine; short trajectories have zero left-padding and explicit boolean masks.
Validation checkpoint selection uses the Phase 02.1 fixed cycle 50/100 prefixes;
full run-to-failure held-out windows are a separate correlated diagnostic.

Sorted 24 public coordinates never depend on local retention or target variation.
Each source fits N0/N1/N2 independently on unique rows of permitted training engines,
not repeated overlapping windows. N0 uses mean/std; N1 uses median/IQR with floor
`max(IQR,.05*(q99-q01),1e-6*max(abs(median),1))` and asinh; N2 uses that global N1
state and conditional sensor states for fixed setting bins[1,.1,1], minimum 128 fit
rows. Unsupported/undersupported bins fall back to source-global N1. Settings keep
global encoding. Constants/all-missing values stay zero, even under later variation.
No target fitting, distribution estimation, nearest-regime adaptation or feature
switching occurs. The dictionaries/statistics are private and are not model weights.

## Preservation and metadata

All 17 manifest files are hash/row verified before and after experiments. The manifest,
academic PDF and source documentation remain unchanged. A separate742-file snapshot
covers raw inputs and historical outputs; report actual after-verification results.
Offline metadata preparation uses the same validators and targets with N0 and fixed
coordinates; its generated numerical arrays are catalog artifacts, not an alternate
training framework. Experimental training uses private configured states directly.
Catalog preparation inspects held-out descriptive summaries and is scheduled only
after the experiment freeze/evaluation; neither summaries nor held-out tests tune fits.
