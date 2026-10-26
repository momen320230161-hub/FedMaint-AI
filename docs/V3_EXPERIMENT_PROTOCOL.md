# V3 Phase 02 experiment protocol

Frozen before fitting on 2026-10-26. `configs/v3.json` is authoritative. No new
normalization ranking is performed. N2 is a candidate informed by historical
source-only evidence, not an established universal best method. Keep N0/N1 support.
The earlier source comparison and Azure event ablations are historical records in
PHASE021_RESULTS.md and PHASE02_1_RESEARCH_PLAN.md, not newly reproduced findings.

## Fixed panel and budget

Seeds42/43, source splits fixed at42. Per seed: Azure telemetry MLP for 3 local
clients plus pooled reference (4 fits); NASA N2 GRU and matched masked MLP each for
3 locals plus pooled (8 fits). Total 24 fits. No learning-rate/window/source search.
Model sizes: Azure16->32->1 (577 parameters); NASA GRU24->32->1 (5601);
NASA masked-window MLP750->32->1 (24065). Neural float32 payload sizes are2308,
22404,96260 bytes respectively, excluding transport overhead and private state.
These are potential payload dimensions, not observed FL communication.

Adam lr .001, batch 128, alpha .0001, maximum 40 epochs, patience 6, single CPU thread.
One epoch visits every permitted record. Pooled references have more steps and are
not matched compute-time experiments. Shared loss is BCE with logits for Azure;
NASA half MSE after dividing RUL by 100. Both add alpha/(2*Nfit) squared weight
penalty, excluding bias. No resampling/class weighting, BatchNorm or dropout.
Optional GPU support is explicit configuration, unverified on this CPU host.

Checkpoints maximize Azure validation AP or negative NASA macro prefix RMSE across
participating domains/cycles 50/100. Strict ties retain earliest epoch. Azure threshold
maximizes validation F1 with highest-threshold tie. Both classes must be present
for selection. Histories, actual selected epochs, sample counts, fit/validation
membership and keys digests are retained per run. Local epoch budgets are validated.

## Private state and evaluation

Local fits use exactly one source state. Pooled references concatenate locally
transformed source samples and permitted validation samples; no pooled scaler is
fitted. Zero-shot routing freezes the source with maximum informative training
coordinates, lexical ties; local fits use their own source. Known source-domain
official tests use each participant's own train-fitted transform.

All selected models, thresholds and private transforms and relevant source/config
files are sealed by hashes before evaluation. NASA official-test/FD004 loading
requires verification of that seal. Final evaluation has an exclusive marker.
Azure source CSV chunks are membership-filtered before sample construction; held-out
model4 feature generation occurs only after freeze. Hash verification may stream
held-out raw bytes without deriving training statistics. Source Azure chronological
test rows may be constructed by the shared causal builder but are not passed to fits,
selection or routing. Subsequent test execution is recorded after freeze.

Evaluate Azure source chronological tests and model4 separately. Report prevalence
(no-skill AP), AP, trapezoidal PR-AUC, F1, precision, recall, threshold and confusion
matrix[[TN,FP],[FN,TP]]. NASA official terminal pools receive RMSE, MAE, asymmetric
NASA **sum**, overflow/log/scientific representations, negative prediction rates and
error tails. Truth remains uncapped; outputs remain raw linear*100, no zero floor.
FD004 official248-engine pool and run-to-failure diagnostic never share a metric row.

Every evaluation saves raw numerical inputs/masks, source-scoped keys, truths and
predictions. Recompute predictions after safe numeric loading and compare exactly;
recompute every metric. Output is offline private research data. Source file hashes,
manifest evidence, environment, seeds and config accompany the run. Output directories
are exclusive; a failed run remains a failed record rather than being overwritten.

## Limits and approval boundaries

FD004/model4 were observed previously; revised held-out findings are retrospective.
No configuration is changed because of these outcomes. Two optimization seeds do
not establish statistical significance; simulated domains are not independent real
factories; per-window RUL rows are correlated. N2 fallback does not learn unseen
operating regimes. The MLP control can disprove broad claims of GRU superiority.
Different local normalization semantics remain a federated optimization risk.

FedAvg/FedProx and explicit privacy remain unimplemented. Tensor export/import and
sample-count contracts establish mechanical readiness only. Source scaler transfer
is not an approved privacy-preserving zero-shot deployment mechanism. No frontend,
KG, LLM, target redefinition, history rewrite or remote push is part of this task.
