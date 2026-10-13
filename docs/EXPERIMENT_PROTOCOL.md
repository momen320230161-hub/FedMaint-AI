# Experiment Protocol — Draft

> **Not finalized:** This file separates requirements from proposed experimental choices. No final split, labels or metric scores exist yet.

## 1. Data tasks

| Track | Prediction unit | Proposed target | Candidate metrics |
| --- | --- | --- | --- |
| Azure | Machine at a timestamp (component-level extension TBD) | Any failure during `(t, t + 24 hours]`; **24h is a proposal** | PR-AUC, precision, recall, F1, calibration |
| NASA C-MAPSS | Engine at a cycle | Remaining useful life in **cycles** | RMSE, MAE, NASA asymmetric scoring function |

**Azure label rule (proposed):** At time `t`, use only observed readings and histories at or before `t`. Label 1 if the relevant asset has a failure strictly after `t` and no later than `t+24h`, else 0 when adequate future observation exists. Exclude windows whose future outcome is unknown at dataset boundaries. Decide whether post-failure/repaired observations require exclusion or a reset.

**NASA label rule:** In run-to-failure **training** sequences, the uncapped per-cycle RUL is `max_cycle_for_engine - current_cycle`; any label capping must be an explicit experimental decision. In the published **test** subset, supplied `RUL_FDxxx.txt` values correspond to the last observed cycle for each test engine. Evaluation on earlier cycles requires a documented reconstruction using the supplied endpoint RUL and observed-cycle offset.

## 2. Simulated factory definitions

- **Azure:** Four machine models offer a candidate non-IID grouping, but actual per-client class counts and useful train/test splits must be checked by EDA. Reserve at least one defined factory/client completely out of the global training and tuning process.
- **NASA:** FD001–FD004 represent distinct combinations of operating conditions and fault modes. Treat them as **domains** suitable for exploratory cross-domain testing, not real-world organizational factories. Select the hold-out before training, and report any support/domain-shift limitations.

**Rule:** Held-out factory examples (including unlabeled samples for fitted scaling) must not influence trained model parameters, normalization, training-time selection or hyperparameters. Report held-out evaluation only after the protocol is fixed.

## 3. Leakage prevention

- Keep all sequences from the same asset inside the intended split; preserve chronological direction when events may overlap.
- Fit missing-value policies, feature selection, imputers and scalers on the appropriate training subset **only**. Federated training uses **client-local statistics**, not shared raw values.
- Never join future maintenance events, future error codes, or outcomes into predictors at an earlier prediction time.
- Sliding windows must not cross machine/engine boundaries or accidentally mix train/validation/test records.
- Establish a clear final-horizon cutoff in Azure so labels do not rely on unobserved future events.
- Reconcile official NASA `train`, `test`, and `RUL` files; do not tune on official test labels.

## 4. Baseline comparisons

Use the **same backbone** and aligned data budgets where possible:

1. **Local-only** model per participating factory.
2. **Centralized reference** trained on pooled records **in an offline benchmark only** (not the proposed deployed privacy configuration).
3. **FedAvg** coordinated model.
4. **FedProx** coordinated model.
5. One federated model with an evaluated privacy mechanism; measure utility and overhead.

Record each run's seed, code revision, dataset manifest, client allocation, target definition, preprocessing settings, model/configuration, rounds, and runtime environment.

## 5. Privacy and explanation experiments

**Privacy:** Define the adversary and information exposed before claiming protection. For DP, report the actual privacy accountant outputs and utility tradeoff; for secure aggregation, document server visibility, trust assumptions and communication costs. These two techniques solve different problems and are not interchangeable guarantees.

**KG / LLM:** Evaluate grounding with/without retrieved graph facts. Every attributed maintenance claim needs a machine-readable evidence reference. Measure unsupported-claim rates and faithfulness, not only linguistic fluency. Do not imply a failure's physical root cause if the data only supports correlation.

## 6. Evaluation and reporting

- Azure: PR-AUC preferred for class imbalance, with threshold-based F1, precision and recall; specify how thresholds are chosen without test leakage.
- NASA: RMSE and the NASA asymmetric score; disclose whether RUL was capped.
- FL: per-client performance, global/held-out evaluation, communication bytes/round and total rounds.
- Generalization: evaluate the **unseen factory** separately and explain domain shift or label availability limitations.
- Reliability: multiple seeds, variance/intervals and appropriately selected significance testing.
- Ablations: without privacy, without graph grounding, and without LLM (when the ablation is meaningful).

## 7. Open items to finalize in Phase 1

- [ ] Azure failure scope: any component vs component-specific outcome.
- [ ] Prediction horizon, observation window, and prediction cadence.
- [ ] Time splits, client partitions, minimum event coverage, hold-out ID.
- [ ] NASA held-out-domain protocol and interpretation of official test sets.
- [ ] Baseline backbone and compute budget.
- [ ] Privacy threat model, mechanism and reporting plan.
- [ ] KG shared ontology and grounded-explanation rubric.

See [DECISIONS.md](DECISIONS.md) before treating any proposed value as final.
