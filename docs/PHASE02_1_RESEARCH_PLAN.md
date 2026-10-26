# Phase 02.1 research alignment and bounded optimization plan

Date: 2026-10-22. Status: **pre-implementation protocol and read-only source diagnostic**. No new predictor, normalization candidate, PyTorch installation, training run, or FD004 evaluation was performed for this document. Phase 02 remains an unchanged historical baseline. Phase 03, privacy mechanisms, KG, LLM, and UI remain outside scope.

## 1. Research objective and academic traceability

The primary objective is generalization to a previously unseen simulated factory under non-IID equipment/operating conditions, while fitting preprocessing locally on permitted training records. A lower previously observed FD004 score is not the selection objective. Do not rebrand retrospective benchmark improvement as blind generalization.

| Original Project2_Description.pdf | Consequence for Phase 02.1 |
| --- | --- |
| Page 1, steps 1-2: non-IID simulated clients, held-out factory, local normalization statistics | Diagnose cross-domain preprocessing failures; fit every transform privately inside its training client; use source-domain holdouts to select changes. No target-factory fitting or pooled normalization statistics. |
| Page 2, step 4: local-only and centralized references, two FL strategies, unseen-site evaluation | Keep Phase 02 references intact. Compare a shared, compact recurrent backbone with a matched MLP control before later FedAvg/FedProx work. Architecture examples in the PDF are not mandates. |
| Page 2, steps 4-5: only model updates leave FL clients; an explicit privacy mechanism and residual-leakage analysis are required | Compatible tensor shapes alone are insufficient. Keep fitted preprocessing out of exchanged model state; identify the deployment tension around frozen source scalers. Do not claim privacy from local scaling, partitioning, or ordinary FL. |
| Page 3, step 7: NASA RMSE/asymmetric score, Azure F1/PR-AUC, ablations, multiple runs, limitations | Report uncapped RUL errors, large failures and overflow faithfully; audit event availability and ablate Azure features; use a bounded, reproducible source-validation design. Two seeds are not a significance study. |

The original PDF is [Project2_Description.pdf](Project2_Description.pdf). Exact windows, candidate normalizers, GRU size, validation prefixes, and budgets below are proposed engineering decisions, not requirements attributed to that PDF.

## 2. What the existing evidence establishes

The Phase 02 record already reports catastrophic FD001/FD003-source inference on FD004 and a more stable FD002-source reference. Those outcomes are **already observed**. They motivate this investigation but must not select a normalizer, feature, regime router, hyperparameter, epoch, source transform, output postprocessing rule, or final architecture.

A new read-only audit used only FD001-FD003 **existing training-engine rows**: 80/208/80 engines and 16,779/43,083/20,074 rows, respectively. No labels, official test files, FD004 numeric file, FD004 prediction CSV, or FD004 statistics were loaded for this calculation. It reused the already fitted source scalers and transformed other permitted source domains' training rows, without fitting anything new.

| Frozen source scaler | Other source domain | 99th percentile of absolute standardized coordinates | Coordinates with absolute standardized value > 10 |
| --- | --- | ---: | ---: |
| FD001 | FD002 | 19,213.80 | 74.96% |
| FD003 | FD002 | 19,172.93 | 74.54% |
| FD002 | FD001 | 1.87 | 0.00% |
| FD002 | FD003 | 1.90 | 0.00% |
| FD001 | FD003 | 28.62 | 3.16% |
| FD003 | FD001 | 2.51 | 0.00% |

Each statistic above is over the target training rows and **the source scaler's retained coordinates**, so counts differ across sources; this is a coordinate-magnitude diagnostic, not an equally dimensioned performance comparison. Quantiles are descriptive and do not fit a target transform.

FD001's first operating-setting training range is [-0.0087, 0.0087], with fitted standard deviation 0.002186; FD002's corresponding source-training range is [0, 42.008]. FD001's second setting scale is 0.0002924 and its sensor-6 scale is 0.001391. The FD001-to-FD002 99th-percentile magnitudes reach approximately 19,217 for setting 1, 11,434 for sensor 6, and 6,704 for sensor 8. FD003 has similarly small operating-setting scales. The amplification affects sensors as well as settings: simply dropping operating settings is not an established solution.

**Supported diagnosis:** nearly constant source features and changes in operating-condition support create extreme standardized inputs. Heterogeneous local scaling is a plausible, directly observed contributor to extrapolation. ReLU/linear regressors are not constrained against large extrapolated outputs. The diagnostic does not establish that scaling is the sole cause of every FD004 error, nor that any proposed correction improves RUL accuracy. Label/censoring distributions, failure modes, sensor responses, and model behavior may also matter.

The audit is saved at `outputs/phase021-alignment/source-diagnostics.json`. Its method is: load only `train_FD001/2/3.txt`; restrict each to the Phase 01 training-engine IDs; apply `LocalPreprocessor.transform` using each different source's frozen state; summarize absolute values without using RUL. These are **not** leave-one-domain-out predictive-validation results.

## 3. Information boundaries and fold protocol

Use exactly three outer folds:

| Pseudo-held source domain | Permitted source training clients |
| --- | --- |
| FD001 | FD002, FD003 |
| FD002 | FD001, FD003 |
| FD003 | FD001, FD002 |

The entire pseudo-held domain is excluded from fitting, inner checkpoint selection, preprocessing statistics, constant detection, scale floors, regime dictionaries, fallback selection, and source-router selection in that fold. Do not reuse its Phase 01 fitted state. In particular, the FD002-held-out fold must not receive FD002's regime statistics from the other folds. Existing three-domain states or pooled Phase 02 checkpoints are not valid fold-fitted controls merely because they are frozen.

Within each permitted source client, retain deterministic engine-wise 80/20 fit/inner-validation membership, using seed 42 and the existing split function. Fit local transforms on unique fit-engine rows, never overlapping-window-weighted copies or inner-validation rows. Fit model weights on their training windows. Select checkpoints using only the two permitted clients' inner-validation engines. The outer pseudo-held domain supplies the score used to compare a **small predeclared candidate set**, not early stopping or per-fold tuning. Fold artifacts and statistics must have separate provenance namespaces.

Use fixed retrospective prefix endpoints **cycle 50 and cycle 100** from complete source-domain trajectories, with the existing 30-cycle history. Report the two endpoints separately; give each engine equal weight within each endpoint. Ground-truth RUL is final observed cycle minus prefix endpoint, never an input. Eligibility/missing endpoints must be reported; no endpoint may be chosen because its error looks favorable. These prefixes give one prediction per engine per endpoint and avoid treating thousands of overlapping windows as independent engines. They are controlled source-validation proxies, not a claim to reproduce official-test censoring. Whole-trajectory window metrics remain secondary diagnostics.

Primary candidate criterion: equally weighted mean RMSE across the two prefix endpoints and the three outer domains. Also report the worst-domain RMSE, per-fold MAE, per-endpoint NASA score sums, and per-engine prediction distributions. No larger domain is allowed to dominate the selection metric through record count. Model training retains a documented per-record objective for matched comparisons; do not silently change training-client weights between candidates.

Promote a change only if the confirmation run supports lower macro RMSE, no worsening of the worst-domain RMSE relative to the matched standard-normalization MLP control, and an improvement in at least two of three domain folds. If these criteria disagree, report a trade-off, not an unqualified improvement. No finite loss/metric may be substituted for a failed numerical run. Three source domains and reused selection folds provide limited evidence; after selecting a candidate their scores are not an independent unbiased test estimate.

## 4. Small normalization comparison

All candidates preserve factory-local fitted values. Use the same public NASA column schema and fixed coordinate order for three settings plus 21 sensors. Locally constant/all-missing coordinates remain neutral with their status recorded privately. Do not choose columns by FD004 variation, or fit medians/scales across clients. The shared schema is a public contract; the values of local fitted statistics are not public metadata.

| Candidate | Local fit and inference rule | What it tests |
| --- | --- | --- |
| N0: standard reference | Refit the existing train-only mean/std approach independently within each permitted fold/client. Preserve locally dropped coordinates as zeros in the shared 24-coordinate schema. | Proper source-fold reproduction of the existing failure-prone baseline. |
| N1: robust compressed | Fit each client's feature median and IQR on its fit engines. For nonconstant coordinates use a source-only stability floor and transform `asinh((x-median)/scale)`. No target clipping/refitting. | Whether robust location/scale and smooth input-tail compression reduce sensitivity to small fitted scales without censoring predicted RUL. |
| N2: condition-aware robust | Use predetermined operating-setting bins; fit per-bin sensor medians/IQRs on the local fit engines only. Apply the same floor/compression as N1. Unseen or undersupported bins fall back to that source client's global N1 state; log unsupported-regime rates. | Whether separating operating-condition baselines helps beyond the robust transformation itself. |

For the proposed N1/N2 floor, predeclare `max(IQR, 0.05*(q99-q01), 1e-6*max(abs(median),1))` per feature; every value in that expression is source-client-fit-only. Exact constants/all-missing columns remain neutral, rather than becoming artificial large signals. Do not tune this floor from pseudo-held observations or FD004. It is a hypothesis to test, not a demonstrated optimum. A plain IQR scaler can still amplify a nearly constant feature; robustness to training outliers alone does not establish robustness to operating-domain shifts. The median/IQR definition follows [RobustScaler documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.RobustScaler.html).

Proposed condition bins quantize the three raw settings at fixed resolutions (1.0, 0.1, 1.0). Keep separate dictionaries per source client; require at least 128 local fit rows before estimating bin-specific sensor statistics. Settings retain source-global N1 encoding, and sensors use conditional residual encoding when supported. No learned target centroids, target frequency estimates, global regime means, nearest unsupported-regime extrapolation, or per-domain neural head is introduced. Bin/floor choices must be frozen in the implementation config before scoring folds.

A single-condition source cannot learn the response at operating regimes it never observed. The fallback must expose this absence, not claim to remove it. Condition-aware preprocessing may fail the FD002-held-out fold; that is important evidence, not grounds to import FD002 statistics or exclude the fold. N1 versus N2 must otherwise use identical features, loss, output handling, seeds, and budgets to isolate the conditioning effect.

For pooled reference models, each training client applies its own private fitted transform, as in Phase 02. At a pseudo-held domain, freeze the inference source using the predeclared maximum-informative-training-coordinate rule with lexical ties, calculated **inside the fold**. Keep this rule identical across candidates; if a proposed strategy needs a different router, it is a separate unverified intervention. Never normalize the pseudo-held site with statistics fitted on its full trajectory or evaluation population.

## 5. Recurrent comparison and unchanged MLP evidence

Preserve all existing Phase 02 weights and reports as a frozen historical reference. Do not present them as leave-one-source-domain-out estimates: those models/contracts were developed with all three source domains available.

For a valid architecture comparison, refit a **matched MLP control** and a compact PyTorch GRU inside each fold under the same N0/N1/N2 transform. This prevents attributing a normalization improvement to a recurrent architecture. The MLP remains 750 -> 32 ReLU -> 1 with appended padding masks; the GRU uses 24 inputs, one unidirectional layer with hidden size 32, and a linear RUL head. Keep dropout zero and omit BatchNorm/client-specific neural branches. A GRU is the single recurrent candidate; do not add an LSTM/grid unless the bounded comparison establishes a reason for a later experiment.

Phase 01 arrays are **left padded**. Extract each sample's valid chronological rows before `pack_sequence(enforce_sorted=False)`, or explicitly repad valid sequences on the right before packing. Do not pass the original left-padded tensor with lengths and assume padding is ignored. Read the final valid hidden state, preserve sample order, and test equality with individually evaluated unpadded sequences. Pad-value mutations and length changes that preserve observations must not affect outputs. Refer to [PyTorch packing](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_sequence.html) and [GRU contracts](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html).

Retain the 30-cycle windows, engine identities, labels, and evaluation samples. Use Adam, learning rate 0.001, batch size 128, at most 40 epochs, patience 6, and fixed 100-cycle optimization scaling. State the loss explicitly and identically for new MLP/GRU controls: half mean squared error in scaled target units plus `alpha/(2*N_fit)*sum(weight**2)`, with alpha 0.0001 and no bias penalty. A PyTorch MLP port must first reproduce saved sklearn forward predictions; refits need not reproduce a different optimizer implementation bit-for-bit. Report parameter counts and time: equal epoch caps are not equal FLOP budgets.

New-model primary metrics use **raw linear-head predictions**, including negative predictions, with no upper cap or post-hoc floor. Record negative prediction rates and both tails. Phase 02 historically floored predictions at zero; preserve that record exactly and label it when comparing with new unmodified outputs. Do not change, clip, winsorize, or cap RUL predictions after seeing validation/FD004 errors. Raw uncapped RUL remains the evaluation truth, even if later training-label caps are studied separately.

With the fixed 24-input contract, a 32-unit one-layer GRU plus scalar head has 5,601 parameters, versus 24,065 for the 750-input/32-unit MLP. These are architectural counts, not measured accuracy or network-traffic results. The GRU is compact and mechanically promising for later model-update communication, but predictive superiority remains unverified.

## 6. Bounded experiment budget and freeze points

No training was run at this alignment stage. Proposed maximum implementation budget:

1. Screening: 3 normalizers x 2 architectures x 3 source folds x seed 42 = **18 fits**. No learning-rate/window/hidden-size search.
2. Confirmation: standard MLP plus a matched MLP/GRU pair under the shortlisted normalization, on the same 3 folds with seed 43 = **at most 9 additional fits**. If candidates coincide, reuse the fit. If confirmation fails the promotion criteria, retain the baseline and report no supported improvement; do not open another search.
3. Only after selection is frozen, fit the baseline and at most two confirmed comparators across three local clients and one pooled reference, seeds 42/43 = **at most 24 final fits**. Local-only and pooled comparisons must use the same frozen rules. No FD004 score selects which final models survive.
4. Azure audit: the 18 fixed fits below. No opportunistic extra search based on high or low scores.

Thus the upper bound is **69 small fits**, fewer when arms coincide, with a global 45-minute training wall-time cap and one CPU thread unless an explicit reproducible hardware profile is agreed before execution. The cap is a proposed stopping rule, not a runtime estimate. Preprocessing, audit, validation and any failed attempts are logged separately; incomplete arms remain incomplete. Do not weaken scientific gates or claim completion to fit the budget. A larger follow-up needs a new explicit plan.

Record config and source/data hashes, dependency versions including a project-local pinned PyTorch build, engine membership, local transform-fit rows, candidate/checkpoint histories, unsupported regimes, per-engine predictions, raw errors, metrics, and runtime. Use fresh `outputs/phase021/...` paths and a separate dependency lock extension; do not replace `outputs/phase02` or its archived dependency/provenance descriptions. Write an exposure/freeze log before any eventual FD004 scoring.

## 7. Raw-RUL evaluation and FD004 exposure

Keep RMSE and MAE in cycles and the original NASA asymmetric **sum**: for `d=prediction-raw_truth`, use `exp(-d/13)-1` for early `d<0`, otherwise `exp(d/10)-1`. Preserve log/scientific score representations and overflow flags, with no capped substitutes. Report sums separately for each endpoint/pool with its engine count; do not average away severe individual-engine errors. Engine-level error quantiles and maximum errors are required alongside means.

FD004 was already evaluated and inspected in Phase 02. Even with a rigorous new source-only selection rule, subsequent FD004 comparisons are **not fully blind**; prior observation may have influenced human hypotheses. The research report must state that limitation prominently. Freezing a new protocol reduces further leakage but does not erase prior exposure or create an independent unseen test set.

During Phase 02.1 selection, do not load FD004 data, statistics, predictions, metrics, or inferred regime distributions. Exclude them from source-selection helpers and candidate dashboards. The selection report must be producible with FD004 paths inaccessible. Prior published FD004 values may remain cited as historical motivation, never as a ranking criterion.

Only after source-only choices and hashes are frozen may an explicitly scheduled final evaluation reuse FD004. Run every predeclared final comparator and report all outcomes, including regressions. Keep the 248-engine official terminal pool separate from the correlated run-to-failure diagnostic. No target fitting, adaptation, source switching, feature changes, or retraining follows that evaluation. A new truly blind factory would be needed for stronger confirmatory generalization claims; its acquisition is outside this phase.

## 8. Azure availability and feature-group audit

Code inspection shows that targets use future failures, whereas telemetry features end at prediction time and errors/maintenance default to a one-hour availability lag. That is necessary but not sufficient evidence of real deployment availability: the source lacks ingestion/posting timestamps. A one-hour lag is an assumption, not measured assurance. Static age remains disabled by default.

Use four fixed feature groups: telemetry only; telemetry + errors; telemetry + maintenance; telemetry + errors + maintenance. Hold labels, time cuts, purged history/horizon support, machine membership, sample timestamps, model architecture/budget, and seeds fixed. **Do not compare different sample populations** because one ablation happens to drop more windows. Fit feature selection/imputation/scaling anew on each allowed training partition. Compare AP and trapezoidal PR-AUC, F1 with validation-only thresholds, precision/recall, prevalence, and confusion matrices. Include a prevalence/no-skill reference and flag the dataset's synthetic nature.

Use leave-one-training-machine-model-out folds across model1/model2/model3, with chronological fitting/inner validation within the two permitted model groups; the pseudo-held model contributes no fit statistics or threshold choice. No model4 result selects a feature group or lag. Four groups x three folds x seed 42 = **12 fits**. Add full-feature lag stress tests at 6 and 12 hours, keeping the original 24-hour history and same sample rows, for **6 further fits**. Do not turn lag into an unbounded optimization grid. These are sensitivity checks, not proof of real event availability.

Audit the timing of errors/maintenance relative to prediction points and failures, duplicate/event boundary handling, maintenance logging versus actual action time, and the daily sampling grid. A large advantage from maintenance features may reflect genuinely known history, simulator construction, or unavailable-at-prediction annotations; ablation magnitude alone does not identify which. Keep exact-time exclusion, future-event mutation tests, and adversarial backdated-posting scenarios. Where source availability cannot be established, designate telemetry-only as the conservative reference rather than claiming the full-feature score is deployment-valid.

No Azure ablations or larger-lag experiments have been executed at this alignment stage. Existing high PR-AUC is neither invalidated nor established as leakage-free by this document.

## 9. Federation and privacy compatibility assessment

| Design aspect | Current assessment |
| --- | --- |
| Common feature order, recurrent architecture and parameter names/shapes | Suitable in principle for subsequent FedAvg/FedProx; implementation and aggregation tests are still required. |
| Local statistics/regime dictionaries kept outside exchanged neural state | Consistent with local-only normalization, provided the future system actually enforces this boundary. Do not average local means, medians, scales or per-regime dictionaries. |
| Heterogeneous local coordinate meanings | Unresolved scientific issue. Common dimensions permit arithmetic aggregation, but do not prove equivalent feature semantics or good federated generalization. Source-only normalization comparisons are needed. |
| Source-trained zero-shot transforms | Valid offline research controls, but redistributing fitted source statistics is **not automatically compatible** with the strict 'only model updates leave clients' deployment description. A trusted local evaluation harness does not solve real unseen-factory deployment. |
| Centralized pooled references | Explicitly non-private experimental controls, distinct from the federated data boundary. |
| Formal privacy | Not satisfied by the current baselines or proposed GRU. The PDF still requires an explicit future mechanism, threat model, accounting/assumptions as appropriate, and residual-leakage analysis. |

In a later federated implementation, aggregate only a shared backbone's parameters, with a documented client-weighting rule and FedProx reference parameters; keep preprocessing private and outside the proximal penalty/aggregation state. A condition-aware transform must always output the same fixed feature contract, without client-specific condition heads or permuted category coordinates.

The zero-shot deployment tension must remain visible: exporting source scalers can disclose source distribution information, sending raw target data to a source would violate the intended raw-data boundary, and fitting target-site scalers is adaptation rather than the strict zero-shot protocol. Do not rename scaler statistics as 'model updates' to evade this issue. A future threat-model-approved deployment protocol or separate adaptation experiment must resolve it. No such protocol or privacy mechanism is implemented here.

**Conclusion:** the proposed compact shared backbone is mechanically compatible with future FL and the local-fitting requirement, conditional on these boundaries. The overall design is not yet a complete implementation of the PDF's federated/privacy requirements, and compatibility of strict zero-shot serving remains unverified.

## 10. Evidence status, preservation, and checks

| Claim | Status as of this alignment |
| --- | --- |
| Small fitted scales plus source operating-condition differences can produce extreme standardized inputs | Supported by the newly executed source-only numerical audit. |
| Phase 02 weights/metrics/checkpoints are reproducible under their recorded protocol | Existing Phase 02 verification evidence; not a new leave-one-domain-out result. |
| Robust compressed normalization improves source-domain RUL accuracy | Unverified; candidate N1 has not been trained or validated. |
| Condition-aware normalization improves source-domain generalization | Unverified; candidate N2 and unsupported-regime fallback require the folds above. |
| GRU improves over a properly matched MLP | Unverified; PyTorch comparison not implemented/run. |
| Source-only-selected changes improve FD004 | Unverified, and any later evaluation is not fully blind. |
| Azure high AP is robust to unavailable events or feature removal | Unverified; code alignment and a one-hour lag do not substitute for availability evidence/ablations. |
| The full proposed system satisfies the academic privacy requirement | Not established; the explicit mechanism and zero-shot deployment policy remain future work. |

This turn creates only this protocol and ignored read-only audit artifacts under `outputs/phase021-alignment/`. Before edits, SHA-256 was recorded for 344 existing files covering raw inputs, Phase 02 artifacts, original documentation, configs, implementation, and tests. The after-check must confirm every recorded file remains byte-identical. The original Phase 02 result generator must not be rerun over its report or prior output directory during this alignment.

The executed audit used source training rows and frozen transformations only; no new model/preprocessing fits or performance experiments occurred. Existing test counts remain historical Phase 02 evidence, not claimed rerun results for this documentation task. Check new links and `git diff --check`; do not invent a Phase 02.1 completion or new accuracy claim. Phase 03 remains unstarted.
