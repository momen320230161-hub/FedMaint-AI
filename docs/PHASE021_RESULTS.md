# Phase 02.1 measured results

Executed 2026-10-22–23 on main. All values derive from saved experiments; no project beyond Phase 02.1 was implemented.

FD004 had already been observed in Phase 02: this evaluation is **not fully blind**. FD001–FD003 selection folds are exploratory and reused for selection/confirmation, not independent unbiased generalization estimates. Two seeds measure optimization variability, not statistical significance.

**Execution deviation:** the historical full-suite preparation/smoke tests also accessed FD004/model4 before this freeze. Their held-out metrics were not inspected or used for selection; the new candidate loader remained gated. Nevertheless the strict no-access-before-freeze rule was not fully met. See outputs/phase021/protocol-deviations.json. The new candidate panel itself was evaluated exactly once after freeze. Do not describe this execution as protocol-perfect.

**69 completed research fits**, 488.25 seconds charged fitting/selection/artifact time, within 69 fits / 2700 seconds. No failed, omitted, or retried research candidate. Test-only fits are excluded.

## Architectures and fixed protocol

MLP: flatten 30 x 24 observations, append 30 boolean validity masks, Linear(750,32), ReLU, Linear(32,1): **24,065 trainable parameters**. GRU: 24 inputs, one unidirectional 32-unit layer, no dropout, final valid hidden state, Linear(32,1): **5,601 parameters**. Left padding is stripped before packing; tests compare mixed-length batches with individual unpadded sequences.

Both use Adam lr=0.001, batch=128, alpha=0.0001, 40 epochs maximum, patience=6, deterministic CPU/one thread; target divided by100 only for optimization. Loss: half MSE plus alpha/(2*Nfit) times the sum of squared weight tensors (no bias penalty). Raw uncapped truth is used everywhere. Checkpoints use inner permitted-source validation engines at cycles50/100. Equal epoch limits do not imply equal FLOPs.

N0 is local train-only mean/std with neutral constant coordinates. N1 is local median/IQR with the predeclared q01/q99 stability floor and asinh compression. N2 adds local condition-bin sensor statistics at fixed resolutions[1,0.1,1], minimum128 rows/bin; unsupported bins fall back to source-global N1. All three output the fixed sorted24-coordinate schema. The full approved method is unchanged in [PHASE02_1_RESEARCH_PLAN.md](PHASE02_1_RESEARCH_PLAN.md).

Primary new-model outputs are raw, including negatives. Zero-floored metrics are separate diagnostics. There is no upper prediction cap or ground-truth cap. NASA score is the official asymmetric **sum**; overflow is retained as null plus natural-log/scientific representations.

## Source-only normalization and architecture selection

Shortlisted normalization: **N2**. Selected configuration: **N2-gru**. Candidates passing all promotion gates in both seeds: `['N2-gru', 'N2-mlp']`. The gate requires lower macro RMSE, no worse worst-domain RMSE, and improvement in at least2/3 domains versus N0-MLP. A normalization shortlist is not itself evidence of a supported improvement.

| Seed | Candidate | Macro RMSE | Worst domain | FD001 | FD002 | FD003 |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | N0-mlp | 24546 | 73455 | 65.764 | 73455 | 116.67 |
| 42 | N0-gru | 81.003 | 120.82 | 57.451 | 64.74 | 120.82 |
| 42 | N1-mlp | 329.96 | 840.25 | 61.832 | 840.25 | 87.8 |
| 42 | N1-gru | 70.866 | 108.3 | 47.254 | 57.044 | 108.3 |
| 42 | N2-mlp | 321.38 | 840.25 | 47.957 | 840.25 | 75.941 |
| 42 | N2-gru | 63.998 | 88.915 | 46.034 | 57.044 | 88.915 |
| 43 | N0-mlp | 32674 | 97823 | 78.822 | 97823 | 118.94 |
| 43 | N2-gru | 69.461 | 89.565 | 49.159 | 69.659 | 89.565 |
| 43 | N2-mlp | 314.81 | 812.23 | 52.036 | 812.23 | 80.177 |

### Matched output-policy and normalization checks

| Seed | Candidate | Raw macro RMSE (selection) | Zero-floor macro RMSE (diagnostic only) |
| --- | --- | --- | --- |
| 42 | N0-gru | 81.003 | 81.003 |
| 42 | N0-mlp | 24546 | 24526 |
| 42 | N1-gru | 70.866 | 70.865 |
| 42 | N1-mlp | 329.96 | 329.96 |
| 42 | N2-gru | 63.998 | 63.997 |
| 42 | N2-mlp | 321.38 | 321.38 |
| 43 | N0-mlp | 32674 | 32668 |
| 43 | N2-gru | 69.461 | 69.451 |
| 43 | N2-mlp | 314.81 | 314.81 |

N2-MLP versus N0-MLP isolates the normalization change within the new MLP protocol and passes the source promotion gates in both seeds, yet retains severe FD002-held-out errors. N2-GRU versus N2-MLP is a matched architecture comparison: GRU has lower RMSE in all three folds in both seeds. N2-GRU versus N1-GRU is supported only by the screening seed; the separate benefit of condition-aware bins beyond N1 has no second-seed confirmation in this bounded plan. N0-GRU also has only one screening seed. Do not extrapolate a universal ordering or attribute historical-versus-new differences solely to the GRU.

### N2 outer-domain unsupported-regime diagnostics

| Seed | Architecture | Held domain | Frozen source | Unsupported rows / all rows |
| --- | --- | --- | --- | --- |
| 42 | mlp | FD001 | FD002 | 0/20631 |
| 42 | mlp | FD002 | FD003 | 45715/53759 |
| 42 | mlp | FD003 | FD002 | 0/24720 |
| 42 | gru | FD001 | FD002 | 0/20631 |
| 42 | gru | FD002 | FD003 | 45715/53759 |
| 42 | gru | FD003 | FD002 | 0/24720 |
| 43 | gru | FD001 | FD002 | 0/20631 |
| 43 | gru | FD002 | FD003 | 45715/53759 |
| 43 | gru | FD003 | FD002 | 0/24720 |
| 43 | mlp | FD001 | FD002 | 0/20631 |
| 43 | mlp | FD002 | FD003 | 45715/53759 |
| 43 | mlp | FD003 | FD002 | 0/24720 |

Each fold averages cycle50/100 RMSE, then folds receive equal weight. Pseudo-held clients never fit transforms/checkpoints or source routing. Permitted clients retain deterministic80/20 engine splits; unique rows fit local statistics. Training retains the same per-record objective. Normalization/architecture/floor/window/router constants were not changed from these outcomes.

### Per-fold, per-endpoint metrics

Each row is one prediction per eligible engine at a fixed endpoint. Maximum errors, error p95, predictions, negative fractions and endpoint eligibility are retained in selection.json and per-run CSV files.

| Seed | Candidate | Held domain | Cycle | n | Raw RMSE | Raw MAE | Raw NASA sum | Negative fraction | Floor RMSE diagnostic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | N0-mlp | FD001 | 50 | 100 | 81.987 | 68.919 | 3.442871894e+7 | 0 | 81.987 |
| 42 | N0-mlp | FD001 | 100 | 100 | 49.541 | 35.364 | 9.683313648e+5 | 0 | 49.541 |
| 42 | N0-mlp | FD002 | 50 | 260 | 74322 | 66605 | 2.209406399e+7016 | 0.030769 | 74264 |
| 42 | N0-mlp | FD002 | 100 | 260 | 72589 | 64121 | 1.158962004e+6882 | 0.034615 | 72528 |
| 42 | N0-mlp | FD003 | 50 | 100 | 133.89 | 106.31 | 3.921805602e+12 | 0 | 133.89 |
| 42 | N0-mlp | FD003 | 100 | 100 | 99.447 | 64.696 | 8.045130667e+10 | 0 | 99.447 |
| 42 | N0-gru | FD001 | 50 | 100 | 71.221 | 56.877 | 1.221354287e+7 | 0 | 71.221 |
| 42 | N0-gru | FD001 | 100 | 100 | 43.681 | 30.87 | 3.248110382e+5 | 0 | 43.681 |
| 42 | N0-gru | FD002 | 50 | 260 | 80.628 | 66.188 | 4.624062115e+8 | 0 | 80.628 |
| 42 | N0-gru | FD002 | 100 | 260 | 48.851 | 35.415 | 1.53942565e+6 | 0 | 48.851 |
| 42 | N0-gru | FD003 | 50 | 100 | 138.81 | 114.43 | 4.617309834e+12 | 0 | 138.81 |
| 42 | N0-gru | FD003 | 100 | 100 | 102.82 | 70.525 | 8.491457863e+10 | 0 | 102.82 |
| 42 | N1-mlp | FD001 | 50 | 100 | 49.707 | 42.226 | 9.623237559e+4 | 0 | 49.707 |
| 42 | N1-mlp | FD001 | 100 | 100 | 73.958 | 67.333 | 1.96621911e+6 | 0 | 73.958 |
| 42 | N1-mlp | FD002 | 50 | 260 | 817.37 | 792.85 | 4.950829218e+56 | 0 | 817.37 |
| 42 | N1-mlp | FD002 | 100 | 260 | 863.14 | 841.69 | 4.825276249e+58 | 0 | 863.14 |
| 42 | N1-mlp | FD003 | 50 | 100 | 97.134 | 63.011 | 4.977955769e+10 | 0 | 97.134 |
| 42 | N1-mlp | FD003 | 100 | 100 | 78.465 | 57.009 | 1.000638129e+9 | 0 | 78.465 |
| 42 | N1-gru | FD001 | 50 | 100 | 45.126 | 32.891 | 2.892420668e+5 | 0 | 45.126 |
| 42 | N1-gru | FD001 | 100 | 100 | 49.383 | 43.015 | 7.503373761e+4 | 0 | 49.383 |
| 42 | N1-gru | FD002 | 50 | 260 | 63.289 | 48.244 | 3.174557706e+7 | 0 | 63.289 |
| 42 | N1-gru | FD002 | 100 | 260 | 50.798 | 40.581 | 6.401694391e+5 | 0.0038462 | 50.796 |
| 42 | N1-gru | FD003 | 50 | 100 | 125.02 | 99.01 | 8.91961999e+11 | 0 | 125.02 |
| 42 | N1-gru | FD003 | 100 | 100 | 91.58 | 60.529 | 1.413569615e+10 | 0 | 91.58 |
| 42 | N2-mlp | FD001 | 50 | 100 | 54.674 | 43.233 | 4.123346428e+5 | 0 | 54.674 |
| 42 | N2-mlp | FD001 | 100 | 100 | 41.239 | 32.185 | 1.579863018e+5 | 0 | 41.239 |
| 42 | N2-mlp | FD002 | 50 | 260 | 817.37 | 792.85 | 4.950829218e+56 | 0 | 817.37 |
| 42 | N2-mlp | FD002 | 100 | 260 | 863.14 | 841.69 | 4.825276249e+58 | 0 | 863.14 |
| 42 | N2-mlp | FD003 | 50 | 100 | 80.956 | 52.621 | 4.15487504e+9 | 0 | 80.956 |
| 42 | N2-mlp | FD003 | 100 | 100 | 70.926 | 49.148 | 3.364740267e+8 | 0 | 70.926 |
| 42 | N2-gru | FD001 | 50 | 100 | 50.707 | 38.129 | 3.855775763e+5 | 0 | 50.707 |
| 42 | N2-gru | FD001 | 100 | 100 | 41.361 | 33.022 | 3.128648954e+4 | 0 | 41.361 |
| 42 | N2-gru | FD002 | 50 | 260 | 63.289 | 48.244 | 3.174557706e+7 | 0 | 63.289 |
| 42 | N2-gru | FD002 | 100 | 260 | 50.798 | 40.581 | 6.401694391e+5 | 0.0038462 | 50.796 |
| 42 | N2-gru | FD003 | 50 | 100 | 99.875 | 62.7 | 1.656819689e+12 | 0 | 99.875 |
| 42 | N2-gru | FD003 | 100 | 100 | 77.955 | 49.583 | 3.613857986e+10 | 0 | 77.955 |
| 43 | N0-mlp | FD001 | 50 | 100 | 97.98 | 86.956 | 1.521154687e+8 | 0 | 97.98 |
| 43 | N0-mlp | FD001 | 100 | 100 | 59.664 | 44.68 | 3.744376468e+6 | 0 | 59.664 |
| 43 | N0-mlp | FD002 | 50 | 260 | 97722 | 87877 | 4.418230835e+9354 | 0.011538 | 97708 |
| 43 | N0-mlp | FD002 | 100 | 260 | 97925 | 87995 | 2.790884392e+9509 | 0.026923 | 97906 |
| 43 | N0-mlp | FD003 | 50 | 100 | 136.28 | 108.03 | 5.73309566e+12 | 0 | 136.28 |
| 43 | N0-mlp | FD003 | 100 | 100 | 101.59 | 66.291 | 1.147931113e+11 | 0 | 101.59 |
| 43 | N2-gru | FD001 | 50 | 100 | 54.547 | 41.899 | 4.554145806e+5 | 0 | 54.547 |
| 43 | N2-gru | FD001 | 100 | 100 | 43.77 | 34.673 | 3.652658349e+4 | 0 | 43.77 |
| 43 | N2-gru | FD002 | 50 | 260 | 57.617 | 47.766 | 9.413351577e+5 | 0 | 57.617 |
| 43 | N2-gru | FD002 | 100 | 260 | 81.701 | 70.906 | 6.154854331e+7 | 0.0038462 | 81.641 |
| 43 | N2-gru | FD003 | 50 | 100 | 100.91 | 63.989 | 8.680700716e+12 | 0 | 100.91 |
| 43 | N2-gru | FD003 | 100 | 100 | 78.217 | 49.338 | 1.019714081e+11 | 0 | 78.217 |
| 43 | N2-mlp | FD001 | 50 | 100 | 59.271 | 46.811 | 9.998487212e+5 | 0 | 59.271 |
| 43 | N2-mlp | FD001 | 100 | 100 | 44.8 | 35.592 | 7.539817332e+5 | 0 | 44.8 |
| 43 | N2-mlp | FD002 | 50 | 260 | 792.33 | 763.74 | 1.061648678e+60 | 0 | 792.33 |
| 43 | N2-mlp | FD002 | 100 | 260 | 832.13 | 807.12 | 1.929973266e+59 | 0 | 832.13 |
| 43 | N2-mlp | FD003 | 50 | 100 | 84.937 | 54.154 | 3.986943313e+10 | 0 | 84.937 |
| 43 | N2-mlp | FD003 | 100 | 100 | 75.417 | 50.65 | 2.808972165e+9 | 0 | 75.417 |

## Final frozen NASA evaluations

Freeze SHA-256: `1b3592442f9548ab99b6a732498b0b606bdc5b967b3e09d5434cbdbaea7f90aa`. Final model panel: `['N0-mlp', 'N2-gru', 'N2-mlp']`. All source-only models, private transforms and routing choices were sealed before any new official-test/FD004 loading. The exclusive evaluation marker prevents silent repeats. Final outcomes did not change the winner.

Local models use their own transform. Pooled known-domain tests use that training client's transform; zero-shot pools use the frozen maximum-informative-training-coordinate source (lexical ties), independent of target statistics. Official terminal pools and correlated run-to-failure windows are kept separate.

### FD004 official terminal pool: per-seed results

All rows use the same248 engines and raw uncapped truth. Floor diagnostics use the same zero floor as historical Phase02, but other training/selection protocols still differ.

| Candidate | Scope | Seed | Source | Raw RMSE | Raw MAE | Raw NASA sum | Negative fraction | Floor RMSE | Floor MAE | Floor NASA sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N0-mlp | FD001 | 42 | FD001 | 35811 | 29550 | 1.950364271e+4677 | 0.59274 | 20888 | 10396 | 1.950364271e+4677 |
| N0-mlp | FD001 | 43 | FD001 | 87410 | 73584 | 6.590488211e+7958 | 0.89919 | 12289 | 3242.4 | 4.449197496e+3534 |
| N0-mlp | FD002 | 42 | FD002 | 54.341 | 43.895 | 1.316920241e+7 | 0.016129 | 54.278 | 43.794 | 1.316916438e+7 |
| N0-mlp | FD002 | 43 | FD002 | 58.659 | 47.476 | 5.483292974e+7 | 0.016129 | 58.617 | 47.329 | 5.483291798e+7 |
| N0-mlp | FD003 | 42 | FD003 | 1.1224e+05 | 98664 | 3.943824208e+11932 | 0.0080645 | 1.1223e+05 | 98541 | 3.943824208e+11932 |
| N0-mlp | FD003 | 43 | FD003 | 1.4958e+05 | 1.3604e+05 | 2.531935497e+13432 | 0.012097 | 1.4956e+05 | 1.3585e+05 | 2.531935497e+13432 |
| N0-mlp | pooled | 42 | FD002 | 53.427 | 44.213 | 1.083741932e+7 | 0.0040323 | 53.421 | 44.198 | 1.083741753e+7 |
| N0-mlp | pooled | 43 | FD002 | 52.029 | 43.1 | 9.066214491e+5 | 0 | 52.029 | 43.1 | 9.066214491e+5 |
| N2-gru | FD001 | 42 | FD001 | 55.97 | 44.881 | 2.899645682e+5 | 0 | 55.97 | 44.881 | 2.899645682e+5 |
| N2-gru | FD001 | 43 | FD001 | 52.056 | 41.457 | 2.542628657e+6 | 0.0040323 | 52.054 | 41.447 | 2.542628252e+6 |
| N2-gru | FD002 | 42 | FD002 | 61.153 | 45.489 | 8.334174262e+6 | 0 | 61.153 | 45.489 | 8.334174262e+6 |
| N2-gru | FD002 | 43 | FD002 | 72.954 | 53.744 | 6.9654062e+7 | 0.020161 | 72.95 | 53.717 | 6.965406081e+7 |
| N2-gru | FD003 | 42 | FD003 | 130.49 | 119.73 | 2.201623383e+10 | 0 | 130.49 | 119.73 | 2.201623383e+10 |
| N2-gru | FD003 | 43 | FD003 | 153.61 | 144.74 | 2.907532264e+11 | 0 | 153.61 | 144.74 | 2.907532264e+11 |
| N2-gru | pooled | 42 | FD002 | 36.891 | 27.358 | 1.661255972e+6 | 0.016129 | 36.865 | 27.308 | 1.661251702e+6 |
| N2-gru | pooled | 43 | FD002 | 32.66 | 26.036 | 2.354015226e+4 | 0 | 32.66 | 26.036 | 2.354015226e+4 |
| N2-mlp | FD001 | 42 | FD001 | 257.82 | 247.31 | 6.634086059e+18 | 0 | 257.82 | 247.31 | 6.634086059e+18 |
| N2-mlp | FD001 | 43 | FD001 | 175.02 | 158.25 | 3.039757753e+16 | 0.0040323 | 175.02 | 158.25 | 3.039757753e+16 |
| N2-mlp | FD002 | 42 | FD002 | 98.501 | 72.464 | 7.218247336e+11 | 0.0040323 | 98.5 | 72.454 | 7.218247336e+11 |
| N2-mlp | FD002 | 43 | FD002 | 112.34 | 81.264 | 2.006025219e+12 | 0.032258 | 112.3 | 81.019 | 2.006025219e+12 |
| N2-mlp | FD003 | 42 | FD003 | 718.43 | 645.58 | 3.816215063e+80 | 0 | 718.43 | 645.58 | 3.816215063e+80 |
| N2-mlp | FD003 | 43 | FD003 | 974.04 | 947.26 | 4.44570105e+68 | 0 | 974.04 | 947.26 | 4.44570105e+68 |
| N2-mlp | pooled | 42 | FD002 | 49.809 | 37.389 | 5.451163656e+9 | 0.012097 | 49.803 | 37.355 | 5.451163655e+9 |
| N2-mlp | pooled | 43 | FD002 | 48.754 | 36.641 | 2.23326316e+8 | 0.052419 | 48.536 | 36.055 | 2.233262648e+8 |

### Historical versus optimized NASA

Historical Phase02 uses zero-floored predictions, a different optimizer/hyperparameter search and window-based validation; Phase02.1 uses fixed learning rate and prefix-based validation. These comparisons involve different protocols and cannot isolate architecture/normalization effects. Matched Phase02.1 source comparisons above support attribution. Historical all-three-seed results remain unchanged in PHASE02_RESULTS.md; this table matches seeds42/43 only.

| Protocol/candidate | Scope | Domain | RMSE mean +/- SD | MAE mean +/- SD | NASA sum mean +/- SD |
| --- | --- | --- | --- | --- | --- |
| Phase02 historical floor | pooled | FD001 | 36.369 +/- 0.4768 | 26.411 +/- 0.2036 | 4.4214e+05 +/- 2.398e+05 |
| Phase02 historical floor | pooled | FD002 | 51.436 +/- 0.3235 | 41.286 +/- 0.738 | 1.8381e+06 +/- 1.243e+06 |
| Phase02 historical floor | pooled | FD003 | 47.251 +/- 1.014 | 33.65 +/- 0.411 | 9.8965e+07 +/- 1.337e+08 |
| Phase02 historical floor | pooled | FD004 | 58.987 +/- 0.2321 | 47.291 +/- 0.1418 | 4.0885e+07 +/- 4.713e+07 |
| Phase02 historical floor | FD001 | FD001 | 39.194 +/- 8.02 | 29.446 +/- 7.311 | 3.6706e+06 +/- 4.947e+06 |
| Phase02 historical floor | FD001 | FD004 | 49507 +/- 6.845e+04 | 31030 +/- 4.361e+04 | overflow; see per-seed score/log |
| Phase02 historical floor | FD002 | FD002 | 48.381 +/- 1.843 | 37.764 +/- 1.778 | 1.4837e+06 +/- 1.404e+06 |
| Phase02 historical floor | FD002 | FD004 | 58.09 +/- 2.865 | 45.954 +/- 1.542 | 2.3131e+08 +/- 2.8e+08 |
| Phase02 historical floor | FD003 | FD003 | 45.583 +/- 6.39 | 35.309 +/- 5 | 1.8482e+06 +/- 2.042e+06 |
| Phase02 historical floor | FD003 | FD004 | 1.5601e+05 +/- 7.715e+04 | 1.223e+05 +/- 8.163e+04 | overflow; see per-seed score/log |
| N0-mlp raw | FD001 | FD001 | 27.124 +/- 0.2691 | 20.565 +/- 1.165 | 9961.4 +/- 853 |
| N0-mlp raw | FD001 | FD004 | 61610 +/- 3.649e+04 | 51567 +/- 3.114e+04 | overflow; see per-seed score/log |
| N0-mlp raw | FD002 | FD002 | 42.549 +/- 1.605 | 33.508 +/- 0.1676 | 3.3624e+05 +/- 2.416e+05 |
| N0-mlp raw | FD002 | FD004 | 56.5 +/- 3.053 | 45.685 +/- 2.532 | 3.4001e+07 +/- 2.946e+07 |
| N0-mlp raw | FD003 | FD003 | 43.498 +/- 1.346 | 32.734 +/- 1.38 | 2.7663e+06 +/- 2.567e+06 |
| N0-mlp raw | FD003 | FD004 | 1.3091e+05 +/- 2.64e+04 | 1.1735e+05 +/- 2.643e+04 | overflow; see per-seed score/log |
| N0-mlp raw | pooled | FD001 | 28.821 +/- 1.587 | 22.467 +/- 0.8818 | 20756 +/- 2.038e+04 |
| N0-mlp raw | pooled | FD002 | 47.519 +/- 1.373 | 38.651 +/- 0.8244 | 8.3375e+05 +/- 2.741e+05 |
| N0-mlp raw | pooled | FD003 | 39.659 +/- 2.822 | 29.936 +/- 0.376 | 5.8532e+05 +/- 7.662e+05 |
| N0-mlp raw | pooled | FD004 | 52.728 +/- 0.9891 | 43.656 +/- 0.7868 | 5.872e+06 +/- 7.022e+06 |
| N2-gru raw | FD001 | FD001 | 24.492 +/- 1.652 | 17.832 +/- 1.138 | 3495.7 +/- 1266 |
| N2-gru raw | FD001 | FD004 | 54.013 +/- 2.768 | 43.169 +/- 2.422 | 1.4163e+06 +/- 1.593e+06 |
| N2-gru raw | FD002 | FD002 | 26.859 +/- 0.2069 | 19.36 +/- 0.4555 | 10006 +/- 115.3 |
| N2-gru raw | FD002 | FD004 | 67.053 +/- 8.345 | 49.616 +/- 5.838 | 3.8994e+07 +/- 4.336e+07 |
| N2-gru raw | FD003 | FD003 | 46.761 +/- 1.758 | 36.112 +/- 1.711 | 9.1522e+06 +/- 3.329e+06 |
| N2-gru raw | FD003 | FD004 | 142.05 +/- 16.35 | 132.23 +/- 17.68 | 1.5638e+11 +/- 1.9e+11 |
| N2-gru raw | pooled | FD001 | 24.089 +/- 0.3406 | 17.341 +/- 0.6959 | 4707.8 +/- 548.5 |
| N2-gru raw | pooled | FD002 | 26.608 +/- 0.2048 | 18.627 +/- 0.321 | 12538 +/- 3720 |
| N2-gru raw | pooled | FD003 | 39.161 +/- 2.818 | 25.419 +/- 2.251 | 1.2102e+08 +/- 1.463e+08 |
| N2-gru raw | pooled | FD004 | 34.776 +/- 2.992 | 26.697 +/- 0.9349 | 8.424e+05 +/- 1.158e+06 |
| N2-mlp raw | FD001 | FD001 | 27.503 +/- 2.12 | 21.505 +/- 0.8636 | 5679.5 +/- 2125 |
| N2-mlp raw | FD001 | FD004 | 216.42 +/- 58.55 | 202.78 +/- 62.97 | 3.3322e+18 +/- 4.67e+18 |
| N2-mlp raw | FD002 | FD002 | 30.002 +/- 0.3445 | 22.03 +/- 0.2523 | 23771 +/- 4152 |
| N2-mlp raw | FD002 | FD004 | 105.42 +/- 9.783 | 76.864 +/- 6.223 | 1.3639e+12 +/- 9.081e+11 |
| N2-mlp raw | FD003 | FD003 | 42.209 +/- 0.6004 | 32.535 +/- 0.2688 | 6.4868e+06 +/- 1.933e+06 |
| N2-mlp raw | FD003 | FD004 | 846.23 +/- 180.7 | 796.42 +/- 213.3 | 1.9081e+80 +/- 2.698e+80 |
| N2-mlp raw | pooled | FD001 | 30.768 +/- 0.9432 | 23.046 +/- 1.162 | 22380 +/- 1.006e+04 |
| N2-mlp raw | pooled | FD002 | 30.207 +/- 0.4977 | 22.413 +/- 0.03526 | 52764 +/- 7062 |
| N2-mlp raw | pooled | FD003 | 40.153 +/- 0.4891 | 28.334 +/- 0.1007 | 2.1169e+07 +/- 1.282e+07 |
| N2-mlp raw | pooled | FD004 | 49.282 +/- 0.746 | 37.015 +/- 0.5291 | 2.8372e+09 +/- 3.697e+09 |

### All final NASA pools and per-seed variability

Sample SD above is across two seeds. The following table and nasa-metrics.csv preserve every known-domain and run-to-failure result; correlated windows are not independent engines.

| Candidate | Scope | Seed | Domain | Pool | n | Raw RMSE | Raw MAE | Raw NASA sum | Overflow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N0-mlp | FD001 | 42 | FD001 | official_test | 100 | 27.314 | 21.389 | 9.358208505e+3 | False |
| N0-mlp | FD001 | 42 | FD004 | held_out_official_test | 248 | 35811 | 29550 | 1.950364271e+4677 | True |
| N0-mlp | FD001 | 42 | FD004 | held_out_run_to_failure | 5516 | 36773 | 29681 | 1.922226765e+5155 | True |
| N0-mlp | FD001 | 43 | FD001 | official_test | 100 | 26.933 | 19.741 | 1.05644921e+4 | False |
| N0-mlp | FD001 | 43 | FD004 | held_out_official_test | 248 | 87410 | 73584 | 6.590488211e+7958 | True |
| N0-mlp | FD001 | 43 | FD004 | held_out_run_to_failure | 5516 | 83237 | 70470 | 6.692816546e+8577 | True |
| N0-mlp | FD002 | 42 | FD002 | official_test | 259 | 41.414 | 33.39 | 1.654289961e+5 | False |
| N0-mlp | FD002 | 42 | FD004 | held_out_official_test | 248 | 54.341 | 43.895 | 1.316920241e+7 | False |
| N0-mlp | FD002 | 42 | FD004 | held_out_run_to_failure | 5516 | 66.976 | 51.25 | 1.753596786e+12 | False |
| N0-mlp | FD002 | 43 | FD002 | official_test | 259 | 43.683 | 33.627 | 5.070519041e+5 | False |
| N0-mlp | FD002 | 43 | FD004 | held_out_official_test | 248 | 58.659 | 47.476 | 5.483292974e+7 | False |
| N0-mlp | FD002 | 43 | FD004 | held_out_run_to_failure | 5516 | 67.37 | 51.742 | 1.200429887e+12 | False |
| N0-mlp | FD003 | 42 | FD003 | official_test | 100 | 44.449 | 33.71 | 9.514873103e+5 | False |
| N0-mlp | FD003 | 42 | FD004 | held_out_official_test | 248 | 1.1224e+05 | 98664 | 3.943824208e+11932 | True |
| N0-mlp | FD003 | 42 | FD004 | held_out_run_to_failure | 5516 | 1.1093e+05 | 98269 | 6.475636591e+11425 | True |
| N0-mlp | FD003 | 43 | FD003 | official_test | 100 | 42.546 | 31.759 | 4.581136931e+6 | False |
| N0-mlp | FD003 | 43 | FD004 | held_out_official_test | 248 | 1.4958e+05 | 1.3604e+05 | 2.531935497e+13432 | True |
| N0-mlp | FD003 | 43 | FD004 | held_out_run_to_failure | 5516 | 1.4853e+05 | 1.3495e+05 | 1.020031797e+14577 | True |
| N0-mlp | pooled | 42 | FD001 | official_test | 100 | 29.943 | 23.09 | 3.516628008e+4 | False |
| N0-mlp | pooled | 42 | FD002 | official_test | 259 | 46.548 | 38.068 | 1.027546826e+6 | False |
| N0-mlp | pooled | 42 | FD003 | official_test | 100 | 41.654 | 30.201 | 1.127077636e+6 | False |
| N0-mlp | pooled | 42 | FD004 | held_out_official_test | 248 | 53.427 | 44.213 | 1.083741932e+7 | False |
| N0-mlp | pooled | 42 | FD004 | held_out_run_to_failure | 5516 | 75.369 | 56.636 | 1.022463569e+14 | False |
| N0-mlp | pooled | 43 | FD001 | official_test | 100 | 27.699 | 21.843 | 6.344983512e+3 | False |
| N0-mlp | pooled | 43 | FD002 | official_test | 259 | 48.49 | 39.234 | 6.399626114e+5 | False |
| N0-mlp | pooled | 43 | FD003 | official_test | 100 | 37.663 | 29.67 | 4.355981666e+4 | False |
| N0-mlp | pooled | 43 | FD004 | held_out_official_test | 248 | 52.029 | 43.1 | 9.066214491e+5 | False |
| N0-mlp | pooled | 43 | FD004 | held_out_run_to_failure | 5516 | 77.187 | 58.273 | 1.841709467e+13 | False |
| N2-gru | FD001 | 42 | FD001 | official_test | 100 | 23.324 | 17.027 | 2.600797187e+3 | False |
| N2-gru | FD001 | 42 | FD004 | held_out_official_test | 248 | 55.97 | 44.881 | 2.899645682e+5 | False |
| N2-gru | FD001 | 42 | FD004 | held_out_run_to_failure | 5516 | 97.516 | 71.151 | 1.578120015e+15 | False |
| N2-gru | FD001 | 43 | FD001 | official_test | 100 | 25.661 | 18.637 | 4.390538111e+3 | False |
| N2-gru | FD001 | 43 | FD004 | held_out_official_test | 248 | 52.056 | 41.457 | 2.542628657e+6 | False |
| N2-gru | FD001 | 43 | FD004 | held_out_run_to_failure | 5516 | 80.889 | 61.105 | 1.000767314e+13 | False |
| N2-gru | FD002 | 42 | FD002 | official_test | 259 | 26.713 | 19.037 | 9.924741476e+3 | False |
| N2-gru | FD002 | 42 | FD004 | held_out_official_test | 248 | 61.153 | 45.489 | 8.334174262e+6 | False |
| N2-gru | FD002 | 42 | FD004 | held_out_run_to_failure | 5516 | 70.042 | 51.546 | 3.222297204e+11 | False |
| N2-gru | FD002 | 43 | FD002 | official_test | 259 | 27.005 | 19.682 | 1.008777057e+4 | False |
| N2-gru | FD002 | 43 | FD004 | held_out_official_test | 248 | 72.954 | 53.744 | 6.9654062e+7 | False |
| N2-gru | FD002 | 43 | FD004 | held_out_run_to_failure | 5516 | 75.101 | 55.607 | 8.907112334e+10 | False |
| N2-gru | FD003 | 42 | FD003 | official_test | 100 | 48.004 | 37.322 | 1.150642717e+7 | False |
| N2-gru | FD003 | 42 | FD004 | held_out_official_test | 248 | 130.49 | 119.73 | 2.201623383e+10 | False |
| N2-gru | FD003 | 42 | FD004 | held_out_run_to_failure | 5516 | 115.94 | 100.91 | 3.655204395e+11 | False |
| N2-gru | FD003 | 43 | FD003 | official_test | 100 | 45.518 | 34.903 | 6.798063698e+6 | False |
| N2-gru | FD003 | 43 | FD004 | held_out_official_test | 248 | 153.61 | 144.74 | 2.907532264e+11 | False |
| N2-gru | FD003 | 43 | FD004 | held_out_run_to_failure | 5516 | 135.8 | 120.26 | 4.437267051e+12 | False |
| N2-gru | pooled | 42 | FD001 | official_test | 100 | 23.848 | 16.849 | 4.319963708e+3 | False |
| N2-gru | pooled | 42 | FD002 | official_test | 259 | 26.464 | 18.4 | 1.516889962e+4 | False |
| N2-gru | pooled | 42 | FD003 | official_test | 100 | 37.169 | 23.827 | 1.757906291e+7 | False |
| N2-gru | pooled | 42 | FD004 | held_out_official_test | 248 | 36.891 | 27.358 | 1.661255972e+6 | False |
| N2-gru | pooled | 42 | FD004 | held_out_run_to_failure | 5516 | 54.825 | 37.683 | 8.89011797e+12 | False |
| N2-gru | pooled | 43 | FD001 | official_test | 100 | 24.33 | 17.833 | 5.09571337e+3 | False |
| N2-gru | pooled | 43 | FD002 | official_test | 259 | 26.753 | 18.854 | 9.90806781e+3 | False |
| N2-gru | pooled | 43 | FD003 | official_test | 100 | 41.154 | 27.01 | 2.244651012e+8 | False |
| N2-gru | pooled | 43 | FD004 | held_out_official_test | 248 | 32.66 | 26.036 | 2.354015226e+4 | False |
| N2-gru | pooled | 43 | FD004 | held_out_run_to_failure | 5516 | 55.577 | 38.061 | 1.477367548e+13 | False |
| N2-mlp | FD001 | 42 | FD001 | official_test | 100 | 26.005 | 20.894 | 4.176775293e+3 | False |
| N2-mlp | FD001 | 42 | FD004 | held_out_official_test | 248 | 257.82 | 247.31 | 6.634086059e+18 | False |
| N2-mlp | FD001 | 42 | FD004 | held_out_run_to_failure | 5516 | 233.83 | 215.71 | 4.676987181e+21 | False |
| N2-mlp | FD001 | 43 | FD001 | official_test | 100 | 29.002 | 22.115 | 7.18225565e+3 | False |
| N2-mlp | FD001 | 43 | FD004 | held_out_official_test | 248 | 175.02 | 158.25 | 3.039757753e+16 | False |
| N2-mlp | FD001 | 43 | FD004 | held_out_run_to_failure | 5516 | 150.71 | 129.01 | 1.248797426e+18 | False |
| N2-mlp | FD002 | 42 | FD002 | official_test | 259 | 29.758 | 21.851 | 2.083540674e+4 | False |
| N2-mlp | FD002 | 42 | FD004 | held_out_official_test | 248 | 98.501 | 72.464 | 7.218247336e+11 | False |
| N2-mlp | FD002 | 42 | FD004 | held_out_run_to_failure | 5516 | 88.423 | 64.03 | 6.648040213e+12 | False |
| N2-mlp | FD002 | 43 | FD002 | official_test | 259 | 30.245 | 22.208 | 2.670689071e+4 | False |
| N2-mlp | FD002 | 43 | FD004 | held_out_official_test | 248 | 112.34 | 81.264 | 2.006025219e+12 | False |
| N2-mlp | FD002 | 43 | FD004 | held_out_run_to_failure | 5516 | 99.611 | 71.014 | 1.476380693e+13 | False |
| N2-mlp | FD003 | 42 | FD003 | official_test | 100 | 41.785 | 32.345 | 7.853689833e+6 | False |
| N2-mlp | FD003 | 42 | FD004 | held_out_official_test | 248 | 718.43 | 645.58 | 3.816215063e+80 | False |
| N2-mlp | FD003 | 42 | FD004 | held_out_run_to_failure | 5516 | 650.92 | 581.74 | 1.593935914e+81 | False |
| N2-mlp | FD003 | 43 | FD003 | official_test | 100 | 42.634 | 32.725 | 5.119927364e+6 | False |
| N2-mlp | FD003 | 43 | FD004 | held_out_official_test | 248 | 974.04 | 947.26 | 4.44570105e+68 | False |
| N2-mlp | FD003 | 43 | FD004 | held_out_run_to_failure | 5516 | 922.52 | 892.3 | 2.055382682e+78 | False |
| N2-mlp | pooled | 42 | FD001 | official_test | 100 | 30.101 | 22.225 | 1.526969837e+4 | False |
| N2-mlp | pooled | 42 | FD002 | official_test | 259 | 30.559 | 22.438 | 5.775803556e+4 | False |
| N2-mlp | pooled | 42 | FD003 | official_test | 100 | 40.499 | 28.263 | 3.023092737e+7 | False |
| N2-mlp | pooled | 42 | FD004 | held_out_official_test | 248 | 49.809 | 37.389 | 5.451163656e+9 | False |
| N2-mlp | pooled | 42 | FD004 | held_out_run_to_failure | 5516 | 59.007 | 42.607 | 3.641053489e+19 | False |
| N2-mlp | pooled | 43 | FD001 | official_test | 100 | 31.435 | 23.867 | 2.949061223e+4 | False |
| N2-mlp | pooled | 43 | FD002 | official_test | 259 | 29.855 | 22.388 | 4.777030142e+4 | False |
| N2-mlp | pooled | 43 | FD003 | official_test | 100 | 39.807 | 28.405 | 1.210672035e+7 | False |
| N2-mlp | pooled | 43 | FD004 | held_out_official_test | 248 | 48.754 | 36.641 | 2.23326316e+8 | False |
| N2-mlp | pooled | 43 | FD004 | held_out_run_to_failure | 5516 | 61.693 | 44.568 | 5.556438213e+18 | False |

## Azure source-model availability ablations

Only model1/2/3 enter these18 fixed fits. Each pseudo-held model's temporal test rows are excluded from all fitting/checkpoint/threshold decisions. Feature groups/lags have identical machine/timestamp populations; each arm refits imputation, scaling and feature selection on permitted training rows. Validation-only thresholds optimize F1; no model4 score/statistic is used. All fits use seed42, so no Azure multi-seed robustness is claimed.

Azure controls use sklearn MLPClassifier: d locally retained/aligned inputs →32 ReLU→1 sigmoid, with32*d+65 trainable parameters; d is listed per arm/fold below. Adam lr0.001, batch128, alpha0.0001, at most40 epochs/patience6; inner validation AP selects the checkpoint. This classification objective is separate from the matched NASA regression objective.

| Arm | Held source model | d | Parameters | n | No-skill AP | AP | PR-AUC trapezoid | F1 | Precision | Recall | Inner threshold | Confusion TN,FP/FN,TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| telemetry | azure-model1 | 12 | 449 | 1456 | 0.028846 | 0.23483 | 0.22742 | 0.15789 | 0.17647 | 0.14286 | 0.20902 | [[1386, 28], [36, 6]] |
| telemetry | azure-model2 | 12 | 449 | 1547 | 0.024564 | 0.23546 | 0.2235 | 0.25 | 0.22 | 0.28947 | 0.16414 | [[1470, 39], [27, 11]] |
| telemetry | azure-model3 | 12 | 449 | 3185 | 0.014757 | 0.21971 | 0.21413 | 0.27551 | 0.18121 | 0.57447 | 0.16465 | [[3016, 122], [20, 27]] |
| errors | azure-model1 | 17 | 609 | 1456 | 0.028846 | 0.99378 | 0.9937 | 0.96386 | 0.97561 | 0.95238 | 0.30586 | [[1413, 1], [2, 40]] |
| errors | azure-model2 | 17 | 609 | 1547 | 0.024564 | 0.99141 | 0.99128 | 0.97368 | 0.97368 | 0.97368 | 0.44704 | [[1508, 1], [1, 37]] |
| errors | azure-model3 | 17 | 609 | 3185 | 0.014757 | 0.99724 | 0.99722 | 0.98925 | 1 | 0.97872 | 0.65058 | [[3138, 0], [1, 46]] |
| maintenance | azure-model1 | 16 | 577 | 1456 | 0.028846 | 0.25651 | 0.24208 | 0.30894 | 0.23457 | 0.45238 | 0.13665 | [[1352, 62], [23, 19]] |
| maintenance | azure-model2 | 16 | 577 | 1547 | 0.024564 | 0.22305 | 0.2124 | 0.28571 | 0.22388 | 0.39474 | 0.13674 | [[1457, 52], [23, 15]] |
| maintenance | azure-model3 | 16 | 577 | 3185 | 0.014757 | 0.20591 | 0.19483 | 0.18349 | 0.16129 | 0.21277 | 0.2438 | [[3086, 52], [37, 10]] |
| full | azure-model1 | 21 | 737 | 1456 | 0.028846 | 0.99413 | 0.99405 | 0.97619 | 0.97619 | 0.97619 | 0.22194 | [[1413, 1], [1, 41]] |
| full | azure-model2 | 21 | 737 | 1547 | 0.024564 | 0.99141 | 0.99128 | 0.96 | 0.97297 | 0.94737 | 0.40621 | [[1508, 1], [2, 36]] |
| full | azure-model3 | 21 | 737 | 3185 | 0.014757 | 0.98757 | 0.98748 | 0.95652 | 0.97778 | 0.93617 | 0.59475 | [[3137, 1], [3, 44]] |
| full_lag6 | azure-model1 | 21 | 737 | 1456 | 0.028846 | 0.9925 | 0.99241 | 0.96386 | 0.97561 | 0.95238 | 0.40181 | [[1413, 1], [2, 40]] |
| full_lag6 | azure-model2 | 21 | 737 | 1547 | 0.024564 | 0.99141 | 0.99128 | 0.96 | 0.97297 | 0.94737 | 0.48125 | [[1508, 1], [2, 36]] |
| full_lag6 | azure-model3 | 21 | 737 | 3185 | 0.014757 | 0.98341 | 0.98327 | 0.95652 | 0.97778 | 0.93617 | 0.67303 | [[3137, 1], [3, 44]] |
| full_lag12 | azure-model1 | 21 | 737 | 1456 | 0.028846 | 0.98681 | 0.98664 | 0.93976 | 0.95122 | 0.92857 | 0.27865 | [[1412, 2], [3, 39]] |
| full_lag12 | azure-model2 | 21 | 737 | 1547 | 0.024564 | 0.99141 | 0.99128 | 0.98701 | 0.97436 | 1 | 0.50503 | [[1508, 1], [0, 38]] |
| full_lag12 | azure-model3 | 21 | 737 | 3185 | 0.014757 | 0.99426 | 0.99421 | 0.95556 | 1 | 0.91489 | 0.85213 | [[3138, 0], [4, 43]] |

### Feature-group summaries (equally weighted source folds)

| Arm | Macro AP | Macro PR-AUC | Macro F1 |
| --- | --- | --- | --- |
| errors | 0.99414 | 0.99406 | 0.9756 |
| full | 0.99104 | 0.99094 | 0.96424 |
| full_lag12 | 0.99083 | 0.99071 | 0.96078 |
| full_lag6 | 0.98911 | 0.98899 | 0.96013 |
| maintenance | 0.22849 | 0.21644 | 0.25938 |
| telemetry | 0.23 | 0.22168 | 0.2278 |

Events use (t-24h,t-lag] and telemetry ends at t; targets use failures in (t,t+24h]. Sampling is daily at the fixed grid; full history and horizon are purged across chronological cuts. Exact-time exclusion, future-event mutation, duplicate rejection and backdated-posting tests passed. Maintenance action versus logging time cannot be separated: posting/ingestion times are absent. Lag6/12 sensitivity and event ablations do not prove deployment availability. Synthetic-dataset structure and grid/event alignment may influence performance. **Use telemetry-only as the conservative reference until posting-time availability is established**, regardless of full-feature AP.

Source-only event timing audit: `{"errors": {"duplicate_events": 0, "events": 2702, "exact_failure_time_events": 3, "posting_timestamps_available": false}, "maint": {"duplicate_events": 0, "events": 2247, "exact_failure_time_events": 764, "posting_timestamps_available": false}}`.

## Leakage, federation and privacy boundaries

Source-only diagnostics reproduced all six frozen-scaler comparisons. Near-constant source scales and operating-condition changes amplify inputs (FD001→FD002 |z|p99≈19214; FD003→FD002≈19173). This supports a scaling-related failure mechanism, not the claim that scaling alone explains RUL failures. No FD004 outcome/statistic/prediction entered selection. Azure model4 was filtered before validation/sample generation; raw CSV ingestion necessarily parses records before filtering, without computing model4 statistics.

Neural NPZ checkpoints contain only common named tensors. Medians/scales/constant masks/regime dictionaries are separate private JSON states. The fixed24-coordinate contract and same tensor names/shapes are mechanically suitable for later FedAvg/FedProx; shared coordinates do not establish equivalent local semantics or accuracy after aggregation. Centralized pooling is a non-private reference.

Phase03 recommendation: retain **N2-gru** as the source-gated backbone. GRU remains a compact alternative only to the extent matched source evidence supports it; target scores cannot promote it. Keep30 chronological steps, sorted24 inputs, neutral constants and boolean masks; aggregate/proximal-penalize neural weights only. No federated implementation has started.

The overall academic privacy requirement is **not yet satisfied**. Local fitting is compatible, but no formal privacy mechanism, threat model, accounting, residual-leakage evaluation or aggregation implementation exists. Strict zero-shot deployment remains unresolved: exporting a source scaler discloses source statistics, moving raw target data violates the intended boundary, and fitting target scalers is adaptation. A later approved protocol must resolve this without relabeling scaler statistics as model updates.

## Preservation, artifacts and reproduction

Independent audit verified323 frozen artifacts, recomputed saved metrics from129 prediction CSVs, reloaded24 final NASA models, and verified344 original files byte-for-byte. The audit did not reread raw held-out data or execute another model forward evaluation.

All new outputs are under outputs/phase021/: diagnostics.json; budget.json; screen/; confirm/; selection.json; azure/ and azure.json; final-models/ and final-models.json; freeze.json and freeze.sha256; evaluation-started.json; evaluation/ and evaluation.json; nasa-metrics.csv; verification.json; preservation.json and preservation-verified.json. Predictions, checkpoint histories, engine membership hashes, local private statistics, unsupported regimes, max/p95 errors and scientific/log overflow scores remain available.

Use the original requirements-lock.txt for Phase02. The separate pinned requirements-phase021.lock contains torch==2.6.0+cpu and preserves every original dependency pin; the original .venv was unchanged. Commands from the repository root:

```powershell
py -3.12 -m venv .venv-phase021
.\.venv-phase021\Scripts\python.exe -m pip install -r requirements-phase021.lock
.\.venv-phase021\Scripts\python.exe -W error scripts/phase021.py diagnostics
.\.venv-phase021\Scripts\python.exe -W error scripts/phase021.py source
.\.venv-phase021\Scripts\python.exe -W error scripts/phase021.py azure
.\.venv-phase021\Scripts\python.exe -W error scripts/phase021.py fit
.\.venv-phase021\Scripts\python.exe -W error scripts/phase021.py freeze
.\.venv-phase021\Scripts\python.exe -W error scripts/phase021.py evaluate
.\.venv-phase021\Scripts\python.exe -W error scripts/verify_phase021.py
.\.venv-phase021\Scripts\python.exe scripts/report_phase021.py
```

Training/evaluation stages deliberately reject reuse of completed run IDs and the final-evaluation marker. Reproduction requires a fresh checkout/output location with the preserved Phase01/02/alignment inputs; do not delete or overwrite this accepted run. Audit/report commands are safe to repeat. Test counts and exact validation commands are recorded in PHASE021_ACCEPTANCE.md.
