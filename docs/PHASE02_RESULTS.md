# Phase 02 measured baseline results

Newly executed local results, 2026-10-21. All values below derive from `outputs/phase02/plan.json`; no historical results are imported.

Completed **36 runs**, with seeds [42, 43, 44], in **71.34 seconds** wall time. Candidate fitting/selection totaled **51.99 seconds**. All planned runs completed; no full-budget run was failed or skipped. The separate two-epoch pytest smoke plan is excluded.

Values are arithmetic mean ± sample standard deviation across the three fixed seeds, unless explicitly per-seed. They measure optimization sensitivity on fixed splits, not confidence intervals or significance. Logistic repeats are deterministic.

## Azure

PR-AUC is trapezoidal area; AP is average precision. Every decision uses its validation-selected threshold. Positive prevalence and all per-seed confusion matrices/thresholds are retained in the plan and `metrics.csv`.

### Permitted known-factory temporal tests

| Architecture | Scope | Frozen held-out source | Test factory | PR-AUC | AP | F1 | Precision | Recall | ROC-AUC |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | centralized | azure-model1 | azure-model1 | 0.9160 ± 0.0000 | 0.9176 ± 0.0000 | 0.8706 ± 0.0000 | 0.8605 ± 0.0000 | 0.8810 ± 0.0000 | 0.9981 ± 0.0000 |
| logistic | centralized | azure-model1 | azure-model2 | 0.8854 ± 0.0000 | 0.8873 ± 0.0000 | 0.7606 ± 0.0000 | 0.8182 ± 0.0000 | 0.7105 ± 0.0000 | 0.9973 ± 0.0000 |
| logistic | centralized | azure-model1 | azure-model3 | 0.8007 ± 0.0000 | 0.8037 ± 0.0000 | 0.8148 ± 0.0000 | 0.7213 ± 0.0000 | 0.9362 ± 0.0000 | 0.9974 ± 0.0000 |
| logistic | local | azure-model1 | azure-model1 | 0.9068 ± 0.0000 | 0.9087 ± 0.0000 | 0.8736 ± 0.0000 | 0.8444 ± 0.0000 | 0.9048 ± 0.0000 | 0.9979 ± 0.0000 |
| logistic | local | azure-model2 | azure-model2 | 0.8835 ± 0.0000 | 0.8862 ± 0.0000 | 0.8533 ± 0.0000 | 0.8649 ± 0.0000 | 0.8421 ± 0.0000 | 0.9975 ± 0.0000 |
| logistic | local | azure-model3 | azure-model3 | 0.8703 ± 0.0000 | 0.8730 ± 0.0000 | 0.8824 ± 0.0000 | 0.8182 ± 0.0000 | 0.9574 ± 0.0000 | 0.9987 ± 0.0000 |
| mlp | centralized | azure-model1 | azure-model1 | 0.9938 ± 0.0001 | 0.9938 ± 0.0001 | 0.9762 ± 0.0000 | 0.9762 ± 0.0000 | 0.9762 ± 0.0000 | 0.9998 ± 0.0000 |
| mlp | centralized | azure-model1 | azure-model2 | 0.9922 ± 0.0011 | 0.9923 ± 0.0011 | 0.9691 ± 0.0079 | 0.9734 ± 0.0004 | 0.9649 ± 0.0152 | 0.9998 ± 0.0000 |
| mlp | centralized | azure-model1 | azure-model3 | 0.9979 ± 0.0019 | 0.9979 ± 0.0019 | 0.9857 ± 0.0064 | 0.9931 ± 0.0120 | 0.9787 ± 0.0213 | 1.0000 ± 0.0000 |
| mlp | local | azure-model1 | azure-model1 | 0.9765 ± 0.0261 | 0.9769 ± 0.0257 | 0.9315 ± 0.0296 | 0.9378 ± 0.0332 | 0.9286 ± 0.0714 | 0.9993 ± 0.0008 |
| mlp | local | azure-model2 | azure-model2 | 0.9845 ± 0.0074 | 0.9847 ± 0.0073 | 0.9177 ± 0.0525 | 0.9366 ± 0.0112 | 0.9035 ± 0.0996 | 0.9996 ± 0.0002 |
| mlp | local | azure-model3 | azure-model3 | 0.9951 ± 0.0039 | 0.9951 ± 0.0039 | 0.9546 ± 0.0212 | 0.9457 ± 0.0395 | 0.9645 ± 0.0246 | 0.9999 ± 0.0001 |

### Zero-shot held-out Azure model4

| Architecture | Scope | Frozen held-out source | Test factory | PR-AUC | AP | F1 | Precision | Recall | ROC-AUC |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | centralized | azure-model1 | azure-model4 | 0.7937 ± 0.0000 | 0.7948 ± 0.0000 | 0.8022 ± 0.0000 | 0.7526 ± 0.0000 | 0.8588 ± 0.0000 | 0.9975 ± 0.0000 |
| logistic | local | azure-model1 | azure-model4 | 0.7510 ± 0.0000 | 0.7522 ± 0.0000 | 0.7760 ± 0.0000 | 0.6963 ± 0.0000 | 0.8765 ± 0.0000 | 0.9969 ± 0.0000 |
| logistic | local | azure-model2 | azure-model4 | 0.7497 ± 0.0000 | 0.7509 ± 0.0000 | 0.7303 ± 0.0000 | 0.6989 ± 0.0000 | 0.7647 ± 0.0000 | 0.9967 ± 0.0000 |
| logistic | local | azure-model3 | azure-model4 | 0.8994 ± 0.0000 | 0.8999 ± 0.0000 | 0.8427 ± 0.0000 | 0.8065 ± 0.0000 | 0.8824 ± 0.0000 | 0.9988 ± 0.0000 |
| mlp | centralized | azure-model1 | azure-model4 | 0.9850 ± 0.0043 | 0.9850 ± 0.0043 | 0.9504 ± 0.0115 | 0.9259 ± 0.0163 | 0.9765 ± 0.0176 | 0.9998 ± 0.0001 |
| mlp | local | azure-model1 | azure-model4 | 0.9344 ± 0.0421 | 0.9346 ± 0.0420 | 0.8544 ± 0.0646 | 0.8528 ± 0.0797 | 0.8627 ± 0.1028 | 0.9984 ± 0.0013 |
| mlp | local | azure-model2 | azure-model4 | 0.9443 ± 0.0237 | 0.9445 ± 0.0236 | 0.8753 ± 0.0156 | 0.8794 ± 0.0460 | 0.8745 ± 0.0500 | 0.9988 ± 0.0009 |
| mlp | local | azure-model3 | azure-model4 | 0.9884 ± 0.0020 | 0.9885 ± 0.0020 | 0.9548 ± 0.0145 | 0.9388 ± 0.0316 | 0.9725 ± 0.0324 | 0.9999 ± 0.0000 |

## NASA

Truth is raw uncapped RUL; predictions are floored at zero but not capped. NASA score is a sum, so pool sizes matter. An overflow is retained in log/scientific form, not replaced by a finite clipped score.

### Known-domain official terminal tests

| Scope | Frozen held-out source | Evaluation domain | RMSE (cycles) | MAE (cycles) | NASA score | ln(NASA score) | Overflow seeds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| centralized | nasa-FD002 | nasa-FD001 | 35.2667 ± 1.9394 | 26.1475 ± 0.4789 | 3.079e+05 ± 2.879e+05 | 12.1394 ± 1.4108 | 0/3 |
| centralized | nasa-FD002 | nasa-FD002 | 52.0543 ± 1.0942 | 41.5914 ± 0.7425 | 2.250e+06 ± 1.132e+06 | 14.5093 ± 0.6397 | 0/3 |
| centralized | nasa-FD002 | nasa-FD003 | 46.1479 ± 2.0415 | 33.3455 ± 0.6027 | 6.816e+07 ± 1.086e+08 | 16.6898 ± 2.0804 | 0/3 |
| local | nasa-FD001 | nasa-FD001 | 37.7135 ± 6.2235 | 28.4893 ± 5.4288 | 2.452e+06 ± 4.085e+06 | 12.5051 ± 3.0812 | 0/3 |
| local | nasa-FD002 | nasa-FD002 | 48.8378 ± 1.5248 | 37.9428 ± 1.2949 | 1.790e+06 ± 1.126e+06 | 14.1729 ± 0.9259 | 0/3 |
| local | nasa-FD003 | nasa-FD003 | 45.5132 ± 4.5200 | 35.3602 ± 3.5364 | 1.494e+06 ± 1.569e+06 | 13.8309 ± 1.0717 | 0/3 |

### FD004 held-out official terminal test (248 engines)

| Scope | Frozen held-out source | Evaluation domain | RMSE (cycles) | MAE (cycles) | NASA score | ln(NASA score) | Overflow seeds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| centralized | nasa-FD002 | nasa-FD004 | 57.7881 ± 2.0822 | 46.4403 ± 1.4773 | 2.871e+07 ± 3.944e+07 | 16.4165 ± 1.5027 | 0/3 |
| local | nasa-FD001 | nasa-FD004 | 53312.6357 ± 48850.2293 | 28591.1152 ± 31122.5135 | undefined/overflow; see per-seed JSON | 25917.9892 ± 22660.9859 | 3/3 |
| local | nasa-FD002 | nasa-FD004 | 59.4234 ± 3.0715 | 47.1531 ± 2.3453 | 2.053e+08 ± 2.030e+08 | 18.6826 ± 1.2855 | 0/3 |
| local | nasa-FD003 | nasa-FD004 | 199614.8578 ± 93166.2365 | 168938.2516 ± 99276.1559 | undefined/overflow; see per-seed JSON | 48298.6839 ± 14982.3151 | 3/3 |

### FD004 run-to-failure window diagnostic (5,516 correlated windows)

| Scope | Frozen held-out source | Evaluation domain | RMSE (cycles) | MAE (cycles) | NASA score | ln(NASA score) | Overflow seeds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| centralized | nasa-FD002 | nasa-FD004 | 77.3577 ± 1.5583 | 57.8593 ± 1.2685 | 1.849e+14 ± 2.655e+14 | 31.9116 ± 1.7452 | 0/3 |
| local | nasa-FD001 | nasa-FD004 | 57276.2147 ± 50656.2462 | 29977.5611 ± 33859.0524 | undefined/overflow; see per-seed JSON | 36545.3215 ± 21870.9362 | 3/3 |
| local | nasa-FD002 | nasa-FD004 | 70.6260 ± 0.4178 | 53.6907 ± 0.6571 | 9.175e+13 ± 1.384e+14 | 30.9375 ± 2.0805 | 0/3 |
| local | nasa-FD003 | nasa-FD004 | 198918.7290 ± 99416.9557 | 167623.3818 ± 104138.9346 | undefined/overflow; see per-seed JSON | 61049.7392 ± 14682.9557 | 3/3 |

The single-condition FD001/FD003 source models extrapolate severely on FD004 under frozen source scaling. These unsuccessful zero-shot results remain part of acceptance evidence. The centralized source FD002 was chosen by training-only feature coverage before evaluation; no transform was changed after seeing these results.

## Exact training config

```json
{
  "seeds": [
    42,
    43,
    44
  ],
  "hidden_units": 32,
  "max_epochs": 40,
  "patience": 6,
  "batch_size": 128,
  "learning_rates": [
    0.001,
    0.003
  ],
  "alpha": 0.0001,
  "logistic_c": [
    0.1,
    1.0,
    10.0
  ],
  "logistic_max_iter": 2000,
  "rul_scale": 100.0
}
```

## Selected checkpoints

Logistic validation scores are AP; MLP validation scores are AP for Azure and negative raw-window RMSE for NASA. Times include every candidate for that run, not just its selected checkpoint.

| Run ID | Selected candidate | Epoch | Validation score | Azure threshold | Training seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| azure-logistic-azure-model1-seed42 | C=0.1 | 1 | 0.83278 | 0.40728174 | 0.045 |
| azure-logistic-azure-model2-seed42 | C=1.0 | 1 | 0.93791 | 0.44077871 | 0.021 |
| azure-logistic-azure-model3-seed42 | C=10.0 | 1 | 0.880947 | 0.3916465 | 0.076 |
| azure-logistic-pooled-seed42 | C=10.0 | 1 | 0.798277 | 0.42722039 | 0.119 |
| azure-logistic-azure-model1-seed43 | C=0.1 | 1 | 0.83278 | 0.40728174 | 0.034 |
| azure-logistic-azure-model2-seed43 | C=1.0 | 1 | 0.93791 | 0.44077871 | 0.032 |
| azure-logistic-azure-model3-seed43 | C=10.0 | 1 | 0.880947 | 0.3916465 | 0.066 |
| azure-logistic-pooled-seed43 | C=10.0 | 1 | 0.798277 | 0.42722039 | 0.121 |
| azure-logistic-azure-model1-seed44 | C=0.1 | 1 | 0.83278 | 0.40728174 | 0.031 |
| azure-logistic-azure-model2-seed44 | C=1.0 | 1 | 0.93791 | 0.44077871 | 0.032 |
| azure-logistic-azure-model3-seed44 | C=10.0 | 1 | 0.880947 | 0.3916465 | 0.068 |
| azure-logistic-pooled-seed44 | C=10.0 | 1 | 0.798277 | 0.42722039 | 0.127 |
| azure-mlp-azure-model1-seed42 | lr=0.003 | 14 | 1 | 0.82282458 | 0.392 |
| azure-mlp-azure-model2-seed42 | lr=0.001 | 23 | 1 | 0.36052656 | 0.344 |
| azure-mlp-azure-model3-seed42 | lr=0.003 | 7 | 0.994955 | 0.28152187 | 0.630 |
| azure-mlp-pooled-seed42 | lr=0.003 | 24 | 0.986323 | 0.45167834 | 1.247 |
| azure-mlp-azure-model1-seed43 | lr=0.001 | 22 | 0.989212 | 0.42482016 | 0.379 |
| azure-mlp-azure-model2-seed43 | lr=0.001 | 10 | 1 | 0.4639684 | 0.718 |
| azure-mlp-azure-model3-seed43 | lr=0.001 | 30 | 0.993087 | 0.33003341 | 2.416 |
| azure-mlp-pooled-seed43 | lr=0.003 | 40 | 0.990962 | 0.25084874 | 6.016 |
| azure-mlp-azure-model1-seed44 | lr=0.001 | 12 | 0.994235 | 0.44031 | 0.808 |
| azure-mlp-azure-model2-seed44 | lr=0.001 | 25 | 1 | 0.52631031 | 0.978 |
| azure-mlp-azure-model3-seed44 | lr=0.003 | 7 | 1 | 0.56976001 | 1.401 |
| azure-mlp-pooled-seed44 | lr=0.001 | 14 | 0.995949 | 0.4707689 | 2.308 |
| nasa-mlp-nasa-FD001-seed42 | lr=0.001 | 39 | -41.8459 | — | 2.158 |
| nasa-mlp-nasa-FD002-seed42 | lr=0.001 | 16 | -54.1836 | — | 3.962 |
| nasa-mlp-nasa-FD003-seed42 | lr=0.003 | 9 | -50.7748 | — | 2.291 |
| nasa-mlp-pooled-seed42 | lr=0.003 | 7 | -53.2722 | — | 5.153 |
| nasa-mlp-nasa-FD001-seed43 | lr=0.003 | 15 | -34.5314 | — | 1.630 |
| nasa-mlp-nasa-FD002-seed43 | lr=0.001 | 8 | -51.2555 | — | 2.254 |
| nasa-mlp-nasa-FD003-seed43 | lr=0.003 | 9 | -49.6135 | — | 1.646 |
| nasa-mlp-pooled-seed43 | lr=0.001 | 9 | -51.7872 | — | 4.457 |
| nasa-mlp-nasa-FD001-seed44 | lr=0.003 | 12 | -35.9691 | — | 1.329 |
| nasa-mlp-nasa-FD002-seed44 | lr=0.001 | 9 | -54.1662 | — | 2.579 |
| nasa-mlp-nasa-FD003-seed44 | lr=0.003 | 9 | -49.2914 | — | 1.970 |
| nasa-mlp-pooled-seed44 | lr=0.003 | 4 | -52.8408 | — | 4.151 |

## Provenance and artifact locations

```json
{
  "source_revision": "7109f69ee292b1fdab57c1379465c2feb2417e4f",
  "source_sha256": "879d15754154059dcdb4027944230619f289966f2c01c19173e3022947fc408a",
  "phase01_catalog_sha256": "51169908bedcbe0461fee9bf20dcd5379fde458ef64a8b92ca398bc9c4d95f1c",
  "manifest_sha256": "d6bd6c7743c3a94f32a75c7e7f0435fe2b7b2a193847283a236eaa0b526cefcd",
  "training_config_sha256": "fc37df0e2ad0f5d3ad02f21c510f8ef41241ae11dd087eff7a210ae775b87588"
}
```

Environment: `{"numpy": "2.2.6", "pandas": "2.2.3", "python": "3.12.10", "scikit-learn": "1.6.1", "scipy": "1.18.1", "threadpoolctl": "3.7.0"}`.

The plan retains the exact source-file digest map and copied Phase 01 catalog. Each `outputs/phase02/<run-id>/` contains `model.json`, `weights.npz`, frozen `transform-*.json` files, and `predictions-*.csv`. All 138 evaluation files contain source-scoped identity, truth, and prediction; Azure also records its decision. `outputs/phase02/metrics.csv` indexes all validation/test/held-out metrics. `verification.json` records the independent reload/reproduction audit.

Interpretation limits and the fixed training/selection/zero-shot rules are in [PHASE02_PROTOCOL.md](PHASE02_PROTOCOL.md). Acceptance commands and remaining boundaries are in [PHASE02_ACCEPTANCE.md](PHASE02_ACCEPTANCE.md). The results show baseline behavior, not a privacy guarantee, deployment readiness, statistical superiority, or Phase 03 completion.
