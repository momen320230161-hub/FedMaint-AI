# V3 measured results

Executed 2026-10-26 on Windows/CPython 3.12, CPU/one thread. This is new V3 evidence. Historical results are retained separately. **All held-out findings are retrospective**, since FD004/model4 were previously observed. The frozen configuration was not changed in response to these outcomes.

Completed **24 fits / 76 evaluation files**, seeds 42/43. Final end-to-end run took 152.52s; summed model fitting/selection time 78.33s. Timing depends on host load. No failed/omitted candidate in the accepted panel.

Freeze SHA-256: `b7840a7bb285e1971777bd6c31ac9686b1144551499a0adbf20031960bf55b99`. Artifacts: `outputs/v3/reference-final/`; authoritative records: `plan.json`, `freeze.json`, `completion.json`, `results.json`, `verification.json` and per-model numerical weights/private states/predictions.

## Training and selection evidence

| Model/scope | Seed | Train samples | Validation samples | Selected epoch | Validation objective | Frozen source |
| --- | --- | --- | --- | --- | --- | --- |
| azure/mlp/azure-model1 | 42 | 3360 | 944 | 39 | 0.444076 | azure-model1 |
| azure/mlp/azure-model2 | 42 | 3570 | 1003 | 40 | 0.32178 | azure-model2 |
| azure/mlp/azure-model3 | 42 | 7350 | 2065 | 25 | 0.168384 | azure-model3 |
| azure/mlp/pooled | 42 | 14280 | 4012 | 19 | 0.239139 | azure-model1 |
| azure/mlp/azure-model1 | 43 | 3360 | 944 | 36 | 0.439767 | azure-model1 |
| azure/mlp/azure-model2 | 43 | 3570 | 1003 | 39 | 0.257314 | azure-model2 |
| azure/mlp/azure-model3 | 43 | 7350 | 2065 | 14 | 0.181122 | azure-model3 |
| azure/mlp/pooled | 43 | 14280 | 4012 | 16 | 0.23628 | azure-model1 |
| nasa/gru/FD001 | 42 | 1481 | 40 | 23 | -30.4613 | FD001 |
| nasa/gru/FD002 | 42 | 3797 | 104 | 24 | -42.5141 | FD002 |
| nasa/gru/FD003 | 42 | 1812 | 40 | 6 | -59.0591 | FD003 |
| nasa/gru/pooled | 42 | 7090 | 184 | 15 | -45.9155 | FD002 |
| nasa/gru/FD001 | 43 | 1481 | 40 | 6 | -34.3428 | FD001 |
| nasa/gru/FD002 | 43 | 3797 | 104 | 23 | -42.3047 | FD002 |
| nasa/gru/FD003 | 43 | 1812 | 40 | 5 | -57.602 | FD003 |
| nasa/gru/pooled | 43 | 7090 | 184 | 19 | -45.3892 | FD002 |
| nasa/mlp/FD001 | 42 | 1481 | 40 | 16 | -29.8846 | FD001 |
| nasa/mlp/FD002 | 42 | 3797 | 104 | 10 | -41.7638 | FD002 |
| nasa/mlp/FD003 | 42 | 1812 | 40 | 13 | -50.6857 | FD003 |
| nasa/mlp/pooled | 42 | 7090 | 184 | 8 | -44.3101 | FD002 |
| nasa/mlp/FD001 | 43 | 1481 | 40 | 4 | -33.271 | FD001 |
| nasa/mlp/FD002 | 43 | 3797 | 104 | 13 | -40.5093 | FD002 |
| nasa/mlp/FD003 | 43 | 1812 | 40 | 18 | -50.6799 | FD003 |
| nasa/mlp/pooled | 43 | 7090 | 184 | 7 | -43.7171 | FD002 |

Azure objective maximizes AP. NASA objective maximizes negative macro prefix RMSE at cycles 50/100; validation sample counts are prefix samples, not the whole-window catalog counts. Client participation/engine membership and loss histories are in every model.json. Epoch cap 40, patience 6, Adam lr .001, batch 128, alpha .0001; no class weighting/resampling. NASA raw optimization scale 100; no output floor/cap.

## Azure telemetry-only

All 16 shared coordinates: four sensors, mean/min/max/count. MLP16->32->1,577 parameters (2308 float32 bytes). Four locally constant count features are neutral zeros;12 informative source features determine routing. Both local and pooled references have the same architecture/budget.

| Scope | Seed | Evaluation factory/pool | n | Prevalence/no-skill AP | PR-AUC | AP | F1 | Precision | Recall | Threshold | TN/FP/FN/TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| azure-model1 | 42 | azure-model1/test | 1456 | 0.02885 | 0.26489 | 0.27992 | 0.24658 | 0.29032 | 0.21429 | 0.301256 | 1392/22/33/9 |
| azure-model1 | 42 | azure-model4/held_out | 11648 | 0.01459 | 0.13240 | 0.13477 | 0.10702 | 0.12403 | 0.09412 | 0.301256 | 11365/113/154/16 |
| azure-model2 | 42 | azure-model2/test | 1547 | 0.02456 | 0.29237 | 0.30256 | 0.30952 | 0.28261 | 0.34211 | 0.224926 | 1476/33/25/13 |
| azure-model2 | 42 | azure-model4/held_out | 11648 | 0.01459 | 0.12344 | 0.12634 | 0.14940 | 0.12653 | 0.18235 | 0.224926 | 11264/214/139/31 |
| azure-model3 | 42 | azure-model3/test | 3185 | 0.01476 | 0.16605 | 0.17225 | 0.27755 | 0.17172 | 0.72340 | 0.0914026 | 2974/164/13/34 |
| azure-model3 | 42 | azure-model4/held_out | 11648 | 0.01459 | 0.17549 | 0.17843 | 0.27594 | 0.17257 | 0.68824 | 0.0914026 | 10917/561/53/117 |
| pooled | 42 | azure-model1/test | 1456 | 0.02885 | 0.20511 | 0.21411 | 0.32857 | 0.23469 | 0.54762 | 0.170062 | 1339/75/19/23 |
| pooled | 42 | azure-model2/test | 1547 | 0.02456 | 0.24670 | 0.25922 | 0.31193 | 0.23944 | 0.44737 | 0.170062 | 1455/54/21/17 |
| pooled | 42 | azure-model3/test | 3185 | 0.01476 | 0.21511 | 0.22652 | 0.29126 | 0.18868 | 0.63830 | 0.170062 | 3009/129/17/30 |
| pooled | 42 | azure-model4/held_out | 11648 | 0.01459 | 0.16405 | 0.16688 | 0.26074 | 0.17235 | 0.53529 | 0.170062 | 11041/437/79/91 |
| azure-model1 | 43 | azure-model1/test | 1456 | 0.02885 | 0.25994 | 0.26875 | 0.35115 | 0.25843 | 0.54762 | 0.196784 | 1348/66/19/23 |
| azure-model1 | 43 | azure-model4/held_out | 11648 | 0.01459 | 0.12803 | 0.13000 | 0.19343 | 0.14021 | 0.31176 | 0.196784 | 11153/325/117/53 |
| azure-model2 | 43 | azure-model2/test | 1547 | 0.02456 | 0.23430 | 0.25000 | 0.36641 | 0.25806 | 0.63158 | 0.149926 | 1440/69/14/24 |
| azure-model2 | 43 | azure-model4/held_out | 11648 | 0.01459 | 0.12547 | 0.12905 | 0.19382 | 0.12731 | 0.40588 | 0.149926 | 11005/473/101/69 |
| azure-model3 | 43 | azure-model3/test | 3185 | 0.01476 | 0.15251 | 0.16362 | 0.15000 | 0.18182 | 0.12766 | 0.219232 | 3111/27/41/6 |
| azure-model3 | 43 | azure-model4/held_out | 11648 | 0.01459 | 0.15943 | 0.16245 | 0.19620 | 0.21233 | 0.18235 | 0.219232 | 11363/115/139/31 |
| pooled | 43 | azure-model1/test | 1456 | 0.02885 | 0.20760 | 0.21868 | 0.31429 | 0.22449 | 0.52381 | 0.168373 | 1338/76/20/22 |
| pooled | 43 | azure-model2/test | 1547 | 0.02456 | 0.22303 | 0.23186 | 0.32381 | 0.25373 | 0.44737 | 0.168373 | 1459/50/21/17 |
| pooled | 43 | azure-model3/test | 3185 | 0.01476 | 0.17230 | 0.17889 | 0.27136 | 0.17763 | 0.57447 | 0.168373 | 3013/125/20/27 |
| pooled | 43 | azure-model4/held_out | 11648 | 0.01459 | 0.17329 | 0.17587 | 0.26136 | 0.17228 | 0.54118 | 0.168373 | 11036/442/78/92 |

PR-AUC is trapezoidal precision-recall area; AP is noninterpolated average precision. Thresholds come exclusively from the selected checkpoint on permitted validation data. Model4 has170 positives/11648 windows (1.4595%). Pooled AP is 0.16688/0.17587 and F1 is 0.26074/0.26136. These telemetry-only findings do not reproduce the event-enriched historical benchmark and are not industrial reliability estimates. Optional event groups were not retrained in this V3 panel.

## NASA official terminal pools

Sorted 24 inputs,30-cycle masks; N2 candidate local normalization. GRU has 5601 parameters (22404 bytes); masked MLP750->32->1 has 24065 (96260 bytes). Parameter bytes exclude transport overhead/private states and are not measured FL traffic. Raw uncapped RUL and unmodified linear outputs are used.

| Architecture/scope | Seed | Factory/pool | n | RMSE | MAE | NASA score sum | Negative fraction | Error p95/max | Unsupported rows/all observed rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gru/FD001 | 42 | FD001/official_test | 100 | 26.42357 | 19.58769 | 6.382499471e+3 | 0.00000 | 53.94686/78.72556 | 0/13096 |
| gru/FD001 | 42 | FD004/held_out_official_test | 248 | 55.08937 | 44.56703 | 2.410460761e+5 | 0.00000 | 107.86596/140.67756 | 35160/41214 |
| gru/FD002 | 42 | FD002/official_test | 259 | 26.99566 | 19.52513 | 1.098067792e+4 | 0.02703 | 56.54307/92.17573 | 0/33991 |
| gru/FD002 | 42 | FD004/held_out_official_test | 248 | 62.64833 | 46.70921 | 1.012595107e+7 | 0.00000 | 122.95556/141.15733 | 0/41214 |
| gru/FD003 | 42 | FD003/official_test | 100 | 49.27492 | 38.28481 | 1.30789595e+7 | 0.05000 | 99.71066/163.66676 | 0/16596 |
| gru/FD003 | 42 | FD004/held_out_official_test | 248 | 132.25813 | 121.73093 | 2.533920649e+10 | 0.00000 | 200.76161/221.57732 | 35160/41214 |
| gru/pooled | 42 | FD001/official_test | 100 | 26.77761 | 19.28192 | 7.8429746e+3 | 0.00000 | 53.49045/80.17418 | 0/13096 |
| gru/pooled | 42 | FD002/official_test | 259 | 25.91344 | 18.40480 | 1.398668737e+4 | 0.02703 | 59.25361/87.62428 | 0/33991 |
| gru/pooled | 42 | FD003/official_test | 100 | 37.87489 | 25.07032 | 6.192593525e+6 | 0.00000 | 77.77903/155.19098 | 0/16596 |
| gru/pooled | 42 | FD004/held_out_official_test | 248 | 33.67810 | 25.21372 | 3.44165453e+5 | 0.00806 | 65.62569/119.04536 | 0/41214 |
| gru/FD001 | 43 | FD001/official_test | 100 | 26.30023 | 20.08469 | 3.720028002e+3 | 0.01000 | 54.06946/67.39307 | 0/13096 |
| gru/FD001 | 43 | FD004/held_out_official_test | 248 | 55.26239 | 45.83169 | 1.658938645e+6 | 0.00000 | 103.87973/129.65216 | 35160/41214 |
| gru/FD002 | 43 | FD002/official_test | 259 | 26.64831 | 19.52297 | 9.364492082e+3 | 0.01931 | 61.27742/92.88003 | 0/33991 |
| gru/FD002 | 43 | FD004/held_out_official_test | 248 | 69.12024 | 51.24936 | 3.515027823e+7 | 0.00000 | 136.42657/152.53223 | 0/41214 |
| gru/FD003 | 43 | FD003/official_test | 100 | 44.60441 | 34.16115 | 5.766600855e+6 | 0.04000 | 88.92495/155.37360 | 0/16596 |
| gru/FD003 | 43 | FD004/held_out_official_test | 248 | 157.00322 | 148.07286 | 4.600461758e+11 | 0.00000 | 233.43928/249.95816 | 35160/41214 |
| gru/pooled | 43 | FD001/official_test | 100 | 27.66725 | 19.49521 | 1.583224789e+4 | 0.00000 | 58.06235/92.08043 | 0/13096 |
| gru/pooled | 43 | FD002/official_test | 259 | 26.06276 | 18.36445 | 9.903270523e+3 | 0.03475 | 55.20261/93.48463 | 0/33991 |
| gru/pooled | 43 | FD003/official_test | 100 | 42.62786 | 28.06799 | 5.31037168e+7 | 0.00000 | 81.44850/174.21330 | 0/16596 |
| gru/pooled | 43 | FD004/held_out_official_test | 248 | 33.10999 | 25.59112 | 4.843920406e+4 | 0.00806 | 64.61309/97.94490 | 0/41214 |
| mlp/FD001 | 42 | FD001/official_test | 100 | 29.39802 | 21.37206 | 6.691886874e+4 | 0.03000 | 67.25841/108.68855 | 0/13096 |
| mlp/FD001 | 42 | FD004/held_out_official_test | 248 | 264.21561 | 245.65266 | 3.073127759e+21 | 0.00403 | 414.41070/493.12384 | 35160/41214 |
| mlp/FD002 | 42 | FD002/official_test | 259 | 30.79909 | 22.70657 | 3.096793802e+4 | 0.05792 | 65.63814/98.88983 | 0/33991 |
| mlp/FD002 | 42 | FD004/held_out_official_test | 248 | 114.46217 | 83.76875 | 1.236318795e+13 | 0.01613 | 218.41008/301.00241 | 0/41214 |
| mlp/FD003 | 42 | FD003/official_test | 100 | 44.70155 | 33.54828 | 8.332104549e+6 | 0.04000 | 84.77175/157.15118 | 0/16596 |
| mlp/FD003 | 42 | FD004/held_out_official_test | 248 | 941.65952 | 902.10052 | 1.488144629e+76 | 0.00000 | 1355.73515/1753.93909 | 35160/41214 |
| mlp/pooled | 42 | FD001/official_test | 100 | 29.71614 | 21.87497 | 3.101869093e+4 | 0.02000 | 64.76438/100.78300 | 0/13096 |
| mlp/pooled | 42 | FD002/official_test | 259 | 30.68159 | 22.05853 | 5.935365822e+4 | 0.01931 | 67.38660/103.42249 | 0/33991 |
| mlp/pooled | 42 | FD003/official_test | 100 | 39.33774 | 28.19846 | 2.826873677e+7 | 0.00000 | 79.20107/171.53922 | 0/16596 |
| mlp/pooled | 42 | FD004/held_out_official_test | 248 | 47.44544 | 35.62284 | 4.648099834e+9 | 0.01613 | 90.40805/222.53555 | 0/41214 |
| mlp/FD001 | 43 | FD001/official_test | 100 | 30.00542 | 23.11031 | 9.437576633e+3 | 0.01000 | 55.65446/78.64938 | 0/13096 |
| mlp/FD001 | 43 | FD004/held_out_official_test | 248 | 247.07654 | 233.04581 | 4.375539454e+18 | 0.00000 | 360.71552/419.33792 | 35160/41214 |
| mlp/FD002 | 43 | FD002/official_test | 259 | 31.93551 | 23.47016 | 5.278001289e+4 | 0.03475 | 66.95587/101.62001 | 0/33991 |
| mlp/FD002 | 43 | FD004/held_out_official_test | 248 | 116.07777 | 83.70972 | 2.090775973e+14 | 0.02823 | 225.21971/329.61722 | 0/41214 |
| mlp/FD003 | 43 | FD003/official_test | 100 | 42.36311 | 32.60043 | 7.645587891e+6 | 0.05000 | 76.90203/156.72937 | 0/16596 |
| mlp/FD003 | 43 | FD004/held_out_official_test | 248 | 950.21305 | 915.93258 | 9.294433262e+64 | 0.00000 | 1315.73605/1495.81262 | 35160/41214 |
| mlp/pooled | 43 | FD001/official_test | 100 | 32.50023 | 24.51334 | 3.860205813e+4 | 0.07000 | 67.06136/101.88705 | 0/13096 |
| mlp/pooled | 43 | FD002/official_test | 259 | 30.78470 | 22.92971 | 5.49264151e+4 | 0.04247 | 63.99676/95.61061 | 0/33991 |
| mlp/pooled | 43 | FD003/official_test | 100 | 42.27450 | 30.24523 | 5.419870561e+7 | 0.00000 | 79.95980/178.03586 | 0/16596 |
| mlp/pooled | 43 | FD004/held_out_official_test | 248 | 53.71024 | 40.30875 | 2.335813799e+8 | 0.04435 | 109.73380/192.14496 | 0/41214 |

Unsupported-regime counts describe all observed frame rows transformed, not only the rows retained in terminal windows. They expose source support limitations and do not trigger refitting. Official FD004 evaluations contain 248 engines, one terminal sample each.

## NASA run-to-failure diagnostic

These 5516 overlapping windows come from 249 FD004 engines. They are correlated and are not comparable to a5516-engine terminal benchmark.

| Architecture/scope | Seed | Windows | RMSE | MAE | NASA sum |
| --- | --- | --- | --- | --- | --- |
| gru/FD001 | 42 | 5516 | 96.23968 | 70.42998 | 1.407600302e+15 |
| gru/FD002 | 42 | 5516 | 69.80021 | 51.62535 | 2.288906215e+11 |
| gru/FD003 | 42 | 5516 | 117.37228 | 102.37676 | 3.835981955e+11 |
| gru/pooled | 42 | 5516 | 54.12824 | 36.66431 | 1.343213255e+13 |
| gru/FD001 | 43 | 5516 | 81.17150 | 63.11947 | 1.769064277e+13 |
| gru/FD002 | 43 | 5516 | 72.63930 | 53.83509 | 9.439660815e+10 |
| gru/FD003 | 43 | 5516 | 138.67403 | 122.91204 | 7.001970181e+12 |
| gru/pooled | 43 | 5516 | 52.64322 | 35.70915 | 5.231836461e+12 |
| mlp/FD001 | 42 | 5516 | 241.58367 | 215.83716 | 9.076320768e+27 |
| mlp/FD002 | 42 | 5516 | 99.78283 | 70.94724 | 1.242424441e+15 |
| mlp/FD003 | 42 | 5516 | 887.48502 | 845.56154 | 2.114509356e+81 |
| mlp/pooled | 42 | 5516 | 57.28597 | 40.92261 | 2.131771488e+20 |
| mlp/FD001 | 43 | 5516 | 217.24147 | 197.98761 | 2.0817285e+21 |
| mlp/FD002 | 43 | 5516 | 104.11203 | 73.23964 | 1.959070776e+17 |
| mlp/FD003 | 43 | 5516 | 893.33733 | 852.57437 | 9.050735564e+72 |
| mlp/pooled | 43 | 5516 | 61.81555 | 45.21105 | 4.120437123e+18 |

## Interpretation and limitations

Pooled GRU FD004 RMSE is 33.67810/33.10999 cycles, versus 47.44544/53.71024 for the matched pooled MLP. This supports this particular pooled comparison under the frozen budget; it does not establish universal GRU superiority. Local-source performance varies sharply; inspect every local row, including regressions. N2 is not selected again from FD004, and no new N0/N1 ranking is claimed.

Two seeds quantify optimization sensitivity, not statistical significance. Source domains, operating regimes and fault modes are confounded; simulated factories are not independent real organizations. N2 fallback cannot learn unobserved operating conditions. Local normalization gives different meanings to common coordinates, an unresolved FL optimization issue. Prior held-out exposure precludes fully blind claims.

Original targets, temporal/engine partitions, official RUL truth and holdout rules remain unchanged. No target-domain scaler, adaptation, threshold change or outcome-based route switching occurred. Raw predictions may be negative. NASA asymmetric sums preserve overflow/log/scientific representations rather than clipping.

Central pooling is non-private. Model updates can leak; private preprocessing and regime dictionaries are separate from exportable neural tensors. A privacy-approved zero-shot deployment policy is unresolved. No FL strategy, differential privacy, secure aggregation, KG/LLM or frontend is implemented.

## Execution deviations and reproducibility

The initial development panel at outputs/v3/reference completed 24 fits/76 evaluations, but its independent verification exposed up to 5.96e-8 probability differences from standalone CPU thread defaults. Inference was corrected to enforce the recorded single-thread policy, with a regression test. The entire unchanged 24-fit configuration was rerun into reference-final; the development directory was retained. No tolerance was relaxed and no candidate setting was selected from this finding. A premature verification command on reference-final before results.json existed also failed; it made no changes and was rerun after completion.

Full-suite data tests inspect held-out records and were run after experimental freeze/evaluation. Preliminary source-only checks did not load NASA official-test/FD004 data through the gated loader. Raw hash verification and CSV membership filtering still read file bytes; this is distinguished from fitting/selection.

Windows sandbox restrictions prevented pytest temporary-directory access on early attempts. Normal host execution passed without security-setting changes or warning suppression. Exact CPU reruns depend on recorded versions, thread/batch policy and hardware; GPU and other platforms have not been validated.
