# FedKG-Maint project overview

Current implementation: [Architecture V3](ARCHITECTURE_V3.md) consolidates Phase01/02/02.1 into one PyTorch foundation. Use [V3 data protocol](V3_DATA_PROTOCOL.md), [experiment protocol](V3_EXPERIMENT_PROTOCOL.md), [results](V3_RESULTS.md) and [acceptance](V3_ACCEPTANCE.md). Phase03 remains unstarted. Sections describing old training below are historical context.

FedKG-Maint studies privacy-preserving predictive maintenance across simulated factories. Factories learn a shared predictive model through federated learning without exchanging raw training records. A factory-local maintenance knowledge graph (KG) and an LLM explain predictions and recommend maintenance actions using evidence. Evaluation must establish whether the framework generalizes to previously unseen factories with different machines and operating conditions.

The academic source of truth is the original three-page [Project2_Description.pdf](Project2_Description.pdf). The requirements below summarize that brief. FastAPI, React, the exact phase order, and frontend-last delivery are additional user requirements.

## Academic requirements

| Area and PDF location | Required work |
| --- | --- |
| Data and simulated factories, page 1 | Use both NASA C-MAPSS and Microsoft Azure Predictive Maintenance. Partition data into simulated non-IID factory clients and reserve a held-out factory never used for training, to measure new-site generalization. FD001-FD004 domains or Azure machine-model groups are examples, not fixed partition decisions. |
| Preprocessing and targets, page 1 | Clean sensor signals, handle missing values, remove constant or uninformative sensors, and normalize using statistics computed locally within each factory. Create time windows and targets such as remaining useful life (RUL) or failure within a chosen future horizon. |
| Local knowledge graphs, page 2 | Agree on a shared schema/ontology. Each factory builds its own local graph linking machines, components, sensors, error codes, failure modes, and maintenance actions where supported by its data. General domain-knowledge enrichment is optional. |
| Predictive learning and comparisons, page 2 | Train a predictive model through FL with only model updates leaving clients. Compare at least two federated strategies with both local-only training in each factory and centralized training on pooled data. Evaluate the held-out factory for cross-factory generalization. LSTM, Transformer, and GNN are architecture examples, not mandatory selections. |
| Explicit privacy, page 2 | Apply at least one privacy mechanism, such as differential privacy on model updates or secure aggregation. Analyze privacy level versus prediction performance and discuss residual leakage through model updates or shared KG information. |
| LLM explanations, page 2 | Generate human-readable explanations and maintenance recommendations for each prediction, grounded in model output and KG facts. Compare LLM explanations with and without KG grounding and check factual correctness and hallucinations. |
| Scientific evaluation, page 3 | Use suitable prediction metrics, including RMSE and the NASA scoring function for RUL, and F1-score and PR-AUC for Azure failure prediction. Measure communication cost and effects of factory count and data heterogeneity. Check explanation faithfulness and correctness; include a small human evaluation if possible. Run ablations, for example without KG, privacy, or the LLM, and report multiple runs with statistical significance testing. Discuss findings and limitations. |

The centralized pooled-data setup is an explicit research reference. It is distinct from the FL setup, where raw client data stays local. Ordinary FL is not itself the required explicit privacy mechanism. A selected mechanism must have a stated threat model and assumptions; report privacy accounting where applicable and limitations rather than claiming complete protection.

## Retained inputs and experimental boundaries

The [manifest](../data/manifest.json) records the accepted exact file hashes and row counts. [Azure](../data/Azure%20Data/) provides telemetry, machines, errors, maintenance, and failure records. [NASA C-MAPSS](../data/CMAPSSData/) provides FD001-FD004 train/test sequences and RUL files, plus its original README and research PDF. All retained bytes and manifest entries remain unchanged.

Plan Azure failure classification and NASA RUL regression as separate experimental tracks, with dataset-appropriate targets, client definitions, and metrics. Do not merge their labels or infer NASA component-maintenance facts that its files do not provide. Simulated clients are experimental groupings, not evidence from multiple real commercial organizations.

Before modeling, specify time/machine/engine split rules and prevent future information from entering features. Restrict learned preprocessing and model selection to permitted training/validation data. Keep the held-out factory out of training, tuning, and fitted transformations; define zero-shot evaluation explicitly. Compare FL and reference models with documented, comparable backbones, data budgets, seeds, and optimization settings. Record dataset/configuration revisions and artifact provenance so reported results can be reproduced.

## User delivery requirements and open decisions

The final deliverable is a complete full-stack project: a FastAPI backend and a React UI. The user selects FedAvg and FedProx as the two federated methods. Development follows [ROADMAP.md](ROADMAP.md), with completed and validated backend/models before any frontend implementation. React begins only in Phase 07 after Phase 06 acceptance; no early UI scaffold or parallel frontend work is authorized.

The [V3 data protocol](V3_DATA_PROTOCOL.md) fixes targets, purged temporal splits,
engine partitions, holdouts and feature contracts. The [V3 experiment protocol](V3_EXPERIMENT_PROTOCOL.md)
fixes matched PyTorch local/pooled references, train-only normalization, validation
checkpoint/threshold selection and retrospective final evaluation. These are
reproducible engineering/research choices, not PDF-mandated optimal hyperparameters.

The active repository contains shared ingestion and metadata interfaces, a single
PyTorch modeling/training system, safe numeric artifacts and validated internal
inference. See [new V3 results](V3_RESULTS.md) and [acceptance evidence](V3_ACCEPTANCE.md).
Earlier Phase01/02/02.1 reports remain historical records with preserved outputs;
their superseded sklearn and optimization training paths are removed. Azure is
telemetry-only by default. NASA N2 is a candidate reference with a GRU and matched
MLP control, not an unquestionable normalization solution.

No federation, explicit privacy mechanism, KG, explanation pipeline or frontend
is claimed. Phase03 has not started. A privacy-compatible zero-shot preprocessing
policy remains unresolved; offline source-normalizer transfer is not an approved
federated message. Prior held-out exposure means V3 results are retrospective.
