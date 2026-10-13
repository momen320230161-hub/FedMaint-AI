# FedMaint-AI
### Privacy-Preserving Predictive Maintenance with Federated Learning, Knowledge Graphs, and LLMs

> **Project status:** Research and implementation planning. The architecture, methods, and experiments below describe the intended work—not completed models or validated performance claims.

## Overview

Industrial equipment continuously generates sensor measurements, operating logs, maintenance events, and failure histories. Predictive maintenance uses this information to anticipate failures and reduce unexpected downtime. However, real organizations may be unable to pool equipment data because of privacy, confidentiality, or organizational constraints.

**FedMaint-AI** explores a collaborative approach where simulated factories train predictive-maintenance models using **federated learning (FL)** without sending their raw training data to a central server. It also explores **privacy-preserving model aggregation**, a **maintenance knowledge graph (KG)**, and **large language models (LLMs)** for evidence-grounded explanations and maintenance recommendations.

The research objective is to evaluate **predictive performance, generalization to previously unseen factories, privacy–utility trade-offs, communication cost, and explanation faithfulness**. This is a research prototype, not an industrial safety or maintenance decision system.

## Project objectives

1. Predict machine failures over a defined future horizon and/or estimate **remaining useful life (RUL)**.
2. Simulate multiple factories with **non-identically distributed (non-IID)** data.
3. Compare **local-only**, **centralized**, **FedAvg**, and **FedProx** training under controlled conditions.
4. Evaluate a privacy mechanism such as **differential privacy** or **secure aggregation**; distinguish FL from a formal privacy guarantee.
5. Represent machines, components, sensors, errors, failures, and maintenance events through a shared **ontology** and **local knowledge graphs**.
6. Generate explanations and candidate maintenance actions grounded in actual model outputs and retrieved KG facts.
7. Evaluate generalization on a **held-out factory** that was never used to train the global model.
8. Report results across multiple runs, appropriate baselines, and ablation studies.

## Proposed system architecture

```mermaid
flowchart TB
    subgraph FACTORIES["Factory clients - raw data stays local"]
        direction LR
        A["Factory A: data, model, local KG"]
        B["Factory B: data, model, local KG"]
        C["Factory C: data, model, local KG"]
    end
    S["Federated server: FedAvg / FedProx + privacy mechanism"]
    A -->|"Protected model updates"| S
    B -->|"Protected model updates"| S
    C -->|"Protected model updates"| S
    S -->|"Updated model parameters"| A
    S -->|"Updated model parameters"| B
    S -->|"Updated model parameters"| C

    subgraph INFERENCE["Inference and explanation at each factory"]
        direction TB
        D["Local sensor readings"] --> P["Local predictive model"]
        P --> R["Failure risk or remaining useful life"]
        K["Local maintenance knowledge graph"] --> Q["Retrieve supporting facts"]
        R --> L["Grounded LLM explanation"]
        Q --> L
        L --> O["Prediction + evidence + candidate maintenance action"]
    end
    S -.->|"Shared model architecture and weights"| P
```

**Privacy boundary:** Telemetry and identifiable maintenance records are intended to remain at each client. Model updates are not inherently private; the design must evaluate possible leakage from both updates and any shared knowledge-graph content. The exact placement and threat model of the privacy mechanism will be documented before experiments.

## Datasets

The two datasets support **separate experiments**. Their labels, measurement scales, and underlying processes differ; they should not be casually combined into one training table.

### 1. Microsoft Azure Predictive Maintenance — primary end-to-end benchmark

**Source:** [Microsoft Azure Predictive Maintenance on Kaggle](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)

| File | Records in the supplied archive | Purpose |
| --- | ---: | --- |
| `PdM_telemetry.csv` | 876,100 | Hourly voltage, rotation, pressure, vibration |
| `PdM_errors.csv` | 3,919 | Machine error events |
| `PdM_maint.csv` | 3,286 | Component replacement / maintenance history |
| `PdM_failures.csv` | 761 | Component failure events |
| `PdM_machines.csv` | 100 | Machine metadata, model, and age |

There are **100 simulated machines** across **four machine models** in the supplied data. A **proposed** task is to predict failure within the next 24 hours, with the prediction horizon and label definition fixed *before* model training. Train/validation/test splits must respect time and machine identities. Maintenance and failure records must never leak future information into predictors.

**Why Azure first?** Its linked machine, component, error, failure, and maintenance tables provide a practical foundation for the KG and explanation pipeline. It is still a **synthetic/simulated benchmark**, not evidence of deployment readiness.

### 2. NASA C-MAPSS — RUL and cross-domain FL benchmark

**Source:** [NASA Open Data Portal: CMAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

C-MAPSS contains engine run-to-failure sequences with operational settings and sensor measurements. Each subset includes `train_FD00x.txt`, `test_FD00x.txt`, and `RUL_FD00x.txt`.

| Subset | Training rows | Testing rows | Operating conditions | Fault modes |
| --- | ---: | ---: | ---: | ---: |
| FD001 | 20,631 | 13,096 | 1 | 1 |
| FD002 | 53,759 | 33,991 | 6 | 1 |
| FD003 | 24,720 | 16,596 | 1 | 2 |
| FD004 | 61,249 | 41,214 | 6 | 2 |

A proposed task is **RUL regression** in operating cycles. The NASA test labels indicate remaining cycles *after each test engine's last observed cycle*; they are not pointwise labels for every earlier cycle without further reconstruction. The FD subsets differ materially, so leave-one-domain-out results must be interpreted carefully and reported separately.

**Note:** C-MAPSS does not contain the same explicit maintenance-action history as Azure; it should not be used to invent component-level maintenance explanations.

## Dataset availability and redistribution

**Raw dataset files are intentionally not tracked in this public repository.** The supplied archives were integrity-checked locally, and their file hashes and record counts are recorded in [`data/manifest.json`](data/manifest.json). This enables reproducible local preparation without claiming that the datasets may be republished.

- The Kaggle Azure dataset currently lists its **license as “Unknown.”**
- The NASA dataset resource page does **not specify a license**.
- A publicly accessible download is **not the same as an explicit grant of redistribution rights**.
- Large raw files also create Git repository maintenance and hosting problems. GitHub recommends keeping individual objects small and enforces a 100 MiB single-object limit.

**Download from the original publishers** and check the applicable terms before redistributing any source files. This repository's own future code license, if added, will not automatically apply to third-party data.

### Prepare from the project's source ZIP files

Requires **Python 3.9+**, using only the Python standard library.

```bash
git clone https://github.com/momen320230161-hub/FedMaint-AI.git
cd FedMaint-AI

# Replace these paths with your actual downloaded ZIP locations.
python scripts/prepare_data.py \
  --azure-zip "/path/to/archive (2).zip" \
  --nasa-zip "/path/to/CMAPSSData.zip"

# Recheck the unpacked datasets at any time:
python scripts/prepare_data.py --verify-only
```

On success, the script extracts the expected files into:

```text
data/
├── manifest.json             # Tracked: expected filenames, counts, SHA-256
├── raw/                      # Local only, ignored by Git
│   ├── azure/
│   │   ├── PdM_telemetry.csv
│   │   ├── PdM_errors.csv
│   │   ├── PdM_maint.csv
│   │   ├── PdM_failures.csv
│   │   └── PdM_machines.csv
│   └── nasa/
│       ├── train_FD001.txt ... train_FD004.txt
│       ├── test_FD001.txt  ... test_FD004.txt
│       └── RUL_FD001.txt   ... RUL_FD004.txt
└── processed/                # Future generated features; also ignored
```

The script verifies each supplied ZIP against its expected **SHA-256** and validates every extracted source file's hash and record count. If your download is a different upstream revision or is repackaged differently, a ZIP-level mismatch does not by itself mean corruption: review the provenance and update the manifest only after independent validation.

## Planned implementation

| Phase | Work | Expected deliverable |
| --- | --- | --- |
| 1. Requirements and design | Prediction tasks, privacy boundary, ontology, experimental splits | Architecture + experiment specification |
| 2. Data preparation | EDA, missing-value checks, local normalization, windowing, leakage-safe labels | Reproducible feature pipeline |
| 3. Baselines | Local and pooled centralized reference models | Comparable baseline results |
| 4. Federated learning | Same predictive backbone with FedAvg and FedProx | Multi-client FL pipeline |
| 5. Privacy | Explicit threat model; DP or secure aggregation; utility analysis | Privacy–utility experiments |
| 6. Knowledge graphs | Shared ontology, client-local KG, evidence provenance | KG construction and retrieval |
| 7. LLM explanations | Ground model predictions in retrieved graph facts | Explainability pipeline |
| 8. Evaluation | Held-out clients/domains, ablations, multi-seed runs, communication cost | Reproducible results and analysis |
| 9. Integration | Research demo, documentation, final write-up | End-to-end project |

### Important experimental controls

- **Non-IID clients:** For Azure, partition machines (potentially by machine model), checking for sufficient client sample and failure counts. For NASA, FD subsets are candidate domains, not actual independent factories.
- **Held-out factory:** Reserve one distinct client or domain *before training and tuning*. Do not include its model updates, examples, normalization statistics, or validation labels in training.
- **Leakage prevention:** Compute features from observations available **at prediction time**. Keep future failures, post-failure replacements, and cross-split transformations out of inputs.
- **FL fairness:** Compare centralized, local-only, FedAvg, and FedProx using a matched model backbone and clearly defined data budgets and optimization settings.
- **Local normalization:** Learn sensor normalization parameters at each client without aggregating raw sensor values.
- **Privacy accountability:** Specify the threat model, update clipping, privacy accounting (if DP is selected), and remaining leakage limitations.
- **Grounding:** Only retrieve supported KG facts. Distinguish correlation, model attribution, and demonstrated causality. Withhold unsupported maintenance advice.

## Evaluation plan

| Task | Candidate metrics |
| --- | --- |
| Azure failure prediction | PR-AUC, F1, precision, recall, confusion matrix, calibration |
| NASA RUL prediction | RMSE, MAE, NASA asymmetric scoring function |
| Federated optimization | Per-client performance, convergence, communication volume, rounds |
| Generalization | Results on strictly held-out factories/domains |
| Privacy | Privacy budget and predictive utility (if DP); protocol assumptions (if secure aggregation) |
| KG + LLM | Evidence coverage, factual correctness, faithfulness, unsupported-claim rate |
| Ablations | Without FL, privacy, KG retrieval, and/or LLM explanations (where meaningful) |

Use multiple random seeds and report variability. No numerical accuracy, privacy guarantee, or explanation-quality improvement is claimed until experiments are performed.

## Planned repository structure

```text
FedMaint-AI/
├── README.md
├── .gitignore
├── data/
│   └── manifest.json
├── scripts/
│   └── prepare_data.py
├── src/                     # Planned: preprocessing, models, FL, privacy, KG, LLM
├── experiments/             # Planned: configs, evaluation, ablations
├── tests/                   # Planned: leakage and pipeline checks
└── docs/                    # Planned: design decisions and project report
```

Only files already committed should be treated as implemented; the remaining folders show the intended organization.

## Research questions

1. When does FL improve cross-factory predictive performance relative to local-only training?
2. How do FedAvg and FedProx perform as operating conditions and client data distributions diverge?
3. What is the measured cost of added privacy protection in prediction quality and communication?
4. Does KG-grounded retrieval improve the correctness and traceability of LLM maintenance explanations?
5. What fails when the model is evaluated on a factory/domain absent from training?

## Original data and related references

- [Microsoft Azure Predictive Maintenance dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
- [NASA C-MAPSS simulated data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- [NASA C-MAPSS research reference — Saxena et al., 2008](https://www.phmsociety.org/node/21)
- [GitHub repository file-size guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

---

**Maintainers:** FedMaint-AI project team.  
**Disclaimer:** Research/educational use; not a safety-certified maintenance decision tool.
