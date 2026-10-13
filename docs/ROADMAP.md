# FedMaint-AI — Project Roadmap

> **Status: working plan, not a report of completed experiments.** This plan follows the nine development stages in the project proposal and adds Phase 0 for repository setup. See [DECISIONS.md](DECISIONS.md) for unresolved design choices.

## Scope

Two **independent experimental tracks** share reusable infrastructure but not one combined prediction target:

- **Track A (primary): Microsoft Azure Predictive Maintenance.** Candidate target: component/machine failure within a future time horizon (24 hours proposed, not finalized). Focus: failure prediction, local maintenance KG and grounded explanations.
- **Track B: NASA C-MAPSS.** Target: engine Remaining Useful Life (RUL) in operating cycles. Focus: time-series prediction and cross-domain FL generalization.

The datasets simulate machines and engine domains, **not multiple real organizations**. "Factories" refers to simulated federated clients.

## Phases and completion gates

| Phase | Name | Dependencies | Deliverables | Exit criteria | Status |
| --- | --- | --- | --- | --- | --- |
| 0 | Repository foundation | Uploaded datasets | Folder structure; preparation script; design/roadmap documentation | Tracked folders and documents exist; links resolve | **Completed (scaffold only)** |
| 1 | Requirements and architecture | Phase 0 | Frozen prediction tasks, client definitions, split policy, initial ontology, privacy threat model, interfaces | Team reviews and records decisions in DECISIONS.md | **In progress (draft only)** |
| 2 | Data understanding and preprocessing | Phase 1 | EDA, validation checks, feature windows, targets, non-IID client partitioning, held-out splits, client-local normalization | Reproducible data reports and leakage tests pass; label definitions documented | Not started |
| 3 | Predictive baselines | Phase 2 | Matched-model local-only and centralized experiments, first sequence model | Training and evaluation repeat from configs; baseline results logged | Not started |
| 4 | Federated learning | Phases 2–3 | FedAvg and FedProx training loops, per-client results, communication tracking | Reproducible FL runs and like-for-like comparison to baselines | Not started |
| 5 | Privacy evaluation | Phase 4 | One chosen privacy mechanism (DP or secure aggregation), threat model, risk discussion | Technique implemented and evaluated; trade-offs/limitations reported | Not started |
| 6 | Maintenance knowledge graphs | Phases 1–2 | Shared ontology; client-local KG construction; provenance-aware retrieval | Retrieval returns traceable facts without invented causal relationships | Not started |
| 7 | LLM-based explainability | Phases 3 and 6 | Explanation pipeline grounded in model output and local KG evidence | Grounded vs ungrounded explanation evaluation and error analysis | Not started |
| 8 | Evaluation and ablations | Phases 3–7 | Cross-client/generalization results, metrics, ablations, privacy/communication analysis, multiple seeds | All required comparisons and limitations documented | Not started |
| 9 | Integration and final report | Phase 8 | End-to-end research demo (if feasible), reproducibility guide, report | Demo and claims match actual experimental evidence | Not started |

Phase 6 can progress **in parallel** with Phases 3–5 after the data and ontology contracts are stable. Phase 7 requires model predictions and a working graph retriever.

## Key experimental comparisons (required by project description)

- **Local-only** vs **centralized pooled-data reference** vs **FedAvg** vs **FedProx** with matched backbones.
- **Trained factories** vs a genuinely **held-out factory** that contributes no training or hyperparameter-selection information.
- At least one **privacy mechanism**; predictive quality vs protection and overhead.
- **KG-grounded explanations** vs **ungrounded LLM explanations**; assess correctness and hallucination.
- Where meaningful, ablations for KG, privacy, and LLM, plus multiple runs and statistical testing.
- Communication volume and impact of client count and heterogeneity.

## Immediate tasks (Phase 1)

- [ ] Choose and freeze the **Azure prediction unit** (machine or component), horizon, and sampling cadence.
- [ ] Decide exactly how Azure machines are assigned to clients and which client is completely held out.
- [ ] Choose NASA training/validation/final hold-out domains and how standard train/test files will be used without leakage.
- [ ] Specify sensor windows, event join rules, early-warning policy, and local normalization.
- [ ] Select a **matched predictive backbone** for centralized/local/FL comparisons.
- [ ] Select a privacy threat model; then choose DP or secure aggregation and measurable claims.
- [ ] Define KG ontology nodes, relations, provenance and what can be shared.
- [ ] Approve first experiment protocol and log decisions in [DECISIONS.md](DECISIONS.md).

### Definition of "done"

A phase is complete only when **code/artefacts, validation, and written results** exist in the repo. A placeholder directory or planning document **does not count** as a model or pipeline implementation.

## Team workflow

Use a branch and pull request per deliverable, put configurations in \`configs/\`, store experiment descriptions in \`experiments/\`, and add automated checks to \`tests/\`. Track design changes in [DECISIONS.md](DECISIONS.md).

**References:** [Project README](../README.md), [architecture](ARCHITECTURE.md), [experiment protocol](EXPERIMENT_PROTOCOL.md).
