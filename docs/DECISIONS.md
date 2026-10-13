# Architecture and Research Decision Log

This document makes assumptions explicit. **Proposal** is not the same as **approved** or **implemented**.

| ID | Decision | Current position | Status |
| --- | --- | --- | --- |
| D-001 | Dataset tracks | Azure classification main track; NASA C-MAPSS separate RUL benchmark | Proposed |
| D-002 | Azure label | Any-component machine failure within next 24h; observation window 48h | **Candidate** — requires time-boundary and class-balance review |
| D-003 | Azure participating FL clients | Group known machines by model2, model3, model4 as 3 non-IID clients | **Candidate** — inspect positive window counts first |
| D-004 | Unseen factory | All 16 model1 machines withheld from global training and hyperparameter search | **Candidate** — fixed only after protocol review |
| D-005 | Predictive backbone | GRU sequence encoder, dataset-specific classifier/RUL head; matched within each strategy | **Candidate** — not benchmarked |
| D-006 | Privacy mechanism | DP or secure aggregation; threat model first; avoid claiming ordinary FL guarantees privacy | Open |
| D-007 | Graph architecture | Shared ontology, client-local event facts, provenance-aware retrieval; no initial GNN requirement | **Candidate** |
| D-008 | LLM evidence policy | Model output + verified local KG facts; compare grounded vs ungrounded | Required, hosting/rubric open |
| D-009 | Dataset separation | Do not merge Azure and NASA feature/label tables or reuse their model weights | Proposed |
| D-010 | NASA windows | 30 cycles as first tested window length; per-cycle RUL reconstruction | **Candidate** |
| D-011 | Unseen-client scaling | Strict zero-shot evaluation needs fixed compatible preprocessing; optional local unsupervised adaptation reported separately | Open |
| D-012 | Client-count study | Repartition training machines into additional whole-machine clients as a secondary experiment | **Candidate** |

## Dataset-based evidence for proposed FL clients

The uploaded Azure machine metadata gives **16 / 17 / 35 / 32** machines in model1 / model2 / model3 / model4. Joining recorded failure events gives **189 / 168 / 221 / 183** failure records respectively (total **761**). This justifies testing model-based non-IID grouping, but **does not establish per-window class balance, the best held-out model, or predictive performance**.

The proposal to hold out model1 is a **working choice, not a finalized benchmark**. In Phase 2, check class prevalence and window feasibility without using held-out outcomes for tuning.

**Design references:** [DATA_FLOW.md](DATA_FLOW.md), [MODEL_DESIGN.md](MODEL_DESIGN.md), [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

## Decision record template

### D-XXX — Title
- **Status:** Proposed / Approved / Superseded
- **Context:** What we need to decide and why
- **Options:** Alternatives and trade-offs
- **Decision:** Agreed choice
- **Evidence:** Dataset inspection, experiment, or source that supports it
- **Impact:** What files/protocols must change
- **Date / review:** YYYY-MM-DD, owner/reviewers

## Guardrails

Do not turn an open design value into a completed implementation claim. Keep data-extract revisions, split policies, privacy budgets, and leakage rules in the experiment records. Review new choices before training.
