# Architecture and Research Decision Log

This document makes assumptions explicit. **Proposal** is not the same as **approved** or **implemented**.

| ID | Decision | Current position | Status |
| --- | --- | --- | --- |
| D-001 | Dataset tracks | Azure primary failure-prediction experiment; NASA C-MAPSS separate RUL benchmark | Proposed |
| D-002 | Azure target | Machine failure within next 24 hours, precise boundaries TBD | Open |
| D-003 | Client definition | Azure grouped by machine model or another non-IID split; NASA grouped by FD domain | Open |
| D-004 | Held-out factory/domain | Entire unseen client excluded from training and tuning | Required, selection open |
| D-005 | Baseline architecture | Matched backbone across local, centralized, FedAvg, FedProx comparisons | Required, backbone open |
| D-006 | Privacy mechanism | Differential privacy or secure aggregation; threat model first | Open |
| D-007 | Graph architecture | Shared ontology; client-local facts; evidence provenance mandatory | Proposed |
| D-008 | LLM evidence policy | Predictions + verified graph facts only; compare grounded vs ungrounded | Required, rubric open |
| D-009 | Model/data use | Keep Azure and NASA labels/experiments separate | Proposed |

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
