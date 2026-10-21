# FedKG-Maint agent workflow

## Roles and implementation

- Codex directly implements, reviews, and tests requested work. Inspect requirements and current state, identify interfaces and acceptance criteria, implement changes, inspect actual files and the full diff, and run relevant gates. Correct defects and repeat validation as needed.
- Do not run implementers concurrently against the same files.
- Do not widen permission grants, use permission-bypass flags, or modify global permissions/security settings without explicit user approval.

## Project boundaries

- This is the active clean-foundation repository. Phase 01 was explicitly authorized and implemented; see [docs/PHASE01_ACCEPTANCE.md](docs/PHASE01_ACCEPTANCE.md) for executed gates. Phase 02 requires further user authorization. Preserve the separately retained legacy repository and do not modify/remove the verified backup.
- Follow [docs/ROADMAP.md](docs/ROADMAP.md). Implement no frontend or React scaffold before Phase 06 backend/model acceptance. The complete UI belongs to Phase 07.
- Preserve original raw dataset, manifest, and project PDF bytes. Never change manifest hashes to accommodate altered inputs. Do not follow links/junctions outside intended source trees during preservation operations.
- Treat [docs/Project2_Description.pdf](docs/Project2_Description.pdf) as the academic source. Distinguish its requirements from user delivery constraints and proposed scientific decisions. Do not carry old implementation choices or results into this restart as approved facts.
- Keep Azure failure prediction and NASA RUL experiments separate. Define non-IID clients and reserve held-out factories before training. No held-out fitting, tuning, feature selection, checkpoint/threshold selection, or adaptation in zero-shot evaluation. Compute preprocessing from permitted local training/history data without future leakage.
- Do not invent measurements, performance improvements, privacy guarantees, maintenance facts, or causal explanations. Ground explanations in model outputs and traceable KG evidence. Ordinary FL alone is not an explicit privacy guarantee.

## Git and acceptance

- Prefer a main-oriented workflow without unnecessary feature branches. Do not create branches automatically. Preserve unrelated changes and the old repository history.
- Remote history rewriting, force-pushing, and remote branch deletion require an exact plan and separate explicit approval. No push, merge, destructive cleanup, or active-directory replacement is authorized merely by preparing a candidate.
- Codex reviews all changed/untracked files and staged changes before accepting work. Commit only under the user's authorized scope.
- Define real validation commands when tooling is introduced. Do not invent passing tests or commands for this documentation-only foundation. Require reproducible configs, data integrity, leakage/holdout checks, appropriate scientific evaluation, and backend API integration gates as each phase is implemented.
