# FedKG-Maint agent workflow

## Roles and delegation

- Use the installed `agy-delegate` skill for every coding or implementation task, including small tasks, unless the user explicitly authorizes Codex to implement directly. A one-task exception applies only to its approved scope.
- Codex is orchestrator and independent reviewer: inspect requirements and current state, identify interfaces and acceptance criteria, prepare a self-contained brief, dispatch AGY through the registered `codex-background` relay, wait for its result, inspect actual files and the full diff, and independently rerun relevant gates. Request focused AGY corrections when needed.
- AGY implements using its configured default model. Never pass `--model`, hardcode a replacement model, or switch models without explicit user approval. AGY must not stage, commit, push, merge, or change security/permission settings.
- Keep the original background-relay request pending until completion. Do not poll logs or status through model-driven loops. Reattach interrupted waits using the same run ID; do not redispatch an existing job.
- If AGY, authentication, the relay, or the workspace is unavailable, report the blocker and request missing setup or a one-task exception. Never silently substitute another implementer.
- On a headless permission denial, inspect partial edits, report the denied operation, and stop that delegation. Do not repeat the same dispatch, widen grants, use permission-bypass flags, or modify global permissions/security settings.
- Codex may perform audits, provenance-preserving copies, validation, briefs, and orchestration documentation. Application implementation remains AGY-first. Do not run implementers concurrently against the same files.

## Project boundaries

- This is the active clean-foundation repository. Phase 01 requires explicit authorization and has not started. Preserve the separately retained legacy repository and do not modify/remove the verified backup.
- Follow [docs/ROADMAP.md](docs/ROADMAP.md). Implement no frontend or React scaffold before Phase 06 backend/model acceptance. The complete UI belongs to Phase 07.
- Preserve original raw dataset, manifest, and project PDF bytes. Never change manifest hashes to accommodate altered inputs. Do not follow links/junctions outside intended source trees during preservation operations.
- Treat [docs/Project2_Description.pdf](docs/Project2_Description.pdf) as the academic source. Distinguish its requirements from user delivery constraints and proposed scientific decisions. Do not carry old implementation choices or results into this restart as approved facts.
- Keep Azure failure prediction and NASA RUL experiments separate. Define non-IID clients and reserve held-out factories before training. No held-out fitting, tuning, feature selection, checkpoint/threshold selection, or adaptation in zero-shot evaluation. Compute preprocessing from permitted local training/history data without future leakage.
- Do not invent measurements, performance improvements, privacy guarantees, maintenance facts, or causal explanations. Ground explanations in model outputs and traceable KG evidence. Ordinary FL alone is not an explicit privacy guarantee.

## Git and acceptance

- Prefer a main-oriented workflow without unnecessary feature branches. Do not create branches automatically. Preserve unrelated changes and the old repository history.
- Remote history rewriting, force-pushing, and remote branch deletion require an exact plan and separate explicit approval. No push, merge, destructive cleanup, or active-directory replacement is authorized merely by preparing a candidate.
- Codex reviews all changed/untracked files and staged changes before accepting work. Commit only under the user's authorized scope; AGY never commits.
- Define real validation commands when tooling is introduced. Do not invent passing tests or commands for this documentation-only foundation. Require reproducible configs, data integrity, leakage/holdout checks, appropriate scientific evaluation, and backend API integration gates as each phase is implemented.
