# FedMaint-AI agent rules

## Scope and roles

- Work on a feature branch. Preserve local changes. Do not push, merge, reset, or overwrite work without user permission.
- Codex orchestrates, checks architecture consistency, independently reviews delegated changes, runs quality gates, and accepts the result. Antigravity implements bounded tasks through the `agy-delegate` workflow; it does not commit.
- Do not run agents concurrently against the same files. Inspect partial edits before retrying a failed delegation.
- Use Claude Sonnet 4.6 for implementation by default. On a genuine model-limit error, retry with Claude Opus 4.6 Thinking, then Gemini 3.8 Flash High, subject to availability.

## Research boundaries

- Treat `docs/DECISIONS.md`, `docs/DATA_FLOW.md`, `docs/MODEL_DESIGN.md`, and `docs/EXPERIMENT_PROTOCOL.md` as proposals unless a decision is explicitly approved.
- The Azure 24-hour failure horizon, 48-hour observation window, GRU, and factory split are candidates, not finalized choices.
- Keep Azure and NASA experimental tracks separate. Do not modify raw datasets or invent measurements, outcomes, or evaluation claims.
- Avoid future information in predictive features and held-out factory outcomes in model selection. Fit preprocessing only on the appropriate training subset when modeling begins.
- Phase 2A is Azure exploratory data analysis only. Do not implement labels, model training, federated learning, privacy mechanisms, knowledge graphs, or LLMs in this phase.

## Phase 2A quality gates

- Keep reusable ingestion, schema and timestamp validation, and summaries under `src/fedmaint/data/`; keep the notebook focused on explanation and presentation.
- Test loader and validation behavior with automated tests. Use `python -m unittest discover -s tests -v` unless the project adds a documented test runner.
- Run `python scripts/prepare_data.py --verify-only --dataset azure` to verify the original CSVs.
- Execute `notebooks/01_Azure_EDA.ipynb` when dependencies permit and inspect its numerical and chart outputs.
- Review every changed file and the full diff before acceptance. Update the roadmap only for verified completed work.


## Lead orchestrator workflow

- Codex is the lead orchestrator and final reviewer. For each task, inspect the repository and requirements, prepare a complete brief, and use the installed agy-delegate workflow when delegation is suitable.
- After delegated implementation, inspect the actual diff, run relevant gates, and correct defects directly or delegate a focused fix. Repeat until acceptance criteria pass or a genuine blocker is identified. Report actual results and limitations.
- Routine, non-destructive implementation, testing, and correction on the current feature branch have standing approval; do not pause for approval at every routine step.
- If headless AGY denies a permission, do not repeat the same delegation. Implement the blocked portion directly with Codex, then validate it independently.
- Use Claude Sonnet 4.6 for AGY implementation by default. Never expand AGY permissions, enable bypass, delete user data, push, merge, or change security configuration without explicit approval.
- Keep terminal tests and Git inspection with Codex when a file-only AGY brief is appropriate. AGY does not commit. Preserve project-specific scope and research rules above.
