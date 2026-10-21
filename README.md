# FedKG-Maint

Privacy-preserving predictive maintenance using federated learning, factory-local knowledge graphs, and evidence-grounded LLM explanations and maintenance recommendations.

This is the active minimal clean foundation. It contains original research inputs and project documentation. Phase 01 has not started; there is no backend, trained model, experiment result, or UI in this candidate. Previous implementation and research artifacts remain in the separately verified Phase 00 backup and the separately preserved legacy repository.

Read the [original academic brief](docs/Project2_Description.pdf), [project overview](docs/PROJECT_OVERVIEW.md), [development roadmap](docs/ROADMAP.md), and [agent workflow](AGENTS.md).

The retained inputs are:

- [Microsoft Azure Predictive Maintenance](data/Azure%20Data/): five original CSV files.
- [NASA C-MAPSS](data/CMAPSSData/): twelve numeric data files and its original source documentation.
- [Verified manifest](data/manifest.json): accepted exact SHA-256 hashes and row counts. Preserve these files byte-for-byte, including existing line endings; do not rewrite hashes to accept changed data.

Development proceeds through data/backend, baselines, federated learning, explicit privacy, local KGs/LLM explanations, and scientific/backend validation. React implementation is exclusively Phase 07, after the backend and models pass Phase 06 acceptance. The final deliverable is a complete full-stack project.

No installation or application commands exist yet. Tooling, dependencies, and runnable verification commands will be established during an authorized Phase 01. AGY remains the default implementation agent; Codex orchestrates and reviews.
