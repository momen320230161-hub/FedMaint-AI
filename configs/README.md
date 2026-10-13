# Experiment Configurations

**Scaffold only.** No training configuration or model implementation has been added yet.

This directory will hold **versioned experiment settings** for the Azure and NASA tracks, including dataset name, target definition, feature window, client split, model backbone, seed, optimizer, FL strategy, privacy settings and evaluation policy.

## Future files (not yet created)

- `azure_baseline.yaml` — local/centralized baseline settings
- `azure_fedavg.yaml`, `azure_fedprox.yaml` — matched federated experiments
- `nasa_rul.yaml` — NASA RUL benchmarking
- `privacy_*.yaml` — mechanism and budget settings

Set values only after [Phase 1 decisions](../docs/DECISIONS.md) are approved. All experiments must record the config and Git revision.
