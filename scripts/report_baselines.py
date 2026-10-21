"""Generate a measured-results report from a completed Phase 02 plan (no training)."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]


def summary(values, scientific=False):
    if any(value is None for value in values):
        return "undefined/overflow; see per-seed JSON"
    center, deviation = mean(values), stdev(values) if len(values) > 1 else 0.
    return f"{center:.3e} ± {deviation:.3e}" if scientific else f"{center:.4f} ± {deviation:.4f}"


def report(plan):
    if plan["status"] != "completed" or any(r["status"] != "completed" for r in plan["runs"]):
        raise ValueError("Do not report an incomplete experiment as completed")
    lines = ["# Phase 02 measured baseline results", "",
             "Newly executed local results, 2026-10-21. All values below derive from `outputs/phase02/plan.json`; no historical results are imported.", "",
             f"Completed **{len(plan['runs'])} runs**, with seeds {plan['config']['seeds']}, in **{plan['runtime_seconds']:.2f} seconds** wall time. "
             f"Candidate fitting/selection totaled **{sum(r['training_seconds'] for r in plan['runs']):.2f} seconds**. "
             "All planned runs completed; no full-budget run was failed or skipped. The separate two-epoch pytest smoke plan is excluded.", "",
             "Values are arithmetic mean ± sample standard deviation across the three fixed seeds, unless explicitly per-seed. "
             "They measure optimization sensitivity on fixed splits, not confidence intervals or significance. Logistic repeats are deterministic.", "",
             "## Azure", "",
             "PR-AUC is trapezoidal area; AP is average precision. Every decision uses its validation-selected threshold. "
             "Positive prevalence and all per-seed confusion matrices/thresholds are retained in the plan and `metrics.csv`.", ""]
    groups = defaultdict(list)
    for run in plan["runs"]:
        for e in run["evaluations"]:
            if e["split"] != "validation":
                groups[(run["dataset"], run["architecture"], run["scope"], run["default_source"], e["factory_id"], e["split"])].append(e["metrics"])
    for split, heading in (("test", "Permitted known-factory temporal tests"), ("held_out", "Zero-shot held-out Azure model4")):
        lines += [f"### {heading}", "", "| Architecture | Scope | Frozen held-out source | Test factory | PR-AUC | AP | F1 | Precision | Recall | ROC-AUC |",
                  "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for key, metrics in sorted(groups.items()):
            dataset, architecture, scope, source, factory, current = key
            if dataset == "azure" and current == split:
                values = [summary([m[k] for m in metrics]) for k in ("pr_auc", "average_precision", "f1", "precision", "recall", "roc_auc")]
                lines.append(f"| {architecture} | {scope} | {source} | {factory} | " + " | ".join(values) + " |")
        lines.append("")
    lines += ["## NASA", "", "Truth is raw uncapped RUL; predictions are floored at zero but not capped. "
              "NASA score is a sum, so pool sizes matter. An overflow is retained in log/scientific form, not replaced by a finite clipped score.", ""]
    for split, heading in (("official_test", "Known-domain official terminal tests"),
                           ("held_out_official_test", "FD004 held-out official terminal test (248 engines)"),
                           ("held_out_run_to_failure", "FD004 run-to-failure window diagnostic (5,516 correlated windows)")):
        lines += [f"### {heading}", "", "| Scope | Frozen held-out source | Evaluation domain | RMSE (cycles) | MAE (cycles) | NASA score | ln(NASA score) | Overflow seeds |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for key, metrics in sorted(groups.items()):
            dataset, _, scope, source, factory, current = key
            if dataset == "nasa" and current == split:
                values = [summary([m[k] for m in metrics], scientific=k == "nasa_score") for k in ("rmse", "mae", "nasa_score", "nasa_score_log")]
                overflows = sum(m["nasa_score_overflow"] for m in metrics)
                lines.append(f"| {scope} | {source} | {factory} | " + " | ".join(values) + f" | {overflows}/{len(metrics)} |")
        lines.append("")
    lines += ["The single-condition FD001/FD003 source models extrapolate severely on FD004 under frozen source scaling. "
              "These unsuccessful zero-shot results remain part of acceptance evidence. The centralized source FD002 was chosen by training-only feature coverage before evaluation; no transform was changed after seeing these results.", "",
              "## Exact training config", "", "```json", json.dumps(plan["config"], indent=2), "```", "",
              "## Selected checkpoints", "", "Logistic validation scores are AP; MLP validation scores are AP for Azure and negative raw-window RMSE for NASA. "
              "Times include every candidate for that run, not just its selected checkpoint.", "",
              "| Run ID | Selected candidate | Epoch | Validation score | Azure threshold | Training seconds |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    experiment = ROOT / "outputs/phase02"
    for run in plan["runs"]:
        bundle = json.loads((experiment / run["model_directory"] / "model.json").read_text())
        chosen = bundle["selection"]["selected"]
        candidate = f"C={chosen['C']}" if "C" in chosen else f"lr={chosen['learning_rate']}"
        threshold = "—" if bundle["threshold"] is None else f"{bundle['threshold']:.8g}"
        lines.append(f"| {run['run_id']} | {candidate} | {chosen['epoch']} | {bundle['selection']['validation_score']:.6g} | {threshold} | {run['training_seconds']:.3f} |")
    lines += ["", "## Provenance and artifact locations", "", "```json", json.dumps(plan["provenance"], indent=2), "```", "",
              "Environment: `" + json.dumps(plan["environment"], sort_keys=True) + "`.", "",
              "The plan retains the exact source-file digest map and copied Phase 01 catalog. Each `outputs/phase02/<run-id>/` contains "
              "`model.json`, `weights.npz`, frozen `transform-*.json` files, and `predictions-*.csv`. All 138 evaluation files contain source-scoped identity, truth, and prediction; Azure also records its decision. "
              "`outputs/phase02/metrics.csv` indexes all validation/test/held-out metrics. `verification.json` records the independent reload/reproduction audit.", "",
              "Interpretation limits and the fixed training/selection/zero-shot rules are in [PHASE02_PROTOCOL.md](PHASE02_PROTOCOL.md). "
              "Acceptance commands and remaining boundaries are in [PHASE02_ACCEPTANCE.md](PHASE02_ACCEPTANCE.md). "
              "The results show baseline behavior, not a privacy guarantee, deployment readiness, statistical superiority, or Phase 03 completion.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/PHASE02_RESULTS.md")
    args = parser.parse_args()
    plan = json.loads((ROOT / "outputs/phase02/plan.json").read_text())
    args.output.write_text(report(plan), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
