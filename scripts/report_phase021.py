"""Report completed frozen Phase 02.1 evidence; never train or evaluate a model."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean,stdev
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fedmaint.optimization.workflow import verify_freeze
from fedmaint.optimization.common import digest,write_json


def fmt(value):
    return "undefined/overflow" if value is None else f"{value:.5g}"


def spread(values):
    if any(v is None for v in values):
        return "overflow; see per-seed score/log"
    return f"{mean(values):.5g} +/- {stdev(values) if len(values)>1 else 0:.4g}"


def table(lines,head,rows):
    lines.extend(["| "+" | ".join(head)+" |","| "+" | ".join(["---"]*len(head))+" |"])
    lines.extend("| "+" | ".join(str(v) for v in row)+" |" for row in rows)
    lines.append("")


def report():
    out=ROOT/"outputs/phase021"
    frozen=verify_freeze(ROOT,out)
    read=lambda name:json.loads((out/name).read_text())
    selection,azure,evaluation,budget,verification=map(read,("selection.json","azure.json","evaluation.json","budget.json","verification.json"))
    assert evaluation["status"]==selection["status"]==azure["status"]=="completed"
    assert verification["status"]=="verified"
    # Historical held-out results are opened only after the final evaluation.
    historical=json.loads((ROOT/"outputs/phase02/plan.json").read_text())
    lines=["# Phase 02.1 measured results","", "Executed 2026-10-22–23 on main. All values derive from saved experiments; no project beyond Phase 02.1 was implemented.","",
           "FD004 had already been observed in Phase 02: this evaluation is **not fully blind**. FD001–FD003 selection folds are exploratory and reused for selection/confirmation, not independent unbiased generalization estimates. Two seeds measure optimization variability, not statistical significance.","",
           "**Execution deviation:** the historical full-suite preparation/smoke tests also accessed FD004/model4 before this freeze. Their held-out metrics were not inspected or used for selection; the new candidate loader remained gated. Nevertheless the strict no-access-before-freeze rule was not fully met. See outputs/phase021/protocol-deviations.json. The new candidate panel itself was evaluated exactly once after freeze. Do not describe this execution as protocol-perfect.","",
           f"**{len(budget['fits'])} completed research fits**, {budget['training_seconds']:.2f} seconds charged fitting/selection/artifact time, within 69 fits / 2700 seconds. No failed, omitted, or retried research candidate. Test-only fits are excluded.","",
           "## Architectures and fixed protocol","",
           "MLP: flatten 30 x 24 observations, append 30 boolean validity masks, Linear(750,32), ReLU, Linear(32,1): **24,065 trainable parameters**. GRU: 24 inputs, one unidirectional 32-unit layer, no dropout, final valid hidden state, Linear(32,1): **5,601 parameters**. Left padding is stripped before packing; tests compare mixed-length batches with individual unpadded sequences.","",
           "Both use Adam lr=0.001, batch=128, alpha=0.0001, 40 epochs maximum, patience=6, deterministic CPU/one thread; target divided by100 only for optimization. Loss: half MSE plus alpha/(2*Nfit) times the sum of squared weight tensors (no bias penalty). Raw uncapped truth is used everywhere. Checkpoints use inner permitted-source validation engines at cycles50/100. Equal epoch limits do not imply equal FLOPs.","",
           "N0 is local train-only mean/std with neutral constant coordinates. N1 is local median/IQR with the predeclared q01/q99 stability floor and asinh compression. N2 adds local condition-bin sensor statistics at fixed resolutions[1,0.1,1], minimum128 rows/bin; unsupported bins fall back to source-global N1. All three output the fixed sorted24-coordinate schema. The full approved method is unchanged in [PHASE02_1_RESEARCH_PLAN.md](PHASE02_1_RESEARCH_PLAN.md).","",
           "Primary new-model outputs are raw, including negatives. Zero-floored metrics are separate diagnostics. There is no upper prediction cap or ground-truth cap. NASA score is the official asymmetric **sum**; overflow is retained as null plus natural-log/scientific representations.","",
           "## Source-only normalization and architecture selection","",
           f"Shortlisted normalization: **{selection['shortlisted_normalizer']}**. Selected configuration: **{selection['winner']}**. Candidates passing all promotion gates in both seeds: `{selection['promoted']}`. The gate requires lower macro RMSE, no worse worst-domain RMSE, and improvement in at least2/3 domains versus N0-MLP. A normalization shortlist is not itself evidence of a supported improvement.",""]
    rows=[]
    for seed,stage in ((42,"screen"),(43,"confirmation")):
        for name,s in selection[stage].items():
            rows.append([seed,name,fmt(s['macro_rmse']),fmt(s['worst_domain_rmse']),*(fmt(s['fold_rmse'][d]) for d in ("FD001","FD002","FD003"))])
    table(lines,["Seed","Candidate","Macro RMSE","Worst domain","FD001","FD002","FD003"],rows)
    lines.extend(["### Matched output-policy and normalization checks",""])
    rows=[]
    for seed in (42,43):
        for candidate in sorted({r['normalizer']+'-'+r['architecture'] for r in selection['runs'] if r['seed']==seed}):
            rs=[r for r in selection['runs'] if r['seed']==seed and r['normalizer']+'-'+r['architecture']==candidate]
            rows.append([seed,candidate,*(fmt(mean(r['outer']['metrics'][str(c)][policy]['rmse'] for r in rs for c in (50,100))) for policy in ('raw','zero_floor_diagnostic'))])
    table(lines,["Seed","Candidate","Raw macro RMSE (selection)","Zero-floor macro RMSE (diagnostic only)"],rows)
    lines.extend(["N2-MLP versus N0-MLP isolates the normalization change within the new MLP protocol and passes the source promotion gates in both seeds, yet retains severe FD002-held-out errors. N2-GRU versus N2-MLP is a matched architecture comparison: GRU has lower RMSE in all three folds in both seeds. N2-GRU versus N1-GRU is supported only by the screening seed; the separate benefit of condition-aware bins beyond N1 has no second-seed confirmation in this bounded plan. N0-GRU also has only one screening seed. Do not extrapolate a universal ordering or attribute historical-versus-new differences solely to the GRU.","",
                  "### N2 outer-domain unsupported-regime diagnostics",""])
    table(lines,["Seed","Architecture","Held domain","Frozen source","Unsupported rows / all rows"],
          [[r['seed'],r['architecture'],r['held_out'],r['source'],f"{r['diagnostic']['unsupported_rows']}/{r['diagnostic']['rows']}"] for r in selection['runs'] if r['normalizer']=='N2'])
    lines.extend(["Each fold averages cycle50/100 RMSE, then folds receive equal weight. Pseudo-held clients never fit transforms/checkpoints or source routing. Permitted clients retain deterministic80/20 engine splits; unique rows fit local statistics. Training retains the same per-record objective. Normalization/architecture/floor/window/router constants were not changed from these outcomes.","",
                  "### Per-fold, per-endpoint metrics","", "Each row is one prediction per eligible engine at a fixed endpoint. Maximum errors, error p95, predictions, negative fractions and endpoint eligibility are retained in selection.json and per-run CSV files.",""])
    rows=[]
    for r in selection['runs']:
        for c,g in r['outer']['metrics'].items():
            m=g['raw'];f=g['zero_floor_diagnostic']
            rows.append([r['seed'],r['normalizer']+'-'+r['architecture'],r['held_out'],c,m['samples'],fmt(m['rmse']),fmt(m['mae']),m['nasa_score_scientific'],fmt(g['negative_rate']),fmt(f['rmse'])])
    table(lines,["Seed","Candidate","Held domain","Cycle","n","Raw RMSE","Raw MAE","Raw NASA sum","Negative fraction","Floor RMSE diagnostic"],rows)
    lines.extend(["## Final frozen NASA evaluations","",f"Freeze SHA-256: `{digest(out/'freeze.json')}`. Final model panel: `{frozen['panel']}`. All source-only models, private transforms and routing choices were sealed before any new official-test/FD004 loading. The exclusive evaluation marker prevents silent repeats. Final outcomes did not change the winner.","",
                  "Local models use their own transform. Pooled known-domain tests use that training client's transform; zero-shot pools use the frozen maximum-informative-training-coordinate source (lexical ties), independent of target statistics. Official terminal pools and correlated run-to-failure windows are kept separate.","",
                  "### FD004 official terminal pool: per-seed results","", "All rows use the same248 engines and raw uncapped truth. Floor diagnostics use the same zero floor as historical Phase02, but other training/selection protocols still differ.",""])
    rows=[];flat=[]
    for r in evaluation['runs']:
        g=r['metrics']['all'];m=g['raw'];f=g['zero_floor_diagnostic']
        flat.append({k:r[k] for k in ('candidate','scope','seed','domain','pool','inference_source')}|{'n':m['samples'],**{'raw_'+k:v for k,v in m.items()},**{'floor_'+k:v for k,v in f.items()},'negative_fraction':g['negative_rate'],'max_absolute_error':g['max_absolute_error'],'error_p95':g['absolute_error_p95']})
        if r['pool']=='held_out_official_test':
            rows.append([r['candidate'],r['scope'],r['seed'],r['inference_source'],fmt(m['rmse']),fmt(m['mae']),m['nasa_score_scientific'],fmt(g['negative_rate']),fmt(f['rmse']),fmt(f['mae']),f['nasa_score_scientific']])
    table(lines,["Candidate","Scope","Seed","Source","Raw RMSE","Raw MAE","Raw NASA sum","Negative fraction","Floor RMSE","Floor MAE","Floor NASA sum"],rows)
    pd.DataFrame(flat).to_csv(out/'nasa-metrics.csv',index=False,float_format='%.17g')
    lines.extend(["### Historical versus optimized NASA","", "Historical Phase02 uses zero-floored predictions, a different optimizer/hyperparameter search and window-based validation; Phase02.1 uses fixed learning rate and prefix-based validation. These comparisons involve different protocols and cannot isolate architecture/normalization effects. Matched Phase02.1 source comparisons above support attribution. Historical all-three-seed results remain unchanged in PHASE02_RESULTS.md; this table matches seeds42/43 only.",""])
    historical_groups=defaultdict(list)
    for r in historical['runs']:
        if r['dataset']=='nasa' and r['seed'] in (42,43):
            for e in r['evaluations']:
                if e['split'] in ('official_test','held_out_official_test'):
                    historical_groups[(r['scope'],r['default_source'].replace('nasa-',''),e['factory_id'].replace('nasa-',''))].append(e['metrics'])
    new_groups=defaultdict(list)
    for r in evaluation['runs']:
        if r['pool'] in ('official_test','held_out_official_test'):
            new_groups[(r['candidate'],r['scope'],r['domain'])].append(r['metrics']['all'])
    rows=[]
    for (scope,source,domain),ms in sorted(historical_groups.items()):
        rows.append(['Phase02 historical floor','pooled' if scope=='centralized' else source,domain,*(spread([m[k] for m in ms]) for k in ('rmse','mae','nasa_score'))])
    for (candidate,scope,domain),gs in sorted(new_groups.items()):
        rows.append([candidate+' raw',scope,domain,*(spread([g['raw'][k] for g in gs]) for k in ('rmse','mae','nasa_score'))])
    table(lines,["Protocol/candidate","Scope","Domain","RMSE mean +/- SD","MAE mean +/- SD","NASA sum mean +/- SD"],rows)
    lines.extend(["### All final NASA pools and per-seed variability","", "Sample SD above is across two seeds. The following table and nasa-metrics.csv preserve every known-domain and run-to-failure result; correlated windows are not independent engines.",""])
    rows=[]
    for r in evaluation['runs']:
        m=r['metrics']['all']['raw']
        rows.append([r['candidate'],r['scope'],r['seed'],r['domain'],r['pool'],m['samples'],fmt(m['rmse']),fmt(m['mae']),m['nasa_score_scientific'],m['nasa_score_overflow']])
    table(lines,["Candidate","Scope","Seed","Domain","Pool","n","Raw RMSE","Raw MAE","Raw NASA sum","Overflow"],rows)
    lines.extend(["## Azure source-model availability ablations","", "Only model1/2/3 enter these18 fixed fits. Each pseudo-held model's temporal test rows are excluded from all fitting/checkpoint/threshold decisions. Feature groups/lags have identical machine/timestamp populations; each arm refits imputation, scaling and feature selection on permitted training rows. Validation-only thresholds optimize F1; no model4 score/statistic is used. All fits use seed42, so no Azure multi-seed robustness is claimed.",""])
    lines.extend(["Azure controls use sklearn MLPClassifier: d locally retained/aligned inputs →32 ReLU→1 sigmoid, with32*d+65 trainable parameters; d is listed per arm/fold below. Adam lr0.001, batch128, alpha0.0001, at most40 epochs/patience6; inner validation AP selects the checkpoint. This classification objective is separate from the matched NASA regression objective.",""])
    rows=[]
    for r in azure['runs']:
        m=r['metrics']
        rows.append([r['arm'],r['held_model'],len(r['features']),32*len(r['features'])+65,m['samples'],fmt(m['positive_prevalence']),fmt(m['average_precision']),fmt(m['pr_auc']),fmt(m['f1']),fmt(m['precision']),fmt(m['recall']),fmt(m['threshold']),str(m['confusion_matrix'])])
    table(lines,["Arm","Held source model","d","Parameters","n","No-skill AP","AP","PR-AUC trapezoid","F1","Precision","Recall","Inner threshold","Confusion TN,FP/FN,TP"],rows)
    lines.extend(["### Feature-group summaries (equally weighted source folds)",""])
    rows=[]
    for arm in sorted({r['arm'] for r in azure['runs']}):
        rs=[r for r in azure['runs'] if r['arm']==arm]
        rows.append([arm,*(fmt(mean(r['metrics'][k] for r in rs)) for k in ('average_precision','pr_auc','f1'))])
    table(lines,["Arm","Macro AP","Macro PR-AUC","Macro F1"],rows)
    lines.extend(["Events use (t-24h,t-lag] and telemetry ends at t; targets use failures in (t,t+24h]. Sampling is daily at the fixed grid; full history and horizon are purged across chronological cuts. Exact-time exclusion, future-event mutation, duplicate rejection and backdated-posting tests passed. Maintenance action versus logging time cannot be separated: posting/ingestion times are absent. Lag6/12 sensitivity and event ablations do not prove deployment availability. Synthetic-dataset structure and grid/event alignment may influence performance. **Use telemetry-only as the conservative reference until posting-time availability is established**, regardless of full-feature AP.","",'Source-only event timing audit: `'+json.dumps(azure['source_event_timing'],sort_keys=True)+'`.',"",
                  "## Leakage, federation and privacy boundaries","", "Source-only diagnostics reproduced all six frozen-scaler comparisons. Near-constant source scales and operating-condition changes amplify inputs (FD001→FD002 |z|p99≈19214; FD003→FD002≈19173). This supports a scaling-related failure mechanism, not the claim that scaling alone explains RUL failures. No FD004 outcome/statistic/prediction entered selection. Azure model4 was filtered before validation/sample generation; raw CSV ingestion necessarily parses records before filtering, without computing model4 statistics.","",
                  "Neural NPZ checkpoints contain only common named tensors. Medians/scales/constant masks/regime dictionaries are separate private JSON states. The fixed24-coordinate contract and same tensor names/shapes are mechanically suitable for later FedAvg/FedProx; shared coordinates do not establish equivalent local semantics or accuracy after aggregation. Centralized pooling is a non-private reference.","",
                  f"Phase03 recommendation: retain **{selection['winner']}** as the source-gated backbone. GRU remains a compact alternative only to the extent matched source evidence supports it; target scores cannot promote it. Keep30 chronological steps, sorted24 inputs, neutral constants and boolean masks; aggregate/proximal-penalize neural weights only. No federated implementation has started.","",
                  "The overall academic privacy requirement is **not yet satisfied**. Local fitting is compatible, but no formal privacy mechanism, threat model, accounting, residual-leakage evaluation or aggregation implementation exists. Strict zero-shot deployment remains unresolved: exporting a source scaler discloses source statistics, moving raw target data violates the intended boundary, and fitting target scalers is adaptation. A later approved protocol must resolve this without relabeling scaler statistics as model updates.","",
                  "## Preservation, artifacts and reproduction","",f"Independent audit verified{verification['frozen_files']} frozen artifacts, recomputed saved metrics from{verification['saved_prediction_files_recomputed']} prediction CSVs, reloaded{verification['final_nasa_models_reloaded']} final NASA models, and verified{verification['preservation']['verified_files']} original files byte-for-byte. The audit did not reread raw held-out data or execute another model forward evaluation.","",
                  "All new outputs are under outputs/phase021/: diagnostics.json; budget.json; screen/; confirm/; selection.json; azure/ and azure.json; final-models/ and final-models.json; freeze.json and freeze.sha256; evaluation-started.json; evaluation/ and evaluation.json; nasa-metrics.csv; verification.json; preservation.json and preservation-verified.json. Predictions, checkpoint histories, engine membership hashes, local private statistics, unsupported regimes, max/p95 errors and scientific/log overflow scores remain available.","",
                  "Use the original requirements-lock.txt for Phase02. The separate pinned requirements-phase021.lock contains torch==2.6.0+cpu and preserves every original dependency pin; the original .venv was unchanged. Commands from the repository root:","","```powershell",
                  "py -3.12 -m venv .venv-phase021",
                  ".\\.venv-phase021\\Scripts\\python.exe -m pip install -r requirements-phase021.lock",
                  ".\\.venv-phase021\\Scripts\\python.exe -W error scripts/phase021.py diagnostics",
                  ".\\.venv-phase021\\Scripts\\python.exe -W error scripts/phase021.py source",
                  ".\\.venv-phase021\\Scripts\\python.exe -W error scripts/phase021.py azure",
                  ".\\.venv-phase021\\Scripts\\python.exe -W error scripts/phase021.py fit",
                  ".\\.venv-phase021\\Scripts\\python.exe -W error scripts/phase021.py freeze",
                  ".\\.venv-phase021\\Scripts\\python.exe -W error scripts/phase021.py evaluate",
                  ".\\.venv-phase021\\Scripts\\python.exe -W error scripts/verify_phase021.py",
                  ".\\.venv-phase021\\Scripts\\python.exe scripts/report_phase021.py","```","",
                  "Training/evaluation stages deliberately reject reuse of completed run IDs and the final-evaluation marker. Reproduction requires a fresh checkout/output location with the preserved Phase01/02/alignment inputs; do not delete or overwrite this accepted run. Audit/report commands are safe to repeat. Test counts and exact validation commands are recorded in PHASE021_ACCEPTANCE.md.",""])
    target=ROOT/'docs/PHASE021_RESULTS.md'
    target.write_text('\n'.join(lines),encoding='utf-8')
    write_json(out/'report-provenance.json',{'freeze_sha256':digest(out/'freeze.json'),'report_sha256':digest(target),'generator_sha256':digest(Path(__file__))})
    print(target)


if __name__=='__main__':
    report()
