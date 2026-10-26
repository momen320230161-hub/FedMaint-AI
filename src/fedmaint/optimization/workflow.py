"""Source selection -> final fit -> immutable freeze -> exactly-once held-out scoring."""

import json
from importlib.metadata import version
import time
import xml.etree.ElementTree as ET

from .common import Budget, DOMAINS, digest, write_json
from .nasa import NasaStore, choose_source, combine
from .normalization import LocalNormalizer
from .selection import score_predictions
from .torch_models import load_weights, predict
from .training import fit


def final_fit(root, output, config):
    if (output / "final-models.json").exists():
        raise ValueError("Final fits already recorded")
    selection = json.loads((output/"selection.json").read_text())
    if selection["status"] != "completed":
        raise ValueError("Source selection is incomplete")
    store, budget, cache, runs = NasaStore(root,config), Budget(output,config), {}, []
    # Panel identity is determined solely by source screening and recorded confirmation.
    for candidate in selection["panel"]:
        kind, architecture = candidate.split("-")
        for members in [*( (d,) for d in DOMAINS), DOMAINS]:
            group = members[0] if len(members)==1 else "pooled"
            key = kind,group
            if key not in cache:
                states,train,val = {},[],[]
                for domain in members:
                    fit_rows,val_rows = store.partition(domain)
                    states[domain]=LocalNormalizer(kind,config).fit(fit_rows,domain=domain,permitted=members)
                    train.append(store.windows(fit_rows,states[domain],"train")[0])
                    val.append(store.windows(val_rows,states[domain],"validation",prefixes=True)[0])
                cache[key]=states,combine(train),combine(val)
            states,train,val=cache[key]
            source=choose_source(states)
            for seed in config["seeds"]:
                directory=output/"final-models"/f"{candidate}-{group}-seed{seed}"
                model,meta=fit(train,val,architecture,seed,config,directory,budget)
                for domain,state in states.items():
                    state.save(directory/f"private-{domain}.json")
                run={"candidate":candidate,"architecture":architecture,"normalizer":kind,"scope":group,"seed":seed,
                     "members":list(members),"inference_source":source,"training":meta,"status":"trained",
                     "directory":str(directory.relative_to(output))}
                write_json(directory/"model.json",run)
                runs.append(run)
                write_json(output/"final-fit-progress.json",{"runs":runs,"access":store.access})
                print(f"FINAL FIT {candidate}/{group}/{seed}",flush=True)
    result={"status":"trained","runs":runs,"access":store.access}
    write_json(output/"final-models.json",result)
    return result


def seal(root,output,config):
    if (output/"freeze.json").exists():
        raise ValueError("Freeze already exists")
    tests=ET.parse(output/"tests-final.xml").getroot().find("testsuite")
    if tests is None or int(tests.attrib["tests"]) < 93 or any(int(tests.attrib[k]) for k in ("errors","failures","skipped")):
        raise ValueError("Required final tests have not passed")
    if not (output/"protocol-deviations.json").is_file():
        raise ValueError("The exposure/deviation record must be explicit before freeze")
    for name in ("selection.json","azure.json"):
        if json.loads((output/name).read_text())["status"] != "completed":
            raise ValueError(f"Cannot freeze an incomplete stage: {name}")
    final=json.loads((output/"final-models.json").read_text())
    selection=json.loads((output/"selection.json").read_text())
    if len(final["runs"]) != len(selection["panel"])*8:
        raise ValueError("Final predeclared model panel incomplete")
    directories={"screen","confirm","final-models","azure"}
    records={"diagnostics.json","preservation.json","selection.json","azure.json","budget.json","final-models.json",
             "protocol-deviations.json","tests-final.xml"}
    files={str(p.relative_to(output)):digest(p) for p in output.rglob("*")
           if p.is_file() and p.suffix in (".json",".npz",".csv",".xml")
           and (p.relative_to(output).parts[0] in directories or str(p.relative_to(output)) in records)}
    source_files={str(p.relative_to(root)):digest(p) for p in sorted((root/"src/fedmaint/optimization").glob("*.py"))}
    for relative in ("configs/phase021.json","requirements-phase021.lock","scripts/phase021.py",
                     "scripts/verify_phase021.py","scripts/report_phase021.py","tests/test_optimization.py"):
        source_files[relative]=digest(root/relative)
    freeze={"schema":1,"created_unix":time.time(),"winner":selection["winner"],"panel":selection["panel"],
            "files":files,"source_files":source_files,"config":config,
            "postprocessing":"raw_primary_zero_floor_diagnostic","fd004_previously_observed":True,
            "selection_exploratory":True,"environment":{p:version(p) for p in ("torch","numpy","pandas","scikit-learn")}}
    write_json(output/"freeze.json",freeze)
    # Immutable digest sidecar prevents silently replacing the freeze itself.
    (output/"freeze.sha256").write_text(digest(output/"freeze.json")+"\n",encoding="ascii")
    return freeze


def verify_freeze(root,output):
    if digest(output/"freeze.json") != (output/"freeze.sha256").read_text().strip():
        raise ValueError("Freeze manifest changed")
    freeze=json.loads((output/"freeze.json").read_text())
    for relative,h in freeze["files"].items():
        if digest(output/relative)!=h:
            raise ValueError(f"Frozen artifact changed: {relative}")
    for relative,h in freeze["source_files"].items():
        if digest(root/relative)!=h:
            raise ValueError(f"Frozen implementation/config changed: {relative}")
    return freeze


def evaluate(root,output,config):
    freeze=verify_freeze(root,output)
    # Exclusive marker: no automatic second held-out evaluation, even after interruption.
    with (output/"evaluation-started.json").open("x",encoding="utf-8") as stream:
        json.dump({"started_unix":time.time(),"freeze_sha256":digest(output/"freeze.json")},stream)
    store=NasaStore(root,config,gate=lambda:verify_freeze(root,output))
    models=json.loads((output/"final-models.json").read_text())
    evaluations=[]
    cache={}
    for run in models["runs"]:
        directory=output/run["directory"]
        model=load_weights(directory/"weights.npz",run["architecture"],config)
        states={d:LocalNormalizer.load(directory/f"private-{d}.json") for d in run["members"]}
        pools=[(domain,"official_test","test",domain,True) for domain in run["members"]]
        pools += [("FD004","held_out_official_test","test",run["inference_source"],True),
                  ("FD004","held_out_run_to_failure","train",run["inference_source"],False)]
        for domain,label,source,state_domain,terminal in pools:
            key=run["normalizer"],state_domain,domain,label
            if key not in cache:
                cache[key]=store.windows(store.read(domain,source),states[state_domain],"evaluation",terminal=terminal)
            batch,diagnostics=cache[key]
            destination=output/"evaluation"/directory.name
            destination.mkdir(parents=True,exist_ok=True)
            result=score_predictions(destination,f"{domain}-{label}",batch,predict(model,batch))
            evaluations.append({"candidate":run["candidate"],"scope":run["scope"],"seed":run["seed"],"domain":domain,
                                "pool":label,"inference_source":state_domain,"diagnostic":diagnostics,
                                "directory":str(destination.relative_to(output)),**result})
        write_json(output/"evaluation-progress.json",{"runs":evaluations,"access":store.access})
        print(f"EVALUATED ONCE {directory.name}",flush=True)
    report={"status":"completed","freeze_sha256":digest(output/"freeze.json"),"fd004_not_blind":True,
            "primary_output":"raw","floor_is_diagnostic_only":True,"runs":evaluations,"access":store.access}
    write_json(output/"evaluation.json",report)
    return report
