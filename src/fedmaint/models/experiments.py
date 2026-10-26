"""Frozen local/pooled references, then retrospective final evaluation."""
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
from fedmaint.common import digest,write_json,FEATURES
from fedmaint.data.integrity import verify_manifest
from fedmaint.data.prepare import safe_output
from fedmaint.data.contracts import DataError
from fedmaint.data.batches import pool
from fedmaint.data.clients import (source_clients,azure_samples,azure_batch,nasa_batch,AZURE_CLIENTS,NASA_CLIENTS)
from .contracts import ExperimentConfig
from .training import fit,predict
from .artifacts import save_bundle,load_bundle
from .metrics import classification_metrics,regression_metrics


def source_hashes(root):
    files=[p for folder in ('src','configs','scripts') for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json')]
    files += [root/'pyproject.toml',root/'data/manifest.json']
    if (root/'requirements-v3.lock').exists():
        files.append(root/'requirements-v3.lock')
    return {str(p.relative_to(root)):digest(p) for p in sorted(files)}


def seal(root, output, config):
    record={'source':source_hashes(root),'config':config.model_dump(mode='json'),
            'files':{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()},
            'exposure':'retrospective; FD004 and model4 were previously observed',
            'selection':'fixed candidate panel; validation checkpoints/thresholds only'}
    write_json(output/'freeze.json',record)
    (output/'freeze.sha256').write_text(digest(output/'freeze.json'),encoding='ascii')


def verify_freeze(root,output):
    if digest(output/'freeze.json')!=(output/'freeze.sha256').read_text():
        raise DataError('Freeze changed')
    record=json.loads((output/'freeze.json').read_text())
    if source_hashes(root)!=record['source']:
        raise DataError('Source/config changed after freeze')
    for filename,expected in record['files'].items():
        path=(output/filename).resolve()
        if not path.is_relative_to(output.resolve()) or digest(path)!=expected:
            raise DataError('Frozen artifact changed')
    return record


class Identity:
    def transform(self,frame):
        return frame[list(FEATURES)].to_numpy(float),{}


def metrics(dataset,y,prediction,threshold):
    if dataset=='azure':
        return classification_metrics(y,prediction,threshold)
    result=regression_metrics(y,prediction)
    result.update(negative_fraction=float(np.mean(prediction<0)),
                  abs_error_p95=float(np.quantile(np.abs(prediction-y),.95)),
                  abs_error_max=float(np.max(np.abs(prediction-y))))
    return result


def run(root,config,output):
    root,output=root.resolve(),safe_output(root,output)
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    records=verify_manifest(root)
    plan={'status':'running','config':config.model_dump(mode='json'),
          'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
          'environment':{'python':platform.python_version(),**{p:version(p) for p in
              ('torch','numpy','pandas','scikit-learn','fastapi','pydantic')}},
          'input_files':records,'runs':[]}
    write_json(output/'plan.json',plan)
    try:
        states,batches,azure_features,store=source_clients(root,config)
        for dataset,clients,architectures,features in (
            ('azure',AZURE_CLIENTS,('mlp',),azure_features),
            ('nasa',NASA_CLIENTS,config.training.nasa_architectures,FEATURES)):
            for architecture in architectures:
                for seed in config.training.seeds:
                    for scope in (*clients,'pooled'):
                        members=clients if scope=='pooled' else (scope,)
                        train=pool([batches[c]['train'] for c in members])
                        validation=pool([batches[c]['validation'] for c in members])
                        source=min(members,key=lambda c:(-states[c].coverage,c))
                        model,training=fit(train,validation,dataset,architecture,config.training,seed,members,
                                           config.pipeline.nasa.window_cycles)
                        run_id=f'{dataset}-{architecture}-{scope}-seed{seed}'
                        save_bundle(output/run_id,model,dataset=dataset,architecture=architecture,features=features,
                                    hidden=config.training.hidden,window=1 if dataset=='azure' else config.pipeline.nasa.window_cycles,
                                    states={c:states[c] for c in members},source=source,training=training,config=config.training)
                        loaded=load_bundle(output/run_id,expected_dataset=dataset)
                        np.testing.assert_array_equal(predict(model,validation,dataset,config.training.rul_scale),
                                                      predict(loaded.network,validation,dataset,config.training.rul_scale))
                        plan['runs'].append({'id':run_id,'dataset':dataset,'scope':scope,'architecture':architecture,
                                             'seed':seed,'members':list(members),'source':source,'training':training,
                                             'parameters':loaded.metadata['parameter_contract']})
                        write_json(output/'plan.json',plan)
                        print('FIT',run_id,'epoch',training['selected_epoch'],flush=True)
        plan['status']='selected_and_frozen_before_evaluation'
        write_json(output/'plan.json',plan)
        seal(root,output,config)
        verify_freeze(root,output)
        with (output/'evaluation-started.json').open('x',encoding='utf-8') as marker:
            json.dump({'retrospective':True,'freeze_sha256':digest(output/'freeze.json')},marker)
        # No target fit, threshold change or retraining beyond this boundary.
        store.gate=lambda:verify_freeze(root,output)
        evaluations=[]
        source_azure=azure_samples(root,config.pipeline.azure)
        target_azure=azure_samples(root,config.pipeline.azure,held_out=True)
        for entry in plan['runs']:
            loaded=load_bundle(output/entry['id'])
            dataset=entry['dataset']
            pools=[]
            if dataset=='azure':
                for client in entry['members']:
                    rows=source_azure.loc[(source_azure.factory_id==client)&(source_azure.split=='test')]
                    pools.append((client,'test',rows,client,False))
                pools.append(('azure-model4','held_out',target_azure,entry['source'],False))
            else:
                for client in entry['members']:
                    pools.append((client,'official_test',store.read(client,'test'),client,True))
                pools.append(('FD004','held_out_official_test',store.read('FD004','test'),entry['source'],True))
                pools.append(('FD004','held_out_run_to_failure',store.read('FD004','train'),entry['source'],False))
            for client,split,frame,source,terminal in pools:
                if dataset=='azure':
                    batch,diagnostic=azure_batch(frame,loaded.states[source],split)
                    raw_x=frame[loaded.metadata['features']].to_numpy(float)
                    mask=None
                else:
                    batch,diagnostic=nasa_batch(store,frame,loaded.states[source],split,terminal=terminal)
                    raw,_=nasa_batch(store,frame,Identity(),split,terminal=terminal,dtype=np.float64)
                    raw_x,mask=raw.x,raw.mask
                prediction=predict(loaded.network,batch,dataset,config.training.rul_scale)
                restored=loaded.predict_raw(raw_x,loaded.metadata['features'],mask,client=source)
                np.testing.assert_allclose(prediction,restored,rtol=0,atol=0)
                filename=f'{entry["id"]}/{client}-{split}.npz'
                arrays={'x':raw_x,'y':batch.y,'prediction':prediction,'keys':batch.keys}
                if mask is not None:
                    arrays['mask']=mask
                np.savez_compressed(output/filename,**arrays)
                result={'run':entry['id'],'dataset':dataset,'client':client,'split':split,'source':source,
                        'seed':entry['seed'],'architecture':entry['architecture'],'scope':entry['scope'],
                        'metrics':metrics(dataset,batch.y,prediction,loaded.metadata['training']['threshold']),
                        'diagnostic':diagnostic,'predictions':filename,'sha256':digest(output/filename)}
                evaluations.append(result)
            print('EVALUATE',entry['id'],flush=True)
        verify_freeze(root,output)
        if verify_manifest(root)!=records:
            raise DataError('Input identity changed')
        write_json(output/'results.json',{'status':'completed','retrospective':True,'evaluations':evaluations,
                                          'seconds':time.perf_counter()-started})
        write_json(output/'completion.json',{'fits':len(plan['runs']),'evaluations':len(evaluations),'status':'completed'})
        return evaluations
    except Exception as exc:
        write_json(output/'failure.json',{'error':f'{type(exc).__name__}: {exc}'})
        raise


def verify_experiment(root,output):
    verify_freeze(root,output)
    results=json.loads((output/'results.json').read_text())
    count=0
    for row in results['evaluations']:
        path=(output/row['predictions']).resolve()
        if not path.is_relative_to(output.resolve()) or digest(path)!=row['sha256']:
            raise DataError('Prediction artifact changed')
        model=load_bundle(output/row['run'],expected_dataset=row['dataset'])
        with np.load(path,allow_pickle=False) as arrays:
            prediction=model.predict_raw(arrays['x'],model.metadata['features'],arrays['mask'] if 'mask' in arrays else None,client=row['source'])
            np.testing.assert_array_equal(prediction,arrays['prediction'])
            actual=metrics(row['dataset'],arrays['y'],prediction,model.metadata['training']['threshold'])
            if actual!=row['metrics']:
                raise DataError('Saved metrics differ from recomputation')
        count+=1
    record={'status':'passed','prediction_files':count,'models':len({r['run'] for r in results['evaluations']})}
    write_json(output/'verification.json',record)
    return record
