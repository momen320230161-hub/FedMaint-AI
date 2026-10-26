"""Numeric-only checkpoints; private transformations are separate from neural state."""
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from fedmaint.common import digest, write_json, FEATURES
from fedmaint.data.contracts import DataError
from fedmaint.data.clients import TabularNormalizer, AZURE_CLIENTS, NASA_CLIENTS
from fedmaint.data.normalization import LocalNormalizer
from fedmaint.data.batches import Batch
from .network import build_model, export_state, import_state, parameter_contract
from .training import predict


def validate_private(state, dataset, features, client):
    if (state.get('features')!=list(features) or state.get('domain')!=client
        or client not in (AZURE_CLIENTS if dataset=='azure' else NASA_CLIENTS)
        or not isinstance(state.get('fit_rows'),int) or state['fit_rows']<=0):
        raise DataError('Invalid private source/feature contract')
    def stats(value):
        if set(value)!= {'median','center','scale','informative'}:
            raise DataError('Invalid statistics fields')
        for key in value:
            a=np.asarray(value[key])
            if a.shape!=(len(features),) or (key=='informative' and a.dtype!=np.bool_) or (key!='informative' and not np.isfinite(a.astype(float)).all()):
                raise DataError('Invalid private statistics')
        if (np.asarray(value['scale'])<=0).any():
            raise DataError('Invalid normalization scale')
    stats(state['global'])
    if dataset=='nasa':
        if (state.get('kind') not in ('N0','N1','N2') or len(state.get('bin_resolution',[]))!=3
            or not np.isfinite(state['bin_resolution']).all() or min(state['bin_resolution'])<=0
            or not isinstance(state.get('min_bin_rows'),int) or state['min_bin_rows']<2):
            raise DataError('Invalid condition-bin configuration')
        for entry in state['bins'].values():
            stats(entry['stats'])
            if entry['rows']<state['min_bin_rows']:
                raise DataError('Unsupported bin stored as supported')
    elif state.get('kind')!='tabular':
        raise DataError('Invalid tabular state')


def save_bundle(directory, model, *, dataset, architecture, features, hidden, window,
                states, source, training, config):
    directory.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(directory/'weights.npz',**export_state(model))
    private={}
    for client,state in states.items():
        filename=f'private-{client}.json'
        write_json(directory/filename,state.state)
        private[client]={'path':filename,'sha256':digest(directory/filename)}
    metadata={'schema':3,'dataset':dataset,'architecture':architecture,'features':list(features),
              'hidden':hidden,'window':window,'source':source,'private':private,
              'weights_sha256':digest(directory/'weights.npz'),'training':training,
              'config':config.model_dump(mode='json'),'parameter_contract':parameter_contract(model),
              'postprocessing':'sigmoid' if dataset=='azure' else 'raw_linear_times_rul_scale'}
    write_json(directory/'model.json',metadata)


@dataclass
class LoadedModel:
    network: object
    metadata: dict
    states: dict

    def predict_raw(self, x, feature_names, mask=None, *, client=None):
        meta=self.metadata
        if list(feature_names)!=meta['features']:
            raise DataError('Feature order differs from saved contract')
        x=np.asarray(x,dtype=float)
        dataset=meta['dataset']
        source=meta['source'] if client is None else client
        if source not in self.states:
            raise DataError('No permitted frozen source transform')
        normal=self.states[source]
        if x.ndim not in (2,3) or x.shape[-1]!=len(feature_names) or len(x)==0:
            raise DataError('Invalid input dimension or empty batch')
        if dataset=='azure':
            if x.ndim!=2 or mask is not None:
                raise DataError('Azure requires tabular inputs')
            normalized,_=normal.transform(pd.DataFrame(x,columns=feature_names))
        else:
            if (x.ndim!=3 or x.shape[1]!=meta['window'] or mask is None or mask.dtype!=np.bool_
                or mask.shape!=x.shape[:2] or not mask[:,-1].all()
                or (mask[:,:-1]&~mask[:,1:]).any()):
                raise DataError('Invalid NASA mask or window')
            normalized=np.zeros_like(x)
            normalized[mask],_=normal.transform(pd.DataFrame(x[mask],columns=feature_names))
        batch=Batch(normalized.astype(np.float32),np.zeros(len(x)),np.arange(len(x)).astype(str),
                    np.repeat(source,len(x)),'inference',mask,np.zeros(len(x)) if mask is not None else None)
        batch.require('inference',tuple(self.states))
        return predict(self.network,batch,dataset,meta['config']['rul_scale'])


def load_bundle(directory, expected_dataset=None):
    directory=Path(directory).resolve()
    try:
        meta=json.loads((directory/'model.json').read_text(encoding='utf-8'))
        dataset=meta['dataset']
        if (meta['schema']!=3 or dataset not in ('azure','nasa')
            or expected_dataset not in (None,dataset)):
            raise DataError('Artifact task/schema mismatch')
        from .contracts import TrainingConfig
        config=TrainingConfig.model_validate(meta['config'])
        threshold=meta['training']['threshold']
        if (dataset=='azure' and (not isinstance(threshold,(int,float)) or not np.isfinite(threshold) or not 0<=threshold<=1)) or (dataset=='nasa' and threshold is not None):
            raise DataError('Invalid prediction threshold contract')
        features=meta['features']
        if (not features or len(set(features))!=len(features) or any(not f.startswith('feat_') for f in features)
            or (dataset=='nasa' and features!=list(FEATURES))
            or not isinstance(meta['hidden'],int) or not 1<=meta['hidden']<=4096
            or not isinstance(meta['window'],int) or not 1<=meta['window']<=4096):
            raise DataError('Invalid model dimensions/features')
        if digest(directory/'weights.npz')!=meta['weights_sha256']:
            raise DataError('Weight hash mismatch')
        model=build_model(dataset,meta['architecture'],len(features),meta['hidden'],meta['window'])
        with np.load(directory/'weights.npz',allow_pickle=False) as arrays:
            import_state(model,{k:arrays[k] for k in arrays.files})
        if parameter_contract(model)!=meta['parameter_contract']:
            raise DataError('Parameter contract mismatch')
        states={}
        for client,record in meta['private'].items():
            path=(directory/record['path']).resolve()
            if not path.is_relative_to(directory) or digest(path)!=record['sha256']:
                raise DataError('Private path/hash invalid')
            state=json.loads(path.read_text(encoding='utf-8'))
            validate_private(state,dataset,features,client)
            obj=TabularNormalizer() if dataset=='azure' else LocalNormalizer(state['kind'],state)
            obj.state=state
            states[client]=obj
        if meta['source'] not in states or set(states)!=set(meta['training']['membership']['train']['factories']):
            raise DataError('Source/participant mismatch')
        model.eval()
        return LoadedModel(model,meta,states)
    except (OSError,KeyError,TypeError,ValueError) as exc:
        raise DataError(f'Invalid V3 artifact: {exc}') from exc
