"""Factory-local preprocessing and fixed shared coordinates."""
import numpy as np
import pandas as pd
from fedmaint.common import FEATURES
from .batches import Batch
from .normalization import LocalNormalizer, _statistics
from .contracts import DataError
from .azure import SCHEMAS, validate_azure, make_azure_samples
from .nasa_clients import NasaStore

AZURE_CLIENTS=('azure-model1','azure-model2','azure-model3')
NASA_CLIENTS=('FD001','FD002','FD003')


class TabularNormalizer:
    def fit(self, frame, features, client, permitted):
        if (client not in permitted or client not in AZURE_CLIENTS or frame.empty
            or not frame.factory_id.eq(client).all() or not frame.split.eq('train').all()
            or not features or len(set(features))!=len(features) or any(not f.startswith('feat_') for f in features)):
            raise DataError('Only source-client training rows can fit normalization')
        x=frame[list(features)].to_numpy(float)
        if np.isinf(x).any():
            raise DataError('Infinite fitting data')
        self.state={'kind':'tabular','domain':client,'fit_rows':len(frame),'features':list(features),
                    'global':_statistics(x,'N0')}
        return self

    @property
    def coverage(self):
        return sum(self.state['global']['informative'])

    def transform(self, frame):
        stats=self.state['global']
        x=frame[self.state['features']].to_numpy(float)
        if np.isinf(x).any():
            raise DataError('Infinite input')
        z=(np.where(np.isnan(x),stats['median'],x)-stats['center'])/stats['scale']
        z[:,~np.asarray(stats['informative'],bool)]=0
        if not np.isfinite(z).all():
            raise DataError('Non-finite normalized input')
        return z, {'rows':len(frame),'unsupported_rows':0}


def azure_samples(root, config, held_out=False):
    # Filter source membership before feature construction/statistics. CSV chunks
    # may contain held-out rows; these are discarded without aggregation or fitting.
    machines=pd.read_csv(root/'data/Azure Data/PdM_machines.csv')
    machines=machines.loc[(machines.model=='model4') if held_out else (machines.model!='model4')]
    allowed=set(machines.machineID)
    tables={'machines':machines}
    for name in SCHEMAS:
        if name!='machines':
            tables[name]=pd.concat([c.loc[c.machineID.isin(allowed)] for c in
                pd.read_csv(root/f'data/Azure Data/PdM_{name}.csv',chunksize=50000)],ignore_index=True)
    return make_azure_samples(validate_azure(tables),config)


def azure_batch(frame, state, split):
    x,diagnostic=state.transform(frame)
    keys=(frame.factory_id+'/'+frame.machineID.astype(str)+'/'+frame.datetime.astype(str)).to_numpy()
    return Batch(x.astype(np.float32),frame.target.to_numpy(float),keys,frame.factory_id.to_numpy(),split),diagnostic


def nasa_batch(store, frame, normalizer, split, **kwargs):
    windows,diagnostic=store.windows(frame,normalizer,split,**kwargs)
    return Batch(windows.x,windows.y,windows.keys,windows.domain,split,windows.mask,windows.cycles),diagnostic


def source_clients(root, config):
    azure=azure_samples(root,config.pipeline.azure)
    features=tuple(sorted(c for c in azure if c.startswith('feat_')))
    states,batches={},{}
    for client in AZURE_CLIENTS:
        rows=azure.loc[azure.factory_id==client]
        state=TabularNormalizer().fit(rows.loc[rows.split=='train'],features,client,AZURE_CLIENTS)
        states[client]=state
        batches[client]={s:azure_batch(rows.loc[rows.split==s],state,s)[0] for s in ('train','validation')}
    # The old evaluation protocol fixes split seed42/fraction.2; config validates it.
    settings=config.training.model_dump(mode='json')|{'window':config.pipeline.nasa.window_cycles,
              'stride':config.pipeline.nasa.stride_cycles,'prefix_cycles':[50,100]}
    store=NasaStore(root,settings)
    for client in NASA_CLIENTS:
        train,validation=store.partition(client)
        state=LocalNormalizer(config.training.normalizer,settings).fit(train,domain=client,permitted=NASA_CLIENTS)
        states[client]=state
        batches[client]={'train':nasa_batch(store,train,state,'train')[0],
                         'validation':nasa_batch(store,validation,state,'validation',prefixes=True)[0]}
    return states,batches,features,store
