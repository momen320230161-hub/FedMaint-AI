import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from fedmaint.common import FEATURES,write_json,digest
from fedmaint.data.batches import Batch,pool
from fedmaint.data.clients import TabularNormalizer,source_clients
from fedmaint.data.normalization import LocalNormalizer
from fedmaint.data.nasa_clients import NasaStore
from fedmaint.data.contracts import DataError
from fedmaint.models.contracts import TrainingConfig,ExperimentConfig
from fedmaint.models.network import RULModel,AzureMLP,parameter_contract,export_state,import_state
from fedmaint.models.training import fit,initialize,predict
from fedmaint.models.artifacts import save_bundle,load_bundle
from fedmaint.models.experiments import seal,verify_freeze
from fedmaint.api.artifacts import load_model_artifact
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def rows():
    frame=pd.DataFrame(np.random.default_rng(2).normal(size=(256,24)),columns=FEATURES)
    frame['feat_setting_1']=np.repeat([0.,10.],128)
    frame['feat_setting_2']=0.
    frame['feat_setting_3']=100.
    frame['feat_sensor_21']=3.
    frame['domain']='FD001'
    frame['engine_id']=np.repeat([1,2],128)
    return frame

@pytest.mark.parametrize('kind',['N0','N1','N2'])
def test_normalization_private_fit_roundtrip_and_constant_neutrality(kind,rows,tmp_path):
    cfg=TrainingConfig().model_dump()
    normal=LocalNormalizer(kind,cfg).fit(rows,domain='FD001',permitted=('FD001',))
    before=copy.deepcopy(normal.state)
    shifted=rows.iloc[:4].copy()
    shifted['feat_sensor_21']=1e12
    shifted['feat_sensor_1']=np.nan
    actual,_=normal.transform(shifted)
    assert actual.shape==(4,24) and np.isfinite(actual).all()
    assert (actual[:,FEATURES.index('feat_sensor_21')]==0).all()
    assert normal.state==before
    path=tmp_path/'state.json';normal.save(path)
    np.testing.assert_array_equal(actual,LocalNormalizer.load(path).transform(shifted)[0])
    for domain,split in [('FD004','train'),('FD001','validation'),('FD001','test')]:
        with pytest.raises(DataError):
            LocalNormalizer(kind,cfg).fit(rows,domain=domain,permitted=('FD001','FD004'),split=split)


def test_unknown_condition_deterministic_global_fallback(rows):
    cfg=TrainingConfig().model_dump()
    n1=LocalNormalizer('N1',cfg).fit(rows,domain='FD001',permitted=('FD001',))
    n2=LocalNormalizer('N2',cfg).fit(rows,domain='FD001',permitted=('FD001',))
    assert len(n2.state['bins'])==2
    unknown=rows.iloc[:10].copy();unknown['feat_setting_1']=42.
    before=copy.deepcopy(n2.state)
    actual,diag=n2.transform(unknown)
    np.testing.assert_array_equal(actual,n1.transform(unknown)[0])
    assert diag['unsupported_rows']==10 and n2.state==before

@pytest.mark.parametrize('architecture,count',[('gru',5601),('mlp',24065)])
def test_masks_individual_sequences_state_and_fl_dimensions(architecture,count):
    initialize(42)
    a=RULModel(architecture)
    initialize(42)
    b=RULModel(architecture)
    assert parameter_contract(a)==parameter_contract(b)
    assert parameter_contract(a)['parameters']==count
    assert parameter_contract(a)['bytes']==4*count
    x=torch.randn(3,30,24)
    mask=torch.ones(3,30,dtype=torch.bool);mask[0,:7]=False;mask[1,:12]=False
    expected=a(x,mask).detach()
    x[~mask]=float('nan')
    torch.testing.assert_close(a(x,mask),expected)
    if architecture=='gru':
        individual=torch.cat([a(row[valid].unsqueeze(0),torch.ones(1,int(valid.sum()),dtype=torch.bool)) for row,valid in zip(x,mask)])
        torch.testing.assert_close(individual,expected,rtol=1e-5,atol=1e-6)
    state=export_state(a);import_state(b,state)
    torch.testing.assert_close(b(x,mask),expected,rtol=0,atol=0)
    state[next(iter(state))].fill(999)
    assert not np.array_equal(export_state(a)[next(iter(state))],state[next(iter(state))])
    with pytest.raises(DataError):
        import_state(b,{'private_medians':np.zeros(24,np.float32)})
    mask[0,-1]=False
    with pytest.raises(DataError):a(x,mask)


def batch(dataset,split):
    rng=np.random.default_rng(42 if split=='train' else 43)
    n=16
    if dataset=='azure':
        return Batch(rng.normal(size=(n,16)).astype(np.float32),np.tile([0.,1.],8),
                     np.array([f'azure-model1/{split}/{i}' for i in range(n)]),np.repeat('azure-model1',n),split)
    return Batch(rng.normal(size=(n,30,24)).astype(np.float32),np.arange(n,dtype=float)+50,
                 np.array([f'FD001/train/{i+(100 if split=="validation" else 0)}/50' for i in range(n)]),
                 np.repeat('FD001',n),split,np.ones((n,30),bool),np.tile([50,100],8))

@pytest.mark.parametrize('dataset,architecture',[('azure','mlp'),('nasa','gru'),('nasa','mlp')])
def test_reproducible_fit_selection_and_contamination_guards(dataset,architecture):
    cfg=TrainingConfig(epochs=2,patience=2)
    train,val=batch(dataset,'train'),batch(dataset,'validation')
    permitted=tuple(set(train.factories))
    a,meta=fit(train,val,dataset,architecture,cfg,42,permitted)
    b,other=fit(train,val,dataset,architecture,cfg,42,permitted)
    for key,value in export_state(a).items():np.testing.assert_array_equal(value,export_state(b)[key])
    assert meta['history']==other['history'] and meta['sample_count']==16
    scores=[h['validation_score'] for h in meta['history']]
    assert meta['selected_epoch']==int(np.argmax(scores))+1
    assert np.isfinite(predict(a,val,dataset)).all()
    val.split='test'
    with pytest.raises(DataError):fit(train,val,dataset,architecture,cfg,42,permitted)
    val.split='validation';val.keys=train.keys.copy()
    with pytest.raises(DataError):fit(train,val,dataset,architecture,cfg,42,permitted)
    with pytest.raises(DataError):pool([train,val, batch(dataset,'test')])

@pytest.mark.parametrize('dataset,architecture',[('azure','mlp'),('nasa','gru')])
def test_internal_raw_inference_and_artifact_tampering(dataset,architecture,rows,tmp_path):
    cfg=TrainingConfig(epochs=2)
    train,val=batch(dataset,'train'),batch(dataset,'validation')
    client=str(train.factories[0])
    model,meta=fit(train,val,dataset,architecture,cfg,42,(client,))
    if dataset=='azure':
        features=[f'feat_{i}' for i in range(16)]
        frame=pd.DataFrame(train.x,columns=features).assign(factory_id=client,split='train')
        state=TabularNormalizer().fit(frame,features,client,(client,))
        raw=train.x
        encoded=state.transform(frame)[0]
    else:
        features=list(FEATURES)
        state=LocalNormalizer('N2',cfg.model_dump()).fit(rows,domain=client,permitted=(client,))
        raw=train.x.astype(float)
        encoded=state.transform(pd.DataFrame(raw.reshape(-1,24),columns=features))[0].reshape(raw.shape)
    directory=tmp_path/'model'
    save_bundle(directory,model,dataset=dataset,architecture=architecture,features=features,hidden=32,
                window=1 if dataset=='azure' else 30,states={client:state},source=client,training=meta,config=cfg)
    loaded=load_model_artifact(directory,expected_dataset=dataset)
    train.x=encoded.astype(np.float32)
    np.testing.assert_array_equal(loaded.predict_raw(raw,features,train.mask),predict(model,train,dataset))
    with pytest.raises(DataError):loaded.predict_raw(raw,list(reversed(features)),train.mask)
    with pytest.raises(DataError):load_bundle(directory,expected_dataset='nasa' if dataset=='azure' else 'azure')
    record=json.loads((directory/'model.json').read_text())
    record['private'][client]['path']='../outside.json'
    write_json(directory/'model.json',record)
    with pytest.raises(DataError):load_bundle(directory)
    record['private'][client]['path']=f'private-{client}.json'
    write_json(directory/'model.json',record)
    weights=export_state(model);weights[next(iter(weights))]=np.zeros(1,np.float32)
    np.savez_compressed(directory/'weights.npz',**weights)
    with pytest.raises(DataError):load_bundle(directory)


def test_freeze_rejects_mutated_source_and_model(tmp_path):
    root=tmp_path/'root';out=root/'outputs/run'
    (root/'src').mkdir(parents=True);out.mkdir(parents=True)
    write_json(root/'pyproject.toml',{})
    write_json(root/'data/manifest.json',{})
    write_json(out/'weights.json',{'value':1})
    cfg=ExperimentConfig.model_validate_json((ROOT/'configs/v3.json').read_text())
    seal(root,out,cfg);verify_freeze(root,out)
    write_json(out/'weights.json',{'value':2})
    with pytest.raises(DataError):verify_freeze(root,out)


def test_loader_forbids_holdout_and_official_test_before_freeze(monkeypatch):
    store=NasaStore(ROOT,{'window':30})
    def forbidden(*args,**kwargs):raise AssertionError('Premature read')
    monkeypatch.setattr(pd,'read_csv',forbidden)
    for domain,source in [('FD004','train'),('FD004','test'),('FD001','test')]:
        with pytest.raises(DataError):store.read(domain,source)


def test_real_source_contract_engine_splits_and_local_fit():
    cfg=ExperimentConfig.model_validate_json((ROOT/'configs/v3.json').read_text())
    states,batches,features,store=source_clients(ROOT,cfg)
    assert len(features)==16 and len(states)==6
    assert len({batches[c]['train'].x.shape[-1] for c in ('FD001','FD002','FD003')})==1
    assert not any(r['domain']=='FD004' or r['source']=='test' for r in store.access)
    for client,splits in batches.items():
        train,val=splits['train'],splits['validation']
        train.require('train',(client,));val.require('validation',(client,))
        assert set(train.keys).isdisjoint(val.keys)
        if client.startswith('FD'):
            fit_frame,_=store.partition(client)
            assert states[client].state['fit_rows']==len(fit_frame)
            assert set(k.rsplit('/',1)[0] for k in train.keys).isdisjoint(k.rsplit('/',1)[0] for k in val.keys)


def test_inference_enforces_recorded_thread_policy_after_process_default_changes():
    initialize(42)
    model=AzureMLP()
    data=batch('azure','validation')
    expected=predict(model,data,'azure')
    torch.set_num_threads(2)
    actual=predict(model,data,'azure')
    assert torch.get_num_threads()==1
    np.testing.assert_array_equal(actual,expected)


def test_nasa_engine_leakage_rejected_even_at_different_cycles():
    train,val=batch('nasa','train'),batch('nasa','validation')
    val.keys=np.array([key.rsplit('/',1)[0]+'/100' for key in train.keys])
    with pytest.raises(DataError,match='Engine leakage'):
        fit(train,val,'nasa','gru',TrainingConfig(epochs=1),42,('FD001',))


def test_future_nasa_observations_cannot_change_earlier_windows(nasa_data):
    from fedmaint.data.nasa import sequences,rul_targets
    from fedmaint.data.contracts import NasaConfig
    frame=rul_targets(nasa_data,'train')
    cfg=NasaConfig(window_cycles=3,stride_cycles=1)
    original=sequences(frame,cfg)
    frame.loc[frame.cycle>5,'feat_sensor_1']=1e9
    changed=sequences(frame,cfg)
    np.testing.assert_array_equal(original.x[original.cycles<=5],changed.x[changed.cycles<=5])


def test_mixed_split_normalization_rows_rejected(rows):
    rows['split']='train';rows.loc[0,'split']='validation'
    with pytest.raises(DataError):
        LocalNormalizer('N2',TrainingConfig().model_dump()).fit(rows,domain='FD001',permitted=('FD001',))
