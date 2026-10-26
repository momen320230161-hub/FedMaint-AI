import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fedmaint.data.contracts import DataError
from fedmaint.data.preprocessing import LocalPreprocessor
from fedmaint.optimization.common import FEATURES, Budget, digest, load_config, write_json
from fedmaint.optimization.nasa import NasaStore, Windows
from fedmaint.optimization.normalization import LocalNormalizer
from fedmaint.optimization.selection import supports
from fedmaint.optimization.torch_models import RULModel, parameter_count, load_weights
from fedmaint.optimization.training import fit
from fedmaint.optimization.workflow import verify_freeze

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(ROOT)


@pytest.fixture
def training_rows():
    rng=np.random.default_rng(42)
    frame=pd.DataFrame(rng.normal(size=(256,24)),columns=FEATURES)
    frame["feat_setting_1"]=np.repeat([0.,10.],128)
    frame["feat_setting_2"]=0.
    frame["feat_setting_3"]=100.
    frame["feat_sensor_21"]=3.
    frame["domain"]="FD001"
    frame["engine_id"]=np.repeat([1,2],128)
    return frame


@pytest.mark.parametrize("kind",["N0","N1","N2"])
def test_local_fit_guards_constant_neutrality_frozen_roundtrip(kind,cfg,training_rows,tmp_path):
    normal=LocalNormalizer(kind,cfg).fit(training_rows,domain="FD001",permitted=("FD001",))
    original=copy.deepcopy(normal.state)
    altered=training_rows.iloc[:4].copy()
    altered["feat_sensor_21"]=1e12
    x,_=normal.transform(altered)
    assert (x[:,FEATURES.index("feat_sensor_21")]==0).all()
    assert normal.state==original
    path=tmp_path/"state.json"
    normal.save(path)
    np.testing.assert_array_equal(x,LocalNormalizer.load(path).transform(altered)[0])
    with pytest.raises(DataError):
        LocalNormalizer(kind,cfg).fit(training_rows,domain="FD004",permitted=("FD004",))
    with pytest.raises(DataError):
        LocalNormalizer(kind,cfg).fit(training_rows,domain="FD001",permitted=("FD001",),split="validation")


def test_n0_matches_original_local_scaler(cfg,training_rows):
    normal=LocalNormalizer("N0",cfg).fit(training_rows,domain="FD001",permitted=("FD001",))
    old=LocalPreprocessor().fit(training_rows.assign(factory_id="FD001",split="train"),list(FEATURES),
                                factory_id="FD001",training_factories={"FD001"})
    x,_=normal.transform(training_rows)
    indices=[FEATURES.index(c) for c in old.columns]
    np.testing.assert_allclose(x[:,indices],old.transform(training_rows),rtol=1e-12,atol=1e-12)


def test_n2_unsupported_bins_use_source_global_n1(cfg,training_rows):
    n1=LocalNormalizer("N1",cfg).fit(training_rows,domain="FD001",permitted=("FD001",))
    n2=LocalNormalizer("N2",cfg).fit(training_rows,domain="FD001",permitted=("FD001",))
    assert len(n2.state["bins"])==2
    target=training_rows.iloc[:3].copy()
    target["feat_setting_1"]=42.
    expected,_=n1.transform(target)
    actual,diagnostic=n2.transform(target)
    np.testing.assert_array_equal(actual,expected)
    assert diagnostic["unsupported_rows"]==3


@pytest.mark.parametrize("architecture,count",[("mlp",24065),("gru",5601)])
def test_parameter_shapes_padding_and_roundtrip(architecture,count,cfg,tmp_path):
    from fedmaint.optimization.torch_models import save_weights
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model=RULModel(architecture)
    assert parameter_count(model)==count
    x=torch.randn(3,30,24)
    mask=torch.ones(3,30,dtype=torch.bool)
    mask[0,:7]=False;mask[1,:12]=False
    expected=model(x,mask).detach()
    x[~mask]=float("nan")
    torch.testing.assert_close(model(x,mask),expected)
    if architecture=="gru":
        individual=torch.cat([model(row[valid].unsqueeze(0),torch.ones(1,int(valid.sum()),dtype=torch.bool)) for row,valid in zip(x,mask)])
        torch.testing.assert_close(individual,expected,rtol=1e-5,atol=1e-6)
    path=tmp_path/"weights.npz"
    save_weights(path,model)
    loaded=load_weights(path,architecture,cfg)
    torch.testing.assert_close(loaded(x,mask),expected,rtol=0,atol=0)
    assert all(not any(word in k for word in ("median","scale","bin")) for k in model.state_dict())


def test_torch_mlp_port_matches_frozen_sklearn_forward():
    from fedmaint.models.artifacts import load_bundle
    source=ROOT/"outputs/phase02/nasa-mlp-pooled-seed42"
    old=load_bundle(source)
    model=RULModel("mlp").double()
    with torch.no_grad():
        model.hidden.weight.copy_(torch.from_numpy(old.network.weights[0].T))
        model.hidden.bias.copy_(torch.from_numpy(old.network.biases[0]))
        model.head.weight.copy_(torch.from_numpy(old.network.weights[1].T))
        model.head.bias.copy_(torch.from_numpy(old.network.biases[1]))
    x=torch.randn(5,30,24,dtype=torch.float64)
    mask=torch.ones(5,30,dtype=torch.bool)
    encoded=torch.cat([x.flatten(1),mask.double()],1).numpy()
    new=np.maximum(model(x,mask).detach().numpy()*100,0)
    np.testing.assert_allclose(new,old.network.predict(encoded),rtol=1e-12,atol=1e-12)


def test_pre_freeze_loader_blocks_tests_and_fd004_before_read(cfg,monkeypatch):
    store=NasaStore(ROOT,cfg)
    def forbidden(*args,**kwargs):
        raise AssertionError("Data must not be read")
    monkeypatch.setattr(pd,"read_csv",forbidden)
    for domain,source in [("FD004","train"),("FD004","test"),("FD001","test")]:
        with pytest.raises(DataError,match="freeze"):
            store.read(domain,source)


def test_freeze_rejects_changed_weights_or_source(tmp_path):
    root=tmp_path/"root";out=root/"output";out.mkdir(parents=True)
    write_json(out/"model.json",{"fixed":1})
    write_json(root/"config.json",{"value":1})
    freeze={"files":{"model.json":digest(out/"model.json")},"source_files":{"config.json":digest(root/"config.json")}}
    write_json(out/"freeze.json",freeze)
    (out/"freeze.sha256").write_text(digest(out/"freeze.json"))
    verify_freeze(root,out)
    write_json(out/"model.json",{"fixed":2})
    with pytest.raises(ValueError,match="changed"):
        verify_freeze(root,out)


def test_promotion_requires_macro_worst_and_two_domains():
    base={"status":"completed","macro_rmse":10.,"worst_domain_rmse":12.,"fold_rmse":{"FD001":8.,"FD002":10.,"FD003":12.}}
    better={"status":"completed","macro_rmse":8.,"worst_domain_rmse":10.,"fold_rmse":{"FD001":6.,"FD002":8.,"FD003":10.}}
    assert supports(better,base)
    worse=copy.deepcopy(better);worse["worst_domain_rmse"]=13
    assert not supports(worse,base)


def test_small_cpu_fit_reproducibility_and_budget(cfg,tmp_path):
    cfg=copy.deepcopy(cfg);cfg.update(epochs=2,patience=2)
    rng=np.random.default_rng(42)
    def batch(split):
        return Windows(rng.normal(size=(8,30,24)).astype(np.float32),np.ones((8,30),bool),
                       np.arange(8,dtype=float)+100,np.array([f"{split}/{i}" for i in range(8)]),
                       np.tile([50,100],4),np.repeat("FD001",8),split)
    train,val=batch("train"),batch("validation")
    budget=Budget(tmp_path,cfg)
    a,_=fit(train,val,"gru",42,cfg,tmp_path/"a",budget)
    b,_=fit(train,val,"gru",42,cfg,tmp_path/"b",budget)
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa,pb,atol=0,rtol=0)
    assert len(budget.state["fits"])==2
    with pytest.raises(DataError,match="repeated"):
        fit(train,val,"gru",42,cfg,tmp_path/"a",budget)


def test_final_evaluation_exclusive_marker_prevents_repeat(tmp_path, monkeypatch):
    import fedmaint.optimization.workflow as workflow
    write_json(tmp_path/"freeze.json", {"fixed":True})
    write_json(tmp_path/"evaluation-started.json", {"already_exposed":True})
    monkeypatch.setattr(workflow,"verify_freeze",lambda *args: {})
    with pytest.raises(FileExistsError):
        workflow.evaluate(tmp_path,tmp_path,{})


def test_backdated_posting_is_unverifiable_without_ingestion_time(azure_data,azure_config):
    from dataclasses import replace
    from fedmaint.data.azure import make_azure_samples
    original=make_azure_samples(azure_data,azure_config)
    # An event allegedly posted at 12:00 but backdated to 09:00: the source
    # schema has no posting time, so a one-hour event-time lag cannot exclude it.
    event=pd.DataFrame({"machineID":[1],"datetime":[pd.Timestamp("2015-01-01 09:00")],"comp":["comp4"]})
    altered=replace(azure_data,maint=pd.concat([azure_data.maint,event],ignore_index=True))
    changed=make_azure_samples(altered,azure_config)
    query=(original.machineID==1)&(original.datetime=="2015-01-01 10:00")
    assert original.loc[query,"feat_maintenance_comp4_count"].iloc[0]==0
    assert changed.loc[query,"feat_maintenance_comp4_count"].iloc[0]==1


def test_official_rul_loader_keeps_uncapped_terminal_truth_and_padding(tmp_path,cfg):
    directory=tmp_path/"data/CMAPSSData"
    directory.mkdir(parents=True)
    rows=np.array([[1,cycle,*([float(cycle)]*24)] for cycle in range(1,5)])
    np.savetxt(directory/"test_FD004.txt",rows)
    np.savetxt(directory/"RUL_FD004.txt",[175.])
    write_json(tmp_path/"data/manifest.json",{"nasa":{"files":{
        name:{"rows":n,"sha256":digest(directory/name)} for name,n in (("test_FD004.txt",4),("RUL_FD004.txt",1))}}})
    called=[]
    store=NasaStore(tmp_path,cfg,gate=lambda:called.append(True))
    frame=store.read("FD004","test")
    assert frame.raw_rul.tolist()==[178.,177.,176.,175.]
    source=frame.assign(domain="FD001")
    normal=LocalNormalizer("N1",cfg).fit(source,domain="FD001",permitted=("FD001",))
    batch,_=store.windows(frame,normal,"evaluation",terminal=True)
    assert batch.y.tolist()==[175.] and batch.mask.sum()==4
    assert batch.keys.tolist()==["FD004/test/1/4"] and called==[True]
