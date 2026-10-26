"""One deterministic PyTorch training loop for both experimental tracks."""
import copy
import hashlib
import random
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score
from fedmaint.data.contracts import DataError
from .network import build_model
from .metrics import validation_threshold


def initialize(seed, device='cpu'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if device == 'cuda' and not torch.cuda.is_available():
        raise DataError('CUDA requested but unavailable')


def predict(model, batch, dataset, scale=100., batch_size=512):
    # Use the recorded CPU numerical policy in standalone inference as well as fitting.
    # Otherwise BLAS thread differences can change float32 outputs after reload.
    torch.set_num_threads(1)
    model.eval()
    device = next(model.parameters()).device
    outputs = []
    with torch.no_grad():
        for start in range(0,len(batch.y),batch_size):
            x = torch.as_tensor(batch.x[start:start+batch_size],dtype=torch.float32,device=device)
            mask = None if batch.mask is None else torch.as_tensor(batch.mask[start:start+batch_size],device=device)
            output = model(x,mask)
            output = torch.sigmoid(output) if dataset == 'azure' else output*scale
            outputs.append(output.cpu().double().numpy())
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise DataError('Non-finite predictions')
    return result


def score_validation(batch, prediction, dataset):
    if dataset == 'azure':
        return float(average_precision_score(batch.y,prediction))
    scores=[]
    for client in sorted(set(batch.factories)):
        for cycle in (50,100):
            index=(batch.factories==client)&(batch.cycles==cycle)
            if not index.any():
                raise DataError('Required validation prefix missing')
            scores.append(np.sqrt(np.mean((prediction[index]-batch.y[index])**2)))
    return -float(np.mean(scores))


def fit(train, validation, dataset, architecture, config, seed, permitted, window=30, local_epochs=None):
    train.require('train',permitted)
    validation.require('validation',permitted)
    if (set(train.keys)&set(validation.keys) or set(train.factories)!=set(validation.factories)
        or train.x.shape[1:]!=validation.x.shape[1:] or seed not in config.seeds
        or any(c in ('FD004','azure-model4') for c in permitted)):
        raise DataError('Fit/validation contract contamination')
    if dataset == 'nasa':
        if (train.y<0).any() or (validation.y<0).any():
            raise DataError('Negative RUL labels')
        train_engines={k.rsplit('/',1)[0] for k in train.keys}
        val_engines={k.rsplit('/',1)[0] for k in validation.keys}
        if train_engines & val_engines:
            raise DataError('Engine leakage')
    elif any(not np.array_equal(np.unique(b.y),[0,1]) for b in (train,validation)):
        raise DataError('Both classes required for train and validation')
    epochs=config.epochs if local_epochs is None else local_epochs
    if not isinstance(epochs,int) or not 1<=epochs<=config.epochs:
        raise DataError('Invalid local epoch budget')
    initialize(seed,config.device)
    model=build_model(dataset,architecture,train.x.shape[-1],config.hidden,window).to(config.device)
    x=torch.as_tensor(train.x,dtype=torch.float32)
    mask=torch.ones((len(train.y),1),dtype=torch.bool) if train.mask is None else torch.from_numpy(train.mask)
    y=torch.as_tensor(train.y/(1 if dataset=='azure' else config.rul_scale),dtype=torch.float32)
    loader=DataLoader(TensorDataset(x,mask,y),batch_size=config.batch_size,shuffle=True,
                      generator=torch.Generator().manual_seed(seed),num_workers=0)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    best,best_score,stale,history,selected=None,-float('inf'),0,[],None
    started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train()
        total=0.
        for bx,bm,by in loader:
            bx,bm,by=bx.to(config.device),bm.to(config.device),by.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            output=model(bx,None if dataset=='azure' else bm)
            base=(torch.nn.functional.binary_cross_entropy_with_logits(output,by) if dataset=='azure'
                  else .5*(output-by).square().mean())
            penalty=sum(p.square().sum() for name,p in model.named_parameters() if 'weight' in name)
            loss=base+config.alpha/(2*len(train.y))*penalty
            if not torch.isfinite(loss):
                raise DataError('Non-finite loss')
            loss.backward()
            optimizer.step()
            total+=float(loss.detach())*len(by)
        prediction=predict(model,validation,dataset,config.rul_scale)
        score=score_validation(validation,prediction,dataset)
        history.append({'epoch':epoch,'loss':total/len(train.y),'validation_score':score})
        if score>best_score:
            best,best_score,stale,selected=copy.deepcopy(model.state_dict()),score,0,epoch
        else:
            stale+=1
        if stale>=config.patience:
            break
    model.load_state_dict(best)
    threshold=(validation_threshold(validation,predict(model,validation,dataset),permitted) if dataset=='azure' else None)
    membership={name:{'keys_sha256':hashlib.sha256('\n'.join(b.keys).encode()).hexdigest(),
                      'units':sorted(set(k.rsplit('/',1)[0] for k in b.keys)),
                      'samples':len(b.y),'factories':sorted(set(b.factories))}
                for name,b in (('train',train),('validation',validation))}
    return model,{'selected_epoch':selected,'validation_score':best_score,'threshold':threshold,
                  'criterion':'validation_ap' if dataset=='azure' else 'negative_macro_prefix_rmse',
                  'history':history,'membership':membership,'seconds':time.perf_counter()-started,
                  'seed':seed,'local_epochs':epochs,'sample_count':len(train.y)}
