"""A shared numerical backbone; private normalization is deliberately not model state."""

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence

from fedmaint.data.contracts import DataError


class RULModel(nn.Module):
    def __init__(self, architecture, features=24, hidden=32, window=30):
        super().__init__()
        if architecture not in ("mlp", "gru"):
            raise DataError("Unsupported backbone")
        self.architecture, self.features, self.window = architecture, features, window
        if architecture == "mlp":
            self.hidden = nn.Linear(window*(features+1), hidden)
        else:
            self.recurrent = nn.GRU(features, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        if (x.ndim != 3 or x.shape[2] != self.features or mask.shape != x.shape[:2]
                or mask.dtype != torch.bool or not bool(mask[:, -1].all())
                or bool((mask[:, :-1] & ~mask[:, 1:]).any())):
            raise DataError("Invalid chronological left-padding mask/input")
        if self.architecture == "mlp":
            if x.shape[1] != self.window:
                raise DataError("MLP requires the frozen window dimension")
            clean = torch.where(mask.unsqueeze(-1), x, 0)
            encoded = torch.cat((clean.flatten(1), mask.to(x.dtype)), dim=1)
            hidden = torch.relu(self.hidden(encoded))
        elif bool(mask.all()):
            _, h = self.recurrent(x)
            hidden = h[-1]
        else:
            # Strip LEFT padding; passing left-padded rows directly to a packer would be wrong.
            packed = pack_sequence([row[valid] for row, valid in zip(x, mask)], enforce_sorted=False)
            _, h = self.recurrent(packed)
            hidden = h[-1]
        return self.head(hidden).squeeze(-1)



class AzureMLP(nn.Module):
    def __init__(self, features=16, hidden=32):
        super().__init__()
        self.features = features
        self.hidden = nn.Linear(features, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, mask=None):
        if x.ndim != 2 or x.shape[1] != self.features or mask is not None:
            raise DataError("Invalid Azure feature contract")
        return self.head(torch.relu(self.hidden(x))).squeeze(-1)


def build_model(dataset, architecture, features, hidden=32, window=30):
    if dataset == 'azure' and architecture == 'mlp':
        return AzureMLP(features, hidden)
    if dataset == 'nasa' and features == 24:
        return RULModel(architecture, features, hidden, window)
    raise DataError('Invalid model/task contract')


def export_state(model):
    """Independent CPU numeric copies; exclusively neural parameters."""
    return {k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}


def import_state(model, state):
    expected = model.state_dict()
    if set(state) != set(expected):
        raise DataError('Model state names differ')
    loaded = {}
    for name,tensor in expected.items():
        value = state[name]
        if value.shape != tuple(tensor.shape) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise DataError('Invalid model state shape/dtype/values')
        loaded[name] = torch.from_numpy(value.copy()).to(tensor.device)
    model.load_state_dict(loaded, strict=True)


def parameter_contract(model):
    state = export_state(model)
    return {'parameters':sum(v.size for v in state.values()), 'bytes':sum(v.nbytes for v in state.values()),
            'tensors':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in state.items()}}
