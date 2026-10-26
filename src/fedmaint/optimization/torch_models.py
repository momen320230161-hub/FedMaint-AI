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


def predict(model, windows, scale=100., batch_size=512):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(windows.y), batch_size):
            x = torch.as_tensor(windows.x[start:start+batch_size])
            mask = torch.as_tensor(windows.mask[start:start+batch_size])
            outputs.append(model(x, mask).double().numpy()*scale)
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise DataError("Non-finite raw model predictions")
    return result


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def save_weights(path, model):
    np.savez_compressed(path, **{name: value.detach().cpu().numpy() for name,value in model.state_dict().items()})


def load_weights(path, architecture, config):
    model = RULModel(architecture, hidden=config["hidden"], window=config["window"])
    expected = model.state_dict()
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected):
            raise DataError("Checkpoint parameter contract mismatch")
        loaded = {}
        for name, tensor in expected.items():
            value = arrays[name]
            if value.shape != tuple(tensor.shape) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise DataError("Checkpoint dimensions/dtype/values invalid")
            loaded[name] = torch.from_numpy(value.copy())
    model.load_state_dict(loaded)
    return model
