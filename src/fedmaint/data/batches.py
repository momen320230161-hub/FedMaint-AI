"""Validated local sample contract, shared by local and pooled training."""
from dataclasses import dataclass
import numpy as np
from fedmaint.data.contracts import DataError

@dataclass
class Batch:
    x: np.ndarray
    y: np.ndarray
    keys: np.ndarray
    factories: np.ndarray
    split: str
    mask: np.ndarray | None = None
    cycles: np.ndarray | None = None

    def require(self, split, permitted):
        n = len(self.y)
        if (self.split != split or not n or self.y.shape != (n,)
            or self.keys.shape != (n,) or self.factories.shape != (n,)
            or self.x.shape[0] != n or self.x.ndim not in (2,3)
            or len(set(self.keys)) != n or not set(self.factories) <= set(permitted)
            or not np.isfinite(self.y).all()):
            raise DataError("Invalid or contaminated batch")
        if self.x.ndim == 3:
            if (self.mask is None or self.mask.shape != self.x.shape[:2]
                or self.mask.dtype != np.bool_ or not self.mask[:, -1].all()
                or (self.mask[:, :-1] & ~self.mask[:, 1:]).any()
                or self.cycles is None or self.cycles.shape != (n,)):
                raise DataError("Invalid chronological sequence contract")
            valid = self.x[self.mask]
        else:
            if self.mask is not None:
                raise DataError("Tabular batches cannot have masks")
            valid = self.x
        if not np.isfinite(valid).all():
            raise DataError("Non-finite observed input")
        return self

    @property
    def sample_count(self):
        return len(self.y)


def pool(batches):
    if not batches or len({b.split for b in batches}) != 1:
        raise DataError("Cannot pool different partitions")
    optional = lambda name: None if getattr(batches[0], name) is None else np.concatenate([getattr(b,name) for b in batches])
    return Batch(*(np.concatenate([getattr(b,name) for b in batches]) for name in ('x','y','keys','factories')),
                 batches[0].split, optional('mask'), optional('cycles'))
