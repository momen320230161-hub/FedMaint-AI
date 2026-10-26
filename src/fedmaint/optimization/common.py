import hashlib
import json
from pathlib import Path
import time

from fedmaint.data.contracts import DataError

FEATURES = tuple(sorted([f"feat_setting_{i}" for i in range(1, 4)] + [f"feat_sensor_{i}" for i in range(1, 22)]))
SETTINGS = tuple(f"feat_setting_{i}" for i in range(1, 4))
DOMAINS = ("FD001", "FD002", "FD003")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(root):
    config = json.loads((Path(root) / "configs/phase021.json").read_text())
    if (config["seeds"] != [42, 43] or config["domains"] != list(DOMAINS)
            or config["fit_limit"] > 69 or config["training_seconds_limit"] > 2700
            or config["normalizers"] != ["N0", "N1", "N2"] or config["architectures"] != ["mlp", "gru"]):
        raise DataError("Config exceeds the approved experiment design")
    return config


class Budget:
    def __init__(self, output, config):
        self.path = Path(output) / "budget.json"
        self.config = config
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"fits": [], "training_seconds": 0.}

    def start(self, run_id):
        if any(r["run_id"] == run_id for r in self.state["fits"]):
            raise DataError("A planned fit may not be silently repeated")
        if len(self.state["fits"]) >= self.config["fit_limit"] or self.remaining <= 0:
            raise DataError("Approved training budget exhausted")
        self.state["fits"].append({"run_id": run_id, "status": "running"})
        write_json(self.path, self.state)
        return time.perf_counter()

    @property
    def remaining(self):
        return self.config["training_seconds_limit"] - self.state["training_seconds"]

    def finish(self, started, status, **details):
        elapsed = time.perf_counter() - started
        self.state["training_seconds"] += elapsed
        self.state["fits"][-1].update(status=status, seconds=elapsed, **details)
        write_json(self.path, self.state)
