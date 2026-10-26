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


