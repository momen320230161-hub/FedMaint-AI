"""Validated internal model interface; no future HTTP endpoints are exposed."""
from pathlib import Path
from fedmaint.models.artifacts import LoadedModel, load_bundle


def load_model_artifact(directory: Path, *, expected_dataset: str) -> LoadedModel:
    return load_bundle(directory,expected_dataset=expected_dataset)
