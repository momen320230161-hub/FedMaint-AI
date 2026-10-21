"""Internal loading interface only; no prediction/training HTTP endpoints in Phase 02."""

from pathlib import Path

from fedmaint.models.artifacts import LoadedModel, load_bundle


def load_model_artifact(directory: Path, *, expected_dataset: str) -> LoadedModel:
    """Load numeric weights plus frozen transformations, rejecting mismatches/tampering."""
    return load_bundle(directory, expected_dataset=expected_dataset)
