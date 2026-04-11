import os
from pathlib import Path

import numpy as np

from learners.alpha_zero.artifacts import (
    AlphaZeroPitArtifactManager,
    AlphaZeroTrainingHistoryManager,
    PendingPitState,
)
from learners.alpha_zero.types import A0NNOutput


def _example(index: int):
    return (
        np.full((3, 3), index, dtype=np.float32),
        A0NNOutput(policy=np.full(9, index, dtype=np.float32), value=float(index)),
    )


def test_training_history_manager_appends_and_prunes(tmp_path: Path):
    manager = AlphaZeroTrainingHistoryManager(str(tmp_path), max_entries=2)

    history = manager.append_episode([], 1, [_example(1)])
    history = manager.append_episode(history, 2, [_example(2)])
    history = manager.append_episode(history, 3, [_example(3)])

    assert [entry.episode for entry in history] == [2, 3]
    assert manager.total_examples(history) == 2
    assert not os.path.exists(tmp_path / manager.training_examples_file(1))
    assert os.path.exists(tmp_path / manager.training_examples_file(2))
    assert os.path.exists(tmp_path / manager.training_examples_file(3))


def test_training_history_manager_requires_manifest(tmp_path: Path):
    manager = AlphaZeroTrainingHistoryManager(str(tmp_path), max_entries=10)
    history = manager.load()

    assert history == []


def test_pit_artifact_manager_persists_and_clears_pending(tmp_path: Path):
    artifact_folder = tmp_path / "artifacts"
    model_folder = tmp_path / "models"
    model_folder.mkdir()

    manager = AlphaZeroPitArtifactManager(str(artifact_folder))
    pending = manager.build_pending_state(5)
    manager.save_pending(pending)
    manager.save_history([(5, True, 3, 1, 0, 0.75)])

    for model_file in (pending.previous_model_file, pending.candidate_model_file):
        (model_folder / model_file).write_text("weights")

    assert manager.load_pending() == PendingPitState(
        episode=5,
        previous_model_file=pending.previous_model_file,
        candidate_model_file=pending.candidate_model_file,
    )
    assert manager.pending_models_exist(str(model_folder), pending)
    assert manager.load_history() == [(5, True, 3, 1, 0, 0.75)]

    manager.clear_pending(str(model_folder))
    assert manager.load_pending() is None
    assert not (model_folder / pending.previous_model_file).exists()
    assert not (model_folder / pending.candidate_model_file).exists()
