import os
import pickle
import tempfile
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple

from games.game import NNInput
from learners.alpha_zero.types import A0NNOutput

PitHistoryEntry = Tuple[int, bool, int, int, int, float]
TrainingExamples = List[Tuple[NNInput, A0NNOutput]]


class PendingPitState(NamedTuple):
    episode: int
    previous_model_file: str
    candidate_model_file: str


class TrainingHistoryEntry(NamedTuple):
    episode: int
    examples_file: str
    num_examples: int


def _ensure_folder(folder: str) -> None:
    if os.path.exists(folder):
        return

    print(f"Making directory for AlphaZero artifacts at: {folder}")
    os.makedirs(folder)


def _write_pickle_atomic(folder: str, path: str, value: object) -> None:
    _ensure_folder(folder)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".pkl", dir=folder)
    try:
        with os.fdopen(fd, "wb") as file:
            pickle.dump(value, file)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


class AlphaZeroTrainingHistoryManager:
    def __init__(self, training_examples_folder: str, max_entries: int) -> None:
        self.training_examples_folder = training_examples_folder
        self.max_entries = max_entries

    def ensure_folder(self) -> None:
        _ensure_folder(self.training_examples_folder)

    def manifest_file(self) -> str:
        return os.path.join(
            self.training_examples_folder, "training_history_manifest.pkl"
        )

    def training_examples_file(self, episode: int) -> str:
        return f"training_examples_ep_{episode:07d}.pkl"

    def training_examples_path(self, examples_file: str) -> str:
        return os.path.join(self.training_examples_folder, examples_file)

    def load(self) -> List[TrainingHistoryEntry]:
        if not os.path.isfile(self.manifest_file()):
            return []

        with open(self.manifest_file(), "rb") as file:
            return pickle.load(file)

    def append_episode(
        self,
        history_entries: Sequence[TrainingHistoryEntry],
        episode: int,
        training_examples: Iterable[Tuple[NNInput, A0NNOutput]],
    ) -> List[TrainingHistoryEntry]:
        entry = self.save_training_examples_episode(episode, training_examples)
        next_history = list(history_entries) + [entry]
        if self.max_entries > 0 and len(next_history) > self.max_entries:
            dropped_entries = next_history[: -self.max_entries]
            kept_entries = next_history[-self.max_entries :]
        else:
            dropped_entries = []
            kept_entries = next_history

        self.save_manifest(kept_entries)
        for dropped_entry in dropped_entries:
            self.delete_training_examples(dropped_entry.examples_file)
        return kept_entries

    def total_examples(self, history_entries: Sequence[TrainingHistoryEntry]) -> int:
        return sum(entry.num_examples for entry in history_entries)

    def load_all(
        self, history_entries: Sequence[TrainingHistoryEntry]
    ) -> List[TrainingExamples]:
        training_history: List[TrainingExamples] = []
        for entry in history_entries:
            with open(self.training_examples_path(entry.examples_file), "rb") as file:
                training_history.append(pickle.load(file))
        return training_history

    def load_flat(
        self, history_entries: Sequence[TrainingHistoryEntry]
    ) -> TrainingExamples:
        training_data: TrainingExamples = []
        for entry in history_entries:
            with open(self.training_examples_path(entry.examples_file), "rb") as file:
                training_data.extend(pickle.load(file))
        return training_data

    def save_training_examples_episode(
        self,
        episode: int,
        training_examples: Iterable[Tuple[NNInput, A0NNOutput]],
    ) -> TrainingHistoryEntry:
        self.ensure_folder()

        training_examples_list = list(training_examples)
        examples_file = self.training_examples_file(episode)
        self._write_pickle_atomic(
            self.training_examples_path(examples_file), training_examples_list
        )

        return TrainingHistoryEntry(
            episode=episode,
            examples_file=examples_file,
            num_examples=len(training_examples_list),
        )

    def save_manifest(self, history_entries: Sequence[TrainingHistoryEntry]) -> None:
        self.ensure_folder()
        _write_pickle_atomic(
            self.training_examples_folder, self.manifest_file(), list(history_entries)
        )

    def delete_training_examples(self, examples_file: str) -> None:
        training_examples_path = self.training_examples_path(examples_file)
        if os.path.isfile(training_examples_path):
            os.remove(training_examples_path)

    @classmethod
    def load_history_data_from_folder(
        cls, training_examples_folder: str
    ) -> List[TrainingExamples]:
        manager = cls(training_examples_folder, max_entries=0)
        history_entries = manager.load()
        return manager.load_all(history_entries)

    def _write_pickle_atomic(self, path: str, value: object) -> None:
        _write_pickle_atomic(self.training_examples_folder, path, value)


class AlphaZeroPitArtifactManager:
    def __init__(self, training_examples_folder: str) -> None:
        self.training_examples_folder = training_examples_folder

    def ensure_folder(self) -> None:
        _ensure_folder(self.training_examples_folder)

    def pending_pit_file(self) -> str:
        return os.path.join(self.training_examples_folder, "pending_pit.pkl")

    def pit_history_file(self) -> str:
        return os.path.join(self.training_examples_folder, "pit_history.pkl")

    def previous_model_file(self, episode: int) -> str:
        return f"pending_pit_ep_{episode:07d}_previous.weights.h5"

    def candidate_model_file(self, episode: int) -> str:
        return f"pending_pit_ep_{episode:07d}_candidate.weights.h5"

    def build_pending_state(self, episode: int) -> PendingPitState:
        return PendingPitState(
            episode=episode,
            previous_model_file=self.previous_model_file(episode),
            candidate_model_file=self.candidate_model_file(episode),
        )

    def save_pending(self, pending_state: PendingPitState) -> None:
        self.ensure_folder()
        self._write_pickle_atomic(self.pending_pit_file(), pending_state)

    def load_pending(self) -> Optional[PendingPitState]:
        if not os.path.isfile(self.pending_pit_file()):
            return None

        with open(self.pending_pit_file(), "rb") as file:
            return pickle.load(file)

    def clear_pending(self, model_folder: str) -> None:
        pending_state = self.load_pending()
        if pending_state is not None:
            for model_file in (
                pending_state.previous_model_file,
                pending_state.candidate_model_file,
            ):
                model_path = os.path.join(model_folder, model_file)
                if os.path.isfile(model_path):
                    os.remove(model_path)

        if os.path.isfile(self.pending_pit_file()):
            os.remove(self.pending_pit_file())

    def pending_models_exist(
        self, model_folder: str, pending_state: PendingPitState
    ) -> bool:
        return all(
            os.path.isfile(os.path.join(model_folder, model_file))
            for model_file in (
                pending_state.previous_model_file,
                pending_state.candidate_model_file,
            )
        )

    def load_history(self) -> List[PitHistoryEntry]:
        if not os.path.isfile(self.pit_history_file()):
            return []

        with open(self.pit_history_file(), "rb") as file:
            return pickle.load(file)

    def save_history(self, pit_history: Sequence[PitHistoryEntry]) -> None:
        self.ensure_folder()
        self._write_pickle_atomic(self.pit_history_file(), list(pit_history))

    def _write_pickle_atomic(self, path: str, value: object) -> None:
        _write_pickle_atomic(self.training_examples_folder, path, value)
