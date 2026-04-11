#!/usr/bin/env python3

import argparse
import os
import pathlib
import pickle
import sys
from typing import List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from learners.alpha_zero.artifacts import (  # noqa: E402
    AlphaZeroTrainingHistoryManager,
    TrainingExamples,
)


def _latest_legacy_snapshot(training_examples_folder: str) -> Optional[str]:
    latest_episode = -1
    latest_path = None
    for filename in os.listdir(training_examples_folder):
        path = os.path.join(training_examples_folder, filename)
        if not os.path.isfile(path):
            continue
        if not filename.startswith("training_examples_") or filename.startswith(
            "training_examples_ep_"
        ):
            continue
        try:
            episode = int(filename.split(".")[0].split("_")[-1])
        except ValueError:
            continue
        if episode >= latest_episode:
            latest_episode = episode
            latest_path = path
    return latest_path


def migrate_training_history(
    training_examples_folder: str, legacy_snapshot_path: str
) -> None:
    with open(legacy_snapshot_path, "rb") as file:
        legacy_history: List[TrainingExamples] = pickle.load(file)

    latest_episode = int(
        os.path.basename(legacy_snapshot_path).split(".")[0].split("_")[-1]
    )
    start_episode = max(0, latest_episode - len(legacy_history) + 1)
    manager = AlphaZeroTrainingHistoryManager(training_examples_folder, max_entries=0)

    history_entries = []
    for offset, training_examples in enumerate(legacy_history):
        history_entries.append(
            manager.save_training_examples_episode(
                start_episode + offset, training_examples
            )
        )
    manager.save_manifest(history_entries)

    print(
        "Migrated legacy AlphaZero training history"
        f" from {os.path.basename(legacy_snapshot_path)}"
        f" into {len(history_entries)} per-episode artifacts"
        f" in {training_examples_folder}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a legacy AlphaZero training_examples_XXXXXXX.pkl snapshot into"
            " the manifest + per-episode artifact format."
        )
    )
    parser.add_argument(
        "training_examples_folder",
        help="Folder containing the legacy training snapshot and/or new artifacts.",
    )
    parser.add_argument(
        "--snapshot",
        help=(
            "Specific legacy snapshot file to migrate. Defaults to the latest"
            " legacy training_examples_XXXXXXX.pkl in the folder."
        ),
    )
    args = parser.parse_args()

    folder = os.path.abspath(args.training_examples_folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"Training examples folder does not exist: {folder}")

    snapshot = os.path.abspath(args.snapshot) if args.snapshot else None
    if snapshot is None:
        snapshot = _latest_legacy_snapshot(folder)
    if snapshot is None or not os.path.isfile(snapshot):
        raise SystemExit(
            "Could not find a legacy AlphaZero snapshot to migrate."
            " Pass --snapshot explicitly if needed."
        )

    migrate_training_history(folder, snapshot)


if __name__ == "__main__":
    main()
