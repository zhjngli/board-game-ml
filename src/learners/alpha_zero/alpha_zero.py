import os
import time
from abc import ABC
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Deque, Generic, List, NamedTuple, Sequence, Tuple

import numpy as np

from games.game import P1, P1WIN, P2WIN, Game, Immutable, NNInput, Player, State
from learners.alpha_zero.artifacts import (
    AlphaZeroPitArtifactManager,
    AlphaZeroTrainingHistoryManager,
    PendingPitState,
    PitHistoryEntry,
    TrainingHistoryEntry,
)
from learners.alpha_zero.monte_carlo_tree_search import (
    MCTSParameters,
    MonteCarloTreeSearch,
)
from learners.alpha_zero.types import A0NNOutput, Policy
from nn.neural_network import NeuralNetwork


class A0Parameters(NamedTuple):
    temp_threshold: int
    pit_games: int
    pit_threshold: float
    training_episodes: int
    training_games_per_episode: int
    training_queue_length: int
    training_hist_max_len: int
    thread_max_workers: int
    training_mcts_params: MCTSParameters
    eval_mcts_params: MCTSParameters


class SelfPlayExample(NamedTuple):
    nn_input: NNInput
    player: Player
    policy: Policy


class PitResult(NamedTuple):
    accepted: bool
    wins: int
    losses: int
    draws: int
    win_rate: float


class AlphaZero(ABC, Generic[State, Immutable]):
    """
    Combines a neural network with Monte Carlo Tree Search to increase training efficiency and reduce memory required for training.
    """

    def __init__(
        self,
        create_game: Callable[[], Game[State, Immutable]],
        create_nn: Callable[[], NeuralNetwork[NNInput, A0NNOutput]],
        params: A0Parameters,
        training_examples_folder: str,
    ) -> None:
        self.create_game = create_game
        self.create_nn = create_nn
        self.nn = create_nn()  # current neural network
        self.pn = create_nn()  # previous neural network for self-play
        self.training_history: List[TrainingHistoryEntry] = []
        self.training_history_manager = AlphaZeroTrainingHistoryManager(
            training_examples_folder, params.training_hist_max_len
        )
        self.pit_artifact_manager = AlphaZeroPitArtifactManager(
            training_examples_folder
        )

        self.training_mcts_params = params.training_mcts_params
        self.eval_mcts_params = params.eval_mcts_params

        self.temperature_threshold = params.temp_threshold
        self.pit_games = params.pit_games
        self.pit_threshold = params.pit_threshold
        self.training_episodes = params.training_episodes
        self.training_games_per_episode = params.training_games_per_episode
        self.training_queue_length = params.training_queue_length
        self.thread_max_workers = params.thread_max_workers

    def train_once(self) -> List[Tuple[NNInput, A0NNOutput]]:
        game = self.create_game()
        game.reset()

        nn = self.create_nn()
        nn.set_weights(self.nn.get_weights())
        m = MonteCarloTreeSearch(self.create_game(), nn, self.training_mcts_params)

        training_data: List[SelfPlayExample] = []
        state = game.state()
        player = state.player

        turn = 0
        while not game.check_finished(state):
            turn += 1
            oriented_state = game.orient_state(state)
            temperature = 1 if turn < self.temperature_threshold else 0
            pi = m.action_probabilities(oriented_state, temperature)

            nn_input = game.to_nn_input(oriented_state)
            syms = game.training_symmetries(nn_input, np.asarray(pi))
            for inp, p in syms:
                training_data.append(
                    SelfPlayExample(nn_input=inp, player=player, policy=p)
                )

            action = np.random.choice(len(pi), p=pi)
            state = game.apply(state, action)
            player = state.player

        reward = game.calculate_reward(state)
        return [
            (
                example.nn_input,
                A0NNOutput(
                    policy=example.policy,
                    value=reward * ((-1) ** (example.player != player)),
                ),
            )
            for example in training_data
        ]

    def train(self) -> None:
        last_ep = self.load_latest_model()
        self.training_history = self.training_history_manager.load()

        self.nn.summary()

        pit_history = self.pit_artifact_manager.load_history()
        last_ep, pit_history = self._resume_pending_pit_if_needed(last_ep, pit_history)

        for i in range(last_ep + 1, self.training_episodes):
            episode_start = time.perf_counter()
            print(f"\n{'='*60}")
            print(f"Episode {i}/{self.training_episodes - 1}")
            print(f"{'='*60}")

            print(
                f"Self-play: {self.training_games_per_episode} games with {self.thread_max_workers} threads..."
            )
            self_play_start = time.perf_counter()
            self_play_data: Deque[Tuple[NNInput, A0NNOutput]] = deque(
                [], maxlen=self.training_queue_length
            )
            with ThreadPoolExecutor(max_workers=self.thread_max_workers) as executor:
                futures = [
                    executor.submit(self.train_once)
                    for _ in range(self.training_games_per_episode)
                ]
                for future in futures:
                    self_play_data.extend(future.result())
            self_play_seconds = time.perf_counter() - self_play_start

            print(
                f"Self-play generated {len(self_play_data)} training examples"
                f" in {self._format_duration(self_play_seconds)}"
                f" ({self.training_games_per_episode / max(self_play_seconds, 1e-9):.2f} games/s)"
            )
            self.training_history = self.training_history_manager.append_episode(
                self.training_history, i, list(self_play_data)
            )

            total_examples = self.training_history_manager.total_examples(
                self.training_history
            )
            print(
                f"Training history: {len(self.training_history)} episodes, {total_examples} total examples"
            )

            print("\nTraining neural network...")
            pending_pit = self.pit_artifact_manager.build_pending_state(i)
            self.nn.save(pending_pit.previous_model_file)
            self.pn.load(pending_pit.previous_model_file)

            nn_train_start = time.perf_counter()
            self._train_current_model()
            self.nn.save(pending_pit.candidate_model_file)
            self.pit_artifact_manager.save_pending(pending_pit)
            nn_train_seconds = time.perf_counter() - nn_train_start
            print(
                f"Neural network training took {self._format_duration(nn_train_seconds)}"
            )

            print(f"\nPitting new model vs previous ({self.pit_games} games)...")
            pit_start = time.perf_counter()
            pit_result = self.pit()
            pit_seconds = time.perf_counter() - pit_start
            pit_history.append(
                PitHistoryEntry(
                    episode=i,
                    accepted=pit_result.accepted,
                    wins=pit_result.wins,
                    losses=pit_result.losses,
                    draws=pit_result.draws,
                    win_rate=pit_result.win_rate,
                )
            )
            print(
                "Episode timing:"
                f" self_play={self._format_duration(self_play_seconds)}"
                f" | nn_train={self._format_duration(nn_train_seconds)}"
                f" | pit={self._format_duration(pit_seconds)}"
                f" | total={self._format_duration(time.perf_counter() - episode_start)}"
            )

            if pit_result.accepted:
                print(f"New model ACCEPTED — saving as ep_{i:07d} and best_model")
                self.nn.save(f"ep_{i:07d}_model.weights.h5")
                self.nn.save("best_model.weights.h5")
            else:
                print("New model REJECTED — reverting to previous model")
                self.nn.load(pending_pit.previous_model_file)

            self.pit_artifact_manager.clear_pending(self.nn.model_folder)
            self.pit_artifact_manager.save_history(pit_history)
            self._print_pit_summary(pit_history)

        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}")

    def _resume_pending_pit_if_needed(
        self, last_ep: int, pit_history: List[PitHistoryEntry]
    ) -> Tuple[int, List[PitHistoryEntry]]:
        pending_pit = self.pit_artifact_manager.load_pending()
        if pending_pit is None:
            return last_ep, pit_history

        if (
            pending_pit.episode <= last_ep
            or not self.pit_artifact_manager.pending_models_exist(
                self.nn.model_folder, pending_pit
            )
        ):
            print("Found stale or incomplete pending pit artifact; removing it.")
            self.pit_artifact_manager.clear_pending(self.nn.model_folder)
            return last_ep, pit_history

        resolved = self.resolve_pending_pit(pending_pit)
        next_pit_history = pit_history + [resolved]
        self.pit_artifact_manager.save_history(next_pit_history)
        return max(last_ep, resolved.episode), next_pit_history

    def _train_current_model(self) -> None:
        while True:
            try:
                # Reload the full rolling window from disk right before fitting.
                # This preserves the old "train on the entire window at once"
                # semantics, including the neural network's global shuffle.
                training_data = self.training_history_manager.load_flat(
                    self.training_history
                )
                self.nn.train(training_data)
                return
            except Exception as e:
                print(f"Failed to train with error {e}, retrying...")

    def pit(self) -> PitResult:
        pit_start = time.perf_counter()
        candidate_wins = 0
        previous_wins = 0
        draws = 0
        for i in range(self.pit_games):
            candidate_is_p1 = i >= int(self.pit_games / 2)

            game = self.create_game()
            game.reset()
            state = game.state()
            player = state.player
            previous_mcts = MonteCarloTreeSearch(
                self.create_game(), self.pn, self.eval_mcts_params
            )
            candidate_mcts = MonteCarloTreeSearch(
                self.create_game(), self.nn, self.eval_mcts_params
            )

            while not game.check_finished(state):
                current_mcts = (
                    candidate_mcts
                    if candidate_is_p1 == (player == P1)
                    else previous_mcts
                )
                a = int(
                    np.argmax(current_mcts.action_probabilities(state, temperature=0))
                )
                state = game.apply(state, a)
                player = state.player

            r = game.calculate_reward(state)
            if r == P1WIN:
                if candidate_is_p1:
                    candidate_wins += 1
                else:
                    previous_wins += 1
            elif r == P2WIN:
                if candidate_is_p1:
                    previous_wins += 1
                else:
                    candidate_wins += 1
            else:
                draws += 1

        win_rate = (
            candidate_wins / (candidate_wins + previous_wins)
            if (candidate_wins + previous_wins) > 0
            else 0.0
        )
        accepted = candidate_wins + previous_wins != 0 and win_rate > self.pit_threshold
        pit_seconds = time.perf_counter() - pit_start
        print(
            f"Pit results: new model {candidate_wins}W / {previous_wins}L / {draws}D"
            f" — win rate {win_rate:.1%} (threshold {self.pit_threshold:.0%})"
            f" | time {self._format_duration(pit_seconds)}"
            f" | avg_game {pit_seconds / max(self.pit_games, 1):.2f}s"
        )
        return PitResult(
            accepted=accepted,
            wins=candidate_wins,
            losses=previous_wins,
            draws=draws,
            win_rate=win_rate,
        )

    @staticmethod
    def _print_pit_summary(pit_history: Sequence[PitHistoryEntry]) -> None:
        def _avg_win_rate(entries: Sequence[PitHistoryEntry]) -> str:
            if not entries:
                return "n/a"
            return f"{sum(entry.win_rate for entry in entries) / len(entries):.1%}"

        def _accept_rate(entries: Sequence[PitHistoryEntry]) -> str:
            if not entries:
                return "n/a"
            accepted = sum(1 for entry in entries if entry.accepted)
            return f"{accepted}/{len(entries)}"

        recent = list(pit_history[-10:])
        print("\nPit history (last 10):")
        for entry in recent:
            status = "ACCEPTED" if entry.accepted else "rejected"
            print(
                f"  ep {entry.episode:>3}: {entry.wins}W/{entry.losses}L/{entry.draws}D"
                f" {entry.win_rate:.1%} {status}"
            )

        last_5 = pit_history[-5:]
        last_10 = pit_history[-10:]
        parts = [f"last 5: {_avg_win_rate(last_5)}"]
        if len(pit_history) > 5:
            parts.append(f"last 10: {_avg_win_rate(last_10)}")
        if len(pit_history) > 10:
            parts.append(f"all: {_avg_win_rate(pit_history)}")
        print(
            f"Accept rate: {_accept_rate(last_10)}  |  Avg win rate: {'  |  '.join(parts)}"
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"

        minutes, rem_seconds = divmod(seconds, 60)
        if minutes < 60:
            return f"{int(minutes)}m {rem_seconds:.1f}s"

        hours, rem_minutes = divmod(minutes, 60)
        return f"{int(hours)}h {int(rem_minutes)}m {rem_seconds:.1f}s"

    def resolve_pending_pit(self, pending_pit: PendingPitState) -> PitHistoryEntry:
        print(
            f"Resuming pending pit for episode {pending_pit.episode}"
            f" from {pending_pit.candidate_model_file}"
        )

        self.pn.load(pending_pit.previous_model_file)
        self.nn.load(pending_pit.candidate_model_file)
        pit_result = self.pit()

        if pit_result.accepted:
            print(
                f"Recovered candidate ACCEPTED — saving as ep_{pending_pit.episode:07d}"
                " and best_model"
            )
            self.nn.save(f"ep_{pending_pit.episode:07d}_model.weights.h5")
            self.nn.save("best_model.weights.h5")
        else:
            print("Recovered candidate REJECTED — reverting to previous model")
            self.nn.load(pending_pit.previous_model_file)

        self.pit_artifact_manager.clear_pending(self.nn.model_folder)
        return PitHistoryEntry(
            episode=pending_pit.episode,
            accepted=pit_result.accepted,
            wins=pit_result.wins,
            losses=pit_result.losses,
            draws=pit_result.draws,
            win_rate=pit_result.win_rate,
        )

    def load_latest_model(self) -> int:
        """
        Loads latest model and returns latest training episode if training stops for whatever reason.
        """
        latest = 0
        if not os.path.isdir(self.nn.model_folder):
            return latest

        latest_model = None
        for filename in os.listdir(self.nn.model_folder):
            path = os.path.join(self.nn.model_folder, filename)
            if not os.path.isfile(path):
                continue
            try:
                episode = int(filename.split("_")[1])  # ep_0001_model.weights.h5
            except ValueError:
                continue
            if episode >= latest:
                latest = episode
                latest_model = path

        if latest_model:
            self.nn.load(latest_model)
            self.pn.load(latest_model)  # TODO: some other form of previous model?
        return latest
