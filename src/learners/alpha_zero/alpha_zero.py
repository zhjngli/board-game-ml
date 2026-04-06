import os
import pickle
import time
from abc import ABC
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Deque, Generic, List, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from games.game import P1, P1WIN, P2WIN, Action, Game, Immutable, NNInput, Player, State
from learners.alpha_zero.monte_carlo_tree_search import (
    MCTSParameters,
    MonteCarloTreeSearch,
)
from learners.alpha_zero.types import A0NNOutput
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


class AlphaZero(ABC, Generic[State, Immutable]):
    """
    Combines a neural network with Monte Carlo Tree Search to increase training efficiency and reduce memory required for training.
    """

    def __init__(
        self,
        create_game: Callable[[], Game[State, Immutable]],
        create_nn: Callable[[], NeuralNetwork],
        params: A0Parameters,
        training_examples_folder: str,
    ) -> None:
        self.create_game = create_game
        self.create_nn = create_nn
        self.nn = create_nn()  # current neural network
        self.pn = create_nn()  # previous neural network for self-play
        self.training_history: List[Deque[Tuple[NNInput, A0NNOutput]]] = []
        self.training_examples_folder = training_examples_folder

        self.training_mcts_params = params.training_mcts_params
        self.eval_mcts_params = params.eval_mcts_params

        self.temperature_threshold = params.temp_threshold
        self.pit_games = params.pit_games
        self.pit_threshold = params.pit_threshold
        self.training_episodes = params.training_episodes
        self.training_games_per_episode = params.training_games_per_episode
        self.training_queue_length = params.training_queue_length
        self.training_hist_max_len = params.training_hist_max_len
        self.thread_max_workers = params.thread_max_workers

    def train_once(self) -> List[Tuple[NNInput, A0NNOutput]]:
        game = self.create_game()
        game.reset()

        nn = self.create_nn()
        nn.set_weights(self.nn.get_weights())
        m = MonteCarloTreeSearch(self.create_game(), nn, self.training_mcts_params)

        training_data: List[Tuple[NNInput, Player, NDArray, Optional[float]]] = []
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
                training_data.append((inp, player, p, None))

            action = np.random.choice(len(pi), p=pi)
            state = game.apply(state, action)
            player = state.player

        reward = game.calculate_reward(state)
        return [
            (x[0], A0NNOutput(policy=x[2], value=reward * ((-1) ** (x[1] != player))))
            for x in training_data
        ]

    def train(self) -> None:
        last_ep = self.load_latest_model()
        self.load_training_history()

        self.nn.summary()

        pit_history: List[Tuple[int, bool, int, int, int, float]] = []
        for i in range(last_ep + 1, self.training_episodes):
            episode_start = time.perf_counter()
            print(f"\n{'='*60}")
            print(f"Episode {i}/{self.training_episodes - 1}")
            print(f"{'='*60}")

            # self play
            print(
                f"Self-play: {self.training_games_per_episode} games with {self.thread_max_workers} threads..."
            )
            self_play_start = time.perf_counter()
            self_play_data: Deque[Tuple[NDArray, A0NNOutput]] = deque(
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
            self.training_history.append(self_play_data)

            if len(self.training_history) > self.training_hist_max_len:
                self.training_history.pop(0)

            total_examples = sum(len(d) for d in self.training_history)
            print(
                f"Training history: {len(self.training_history)} episodes, {total_examples} total examples"
            )

            # i-1: last episode's model played these games
            self.save_training_history(f"training_examples_{i-1:07d}.pkl")

            # train model
            print("\nTraining neural network...")
            self.nn.save("temp_model.weights.h5")
            self.pn.load("temp_model.weights.h5")

            training_data = [
                d for game_data in self.training_history for d in game_data
            ]
            successful_train = False
            nn_train_start = time.perf_counter()
            while not successful_train:
                try:
                    # shuffle(training_data)  # can shuffle in nn.train()
                    self.nn.train(training_data)
                    successful_train = True
                except Exception as e:
                    print(f"Failed to train with error {e}, retrying...")
            nn_train_seconds = time.perf_counter() - nn_train_start
            print(
                f"Neural network training took {self._format_duration(nn_train_seconds)}"
            )

            # if model is good enough, keep it
            print(f"\nPitting new model vs previous ({self.pit_games} games)...")
            pit_start = time.perf_counter()
            accepted, wins, losses, draws, win_rate = self.pit()
            pit_seconds = time.perf_counter() - pit_start
            pit_history.append((i, accepted, wins, losses, draws, win_rate))
            print(
                "Episode timing:"
                f" self_play={self._format_duration(self_play_seconds)}"
                f" | nn_train={self._format_duration(nn_train_seconds)}"
                f" | pit={self._format_duration(pit_seconds)}"
                f" | total={self._format_duration(time.perf_counter() - episode_start)}"
            )

            if accepted:
                print(f"New model ACCEPTED — saving as ep_{i:07d} and best_model")
                self.nn.save(f"ep_{i:07d}_model.weights.h5")
                self.nn.save("best_model.weights.h5")
            else:
                print("New model REJECTED — reverting to previous model")
                self.nn.load("temp_model.weights.h5")

            self._print_pit_summary(pit_history)

        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}")

    def pit(self) -> Tuple[bool, int, int, int, float]:
        pit_start = time.perf_counter()
        prev_mtcs = MonteCarloTreeSearch(
            self.create_game(), self.pn, self.eval_mcts_params
        )
        candidate = MonteCarloTreeSearch(
            self.create_game(), self.nn, self.eval_mcts_params
        )
        play1: Callable[[State], Action] = lambda s: int(
            np.argmax(prev_mtcs.action_probabilities(s, temperature=0))
        )
        play2: Callable[[State], Action] = lambda s: int(
            np.argmax(candidate.action_probabilities(s, temperature=0))
        )

        # TODO: what's the effect of using the neural network's prediction instead of tree search?
        # play1 = lambda s: np.argmax(self.pn.predict(s)[0])
        # play2 = lambda s: np.argmax(self.nn.predict(s)[0])

        p1wins = 0
        p2wins = 0
        draws = 0
        for i in range(self.pit_games):
            if i == int(self.pit_games / 2):
                # switch first player
                play1, play2 = play2, play1
                p1wins, p2wins = p2wins, p1wins

            game = self.create_game()
            game.reset()
            state = game.state()
            player = state.player

            while not game.check_finished(state):
                # oriented_state = self.game.oriented_state(state)  # only needed for nn prediction
                play = play1 if player == P1 else play2
                a = play(state)
                state = game.apply(state, a)
                player = state.player

            r = game.calculate_reward(state)
            if r == P1WIN:
                p1wins += 1
            elif r == P2WIN:
                p2wins += 1
            else:
                draws += 1

        # TODO: should win percentage be based on total games?
        # candidate becomes p1 after the switch
        win_rate = p1wins / (p1wins + p2wins) if (p1wins + p2wins) > 0 else 0.0
        accepted = p1wins + p2wins != 0 and win_rate > self.pit_threshold
        pit_seconds = time.perf_counter() - pit_start
        print(
            f"Pit results: new model {p1wins}W / {p2wins}L / {draws}D"
            f" — win rate {win_rate:.1%} (threshold {self.pit_threshold:.0%})"
            f" | time {self._format_duration(pit_seconds)}"
            f" | avg_game {pit_seconds / max(self.pit_games, 1):.2f}s"
        )
        return accepted, p1wins, p2wins, draws, win_rate

    @staticmethod
    def _print_pit_summary(
        pit_history: List[Tuple[int, bool, int, int, int, float]],
    ) -> None:
        def _avg_win_rate(
            entries: List[Tuple[int, bool, int, int, int, float]],
        ) -> str:
            if not entries:
                return "n/a"
            return f"{sum(e[5] for e in entries) / len(entries):.1%}"

        def _accept_rate(
            entries: List[Tuple[int, bool, int, int, int, float]],
        ) -> str:
            if not entries:
                return "n/a"
            accepted = sum(1 for e in entries if e[1])
            return f"{accepted}/{len(entries)}"

        # pit history table (last 10)
        recent = pit_history[-10:]
        print("\nPit history (last 10):")
        for ep, accepted, w, l, d, wr in recent:
            status = "ACCEPTED" if accepted else "rejected"
            print(f"  ep {ep:>3}: {w}W/{l}L/{d}D {wr:.1%} {status}")

        # rolling avg win rates
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

    def save_training_history(self, file: str) -> None:
        if not os.path.exists(self.training_examples_folder):
            print(
                f"Making directory for training examples at: {self.training_examples_folder}"
            )
            os.makedirs(self.training_examples_folder)

        training_examples_path = os.path.join(self.training_examples_folder, file)
        with open(training_examples_path, "wb") as f:
            pickle.dump(self.training_history, f)

    def load_training_history(self) -> None:
        if not os.path.isdir(self.training_examples_folder):
            return

        latest = 0
        latest_training_examples = None
        for filename in os.listdir(self.training_examples_folder):
            f = os.path.join(self.training_examples_folder, filename)
            if os.path.isfile(f):
                try:
                    # training_examples_0001.pkl
                    i = int(filename.split(".")[0].split("_")[-1])
                except ValueError:
                    # any other training example file
                    continue
                if i >= latest:
                    latest = i
                    latest_training_examples = f

        if latest_training_examples:
            with open(latest_training_examples, "rb") as file:
                self.training_history = pickle.load(file)

    def load_latest_model(self) -> int:
        """
        Loads latest model and returns latest training episode if training stops for whatever reason.
        """
        latest = 0
        if not os.path.isdir(self.nn.model_folder):
            return latest

        latest_model = None
        for filename in os.listdir(self.nn.model_folder):
            f = os.path.join(self.nn.model_folder, filename)
            if os.path.isfile(f):
                try:
                    i = int(filename.split("_")[1])  # ep_0001_model.weights.h5
                except ValueError:
                    # best_model.weights.h5 or temp_model.weights.h5
                    continue
                if i >= latest:
                    latest = i
                    latest_model = f

        if latest_model:
            self.nn.load(latest_model)
            self.pn.load(latest_model)  # TODO: some other form of previous model?
        return latest
