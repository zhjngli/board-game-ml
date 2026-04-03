import copy
import os
import pickle
from collections import deque
from typing import Callable, Deque, Generic, List, NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray

from games.game import VALID, Action, Game, Immutable, State
from nn.neural_network import NeuralNetwork


class DeepQParameters(NamedTuple):
    alpha: float
    gamma: float
    min_epsilon: float
    max_epsilon: float
    epsilon_decay: float
    valid_action_reward: float
    memory_size: int
    min_replay_size: int
    minibatch_size: int
    steps_to_train_longterm: int
    steps_to_train_shortterm: int
    steps_per_target_update: int
    training_episodes: int
    episodes_per_model_save: int
    episodes_per_memory_save: int
    episodes_per_stats_print: int
    episodes_per_evaluation: int


Policy = NDArray  # TODO: one dimensional NDArray of arbitrary length
Value = float


class DQNOutput(NamedTuple):
    policy: Policy
    value: Value


Reward = float
Memory = Tuple[State, Action, State, Reward, bool]


class EpisodeStats(NamedTuple):
    final_reward: Reward
    steps: int


class DeepQLearner(Generic[State, Immutable]):
    def __init__(
        self,
        game: Game[State, Immutable],
        nn: NeuralNetwork[State, DQNOutput],
        target_nn: NeuralNetwork[State, DQNOutput],
        params: DeepQParameters,
        memory_folder: str,
        evaluator: Callable[[], str] | None = None,
    ) -> None:
        self.game = game
        self.predict_nn = nn
        self.target_nn = target_nn
        self.memory: Deque[Memory] = deque([], maxlen=params.memory_size)
        self.memory_folder = memory_folder

        self.training_episodes = params.training_episodes
        self.episodes_per_model_save = params.episodes_per_model_save
        self.episodes_per_memory_save = params.episodes_per_memory_save
        self.episodes_per_stats_print = params.episodes_per_stats_print
        self.episodes_per_evaluation = params.episodes_per_evaluation

        self.min_replay_size = params.min_replay_size
        self.minibatch_size = params.minibatch_size

        self.steps_to_train_longterm = params.steps_to_train_longterm
        self.steps_to_train_shortterm = params.steps_to_train_shortterm
        self.steps_per_target_update = params.steps_per_target_update

        self.alpha = params.alpha
        self.gamma = params.gamma
        self.min_epsilon = params.min_epsilon
        self.max_epsilon = params.max_epsilon
        self.epsilon_decay = params.epsilon_decay
        self.valid_action_reward = params.valid_action_reward

        self.rng = np.random.default_rng()
        self.steps = 0
        self.epsilon = self.max_epsilon
        self.invalid_action_count = 0
        self.shortterm_replay_calls = 0
        self.longterm_replay_calls = 0
        self.target_syncs = 0
        self.evaluator = evaluator

    def _valid_actions(self, state: State) -> NDArray[np.int_]:
        action_statuses = np.asarray(self.game.actions(state))
        return np.flatnonzero(action_statuses == VALID)

    def _freeze_transition(
        self,
        state: State,
        action: Action,
        next_state: State,
        reward: Reward,
        done: bool,
    ) -> Memory:
        return (
            state,
            action,
            copy.deepcopy(next_state),
            reward,
            done,
        )

    def calculate_epsilon(self, episode: int) -> float:
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
            -self.epsilon_decay * episode
        )

    def train(self) -> None:
        self.load_memory()
        latest_ep = self.load_latest_model()

        self.steps = 0
        self.epsilon = self.calculate_epsilon(latest_ep)
        report_rewards: List[float] = []
        report_steps = 0
        last_invalid_actions = self.invalid_action_count
        last_shortterm_replays = self.shortterm_replay_calls
        last_longterm_replays = self.longterm_replay_calls
        last_target_syncs = self.target_syncs
        for i in range(latest_ep + 1, self.training_episodes + 1):
            episode_stats = self.run_game_once()
            report_rewards.append(float(episode_stats.final_reward))
            report_steps += episode_stats.steps

            self.epsilon = self.calculate_epsilon(i)

            if i % self.episodes_per_model_save == 0:
                self.predict_nn.save(f"ep_{i:07d}_model.weights.h5")

            if i % self.episodes_per_memory_save == 0:
                self.save_memory(f"ep_{i:07d}_memory.pkl")

            if (
                self.episodes_per_stats_print > 0
                and i % self.episodes_per_stats_print == 0
            ):
                print(
                    "Episode"
                    f" {i}: epsilon={self.epsilon:.4f},"
                    f" avg_final_reward={np.mean(report_rewards):.4f},"
                    f" avg_steps={report_steps / len(report_rewards):.2f},"
                    f" invalid_actions={self.invalid_action_count - last_invalid_actions},"
                    f" short_replays={self.shortterm_replay_calls - last_shortterm_replays},"
                    f" long_replays={self.longterm_replay_calls - last_longterm_replays},"
                    f" target_syncs={self.target_syncs - last_target_syncs}"
                )
                report_rewards = []
                report_steps = 0
                last_invalid_actions = self.invalid_action_count
                last_shortterm_replays = self.shortterm_replay_calls
                last_longterm_replays = self.longterm_replay_calls
                last_target_syncs = self.target_syncs

            # TODO: track efficacy of learning (e.g. play some number of games and track score)
            if (
                self.evaluator is not None
                and self.episodes_per_evaluation > 0
                and i % self.episodes_per_evaluation == 0
            ):
                print(f"Evaluation after episode {i}: {self.evaluator()}")

    def run_game_once(self) -> EpisodeStats:
        self.game.reset()
        state = self.game.state()
        episode_steps = 0

        while not self.game.check_finished(state):
            self.steps += 1
            episode_steps += 1

            score = self.game.calculate_reward(state)
            state_snapshot = copy.deepcopy(state)
            # print(f"state:\n{state.board}")
            # print(f"next: {state.next}")  # type: ignore

            # epsilon greedy over legal actions only
            valid_actions = self._valid_actions(state)
            if len(valid_actions) == 0:
                raise ValueError(
                    "Cannot choose an action when no valid actions are available"
                )

            if np.random.sample() < self.epsilon:
                a = int(np.random.choice(valid_actions))
            else:
                dqn_out: DQNOutput = self.predict_nn.predict([state])[0]
                valid_qs = dqn_out.policy[valid_actions]
                a = int(valid_actions[int(np.argmax(valid_qs))])

            # calculations based on action chosen
            # TODO: very sparse rewards, only at game end
            try:
                next_state = self.game.apply(state, a)
                new_score = self.game.calculate_reward(next_state)
                reward = new_score - score + self.valid_action_reward
                game_end = self.game.check_finished(next_state)
            except ValueError:
                # punish invalid actions
                next_state = copy.deepcopy(state)
                reward = -1
                game_end = False
                self.invalid_action_count += 1

            mem = self._freeze_transition(
                state_snapshot, a, next_state, reward, game_end
            )
            self.memory.append(mem)

            # replay memory
            if (
                self.steps_to_train_shortterm > 0
                and self.steps % self.steps_to_train_shortterm == 0
            ):
                self.shortterm_replay_calls += 1
                self.replay_memory(np.asarray([mem], dtype=object))

            if (
                self.steps_to_train_longterm > 0
                and self.steps % self.steps_to_train_longterm == 0
                and len(self.memory) > self.minibatch_size
                and len(self.memory) > self.min_replay_size
            ):
                self.longterm_replay_calls += 1
                minibatch = self.rng.choice(
                    np.asarray(self.memory, dtype=object),
                    size=self.minibatch_size,
                    replace=False,
                )
                self.replay_memory(minibatch)

            # update target network weights
            if self.steps % self.steps_per_target_update == 0:
                self.target_nn.set_weights(self.predict_nn.get_weights())
                self.target_syncs += 1

            state = next_state

        return EpisodeStats(
            final_reward=self.game.calculate_reward(state),
            steps=episode_steps,
        )

    def replay_memory(self, minibatch: NDArray) -> None:
        # minibatch is an array converted from: List[Tuple[State, Action, State, Reward, bool]]
        states = minibatch[:, 0]
        actions = minibatch[:, 1].astype(Action)
        next_states = minibatch[:, 2]
        rewards = minibatch[:, 3]
        game_ends = minibatch[:, 4]

        dqn_outs: List[DQNOutput] = self.predict_nn.predict(list(states))
        next_dqn_outs: List[DQNOutput] = self.target_nn.predict(list(next_states))

        max_next_qs = np.asarray(rewards, dtype=float)
        for i, next_state in enumerate(next_states):
            if bool(game_ends[i]):
                continue

            valid_actions = self._valid_actions(next_state)
            if len(valid_actions) == 0:
                continue

            next_policy = next_dqn_outs[i].policy[valid_actions]
            max_next_qs[i] = float(rewards[i]) + self.gamma * float(np.max(next_policy))

        # np.arange(len(qs)) instead of `:`?
        for i in range(len(dqn_outs)):
            dqn_outs[i].policy[actions[i]] = (1 - self.alpha) * dqn_outs[i].policy[
                actions[i]
            ] + self.alpha * max_next_qs[i]

        # TODO: make train type signature flexible so i don't have to convert to list?
        successful_train = False
        while not successful_train:
            try:
                self.predict_nn.train(list(zip(states, dqn_outs)))
                successful_train = True
            except Exception as e:
                print(f"Failed to train with error {e}, retrying...")

    def load_latest_model(self) -> int:
        """
        Loads latest model and returns latest training episode if training stops for whatever reason.
        """
        latest = 0
        if not os.path.isdir(self.predict_nn.model_folder):
            return latest

        latest_model = None
        for filename in os.listdir(self.predict_nn.model_folder):
            f = os.path.join(self.predict_nn.model_folder, filename)
            if os.path.isfile(f):
                try:
                    i = int(filename.split("_")[1])  # ep_0001_model.h5
                except ValueError:
                    # any other model
                    continue
                if i >= latest:
                    latest = i
                    latest_model = f

        if latest_model:
            print(f"Loading latest model from: {latest_model}")
            self.predict_nn.load(latest_model)
            self.target_nn.load(latest_model)
        return latest

    def save_memory(self, memory_file: str) -> None:
        if not os.path.exists(self.memory_folder):
            print(f"Making directory for play memory at: {self.memory_folder}")
            os.makedirs(self.memory_folder)

        memory_path = os.path.join(self.memory_folder, memory_file)
        with open(memory_path, "wb") as f:
            pickle.dump(self.memory, f)

    def load_memory(self) -> None:
        if not os.path.isdir(self.memory_folder):
            return

        latest = 0
        latest_memory = None
        for filename in os.listdir(self.memory_folder):
            f = os.path.join(self.memory_folder, filename)
            if os.path.isfile(f):
                try:
                    i = int(filename.split("_")[1])  # ep_0001_memory.pkl
                except ValueError:
                    # any other memory
                    continue
                if i >= latest:
                    latest = i
                    latest_memory = f

        if latest_memory:
            print(f"Loading replay memory from: {latest_memory}")
            with open(latest_memory, "rb") as file:
                self.memory = pickle.load(file)
