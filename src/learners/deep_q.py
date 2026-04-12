import copy
import hashlib
import os
import pickle
import time
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
    state_tracker_size_bits: int
    state_tracker_num_hashes: int


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


class EvaluationResult(NamedTuple):
    score: float
    summary: str


# The unique-state tracker is only for observability. We intentionally use Bloom
# filters instead of exact sets so the persisted tracker stays bounded in size.
# The tradeoff is occasional false positives, which makes the counts approximate
# lower bounds on how many observable states we have seen.
class BloomFilter:
    def __init__(self, size_bits: int, num_hashes: int) -> None:
        self.size_bits = size_bits
        self.num_hashes = num_hashes
        self.bits = bytearray((size_bits + 7) // 8)

    @staticmethod
    def _digest(payload: bytes) -> tuple[int, int]:
        # Bloom filters need multiple bit indexes per item. We derive those
        # indexes from a single stable blake2b digest by splitting the 128-bit
        # digest into two 64-bit values and then using double hashing below.
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], byteorder="big", signed=False)
        h2 = int.from_bytes(digest[8:], byteorder="big", signed=False)
        if h2 == 0:
            # A zero step would collapse every derived Bloom-filter index onto
            # the same bit position. We swap in a fixed odd 64-bit constant
            # (the golden-ratio hashing constant) so persisted tracker files
            # stay deterministic while still spreading indexes out well.
            h2 = 0x9E3779B97F4A7C15
        return h1, h2

    def _indexes(self, payload: bytes) -> list[int]:
        h1, h2 = self._digest(payload)
        return [int((h1 + i * h2) % self.size_bits) for i in range(self.num_hashes)]

    def probably_contains(self, payload: bytes) -> bool:
        for index in self._indexes(payload):
            if not (self.bits[index // 8] & (1 << (index % 8))):
                return False
        return True

    def add(self, payload: bytes) -> None:
        for index in self._indexes(payload):
            self.bits[index // 8] |= 1 << (index % 8)

    def add_if_new(self, payload: bytes) -> bool:
        is_new = not self.probably_contains(payload)
        self.add(payload)
        return is_new


class UniqueStateTracker(Generic[Immutable]):
    def __init__(self, size_bits: int, num_hashes: int) -> None:
        self.size_bits = size_bits
        self.num_hashes = num_hashes
        self.states_seen = BloomFilter(size_bits=size_bits, num_hashes=num_hashes)
        self.states_seen_count = 0
        self.states_seen_by_ply: dict[int, BloomFilter] = {}
        self.states_seen_count_by_ply: dict[int, int] = {}

    # We use a stable digest of the game's immutable state rather than Python's
    # built-in hash() so tracker files can be reused across processes.
    @staticmethod
    def _payload(state: Immutable) -> bytes:
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    def observe(self, state: Immutable, ply: int) -> None:
        payload = self._payload(state)
        if self.states_seen.add_if_new(payload):
            self.states_seen_count += 1

        if ply not in self.states_seen_by_ply:
            self.states_seen_by_ply[ply] = BloomFilter(
                size_bits=self.size_bits, num_hashes=self.num_hashes
            )
            self.states_seen_count_by_ply[ply] = 0
        if self.states_seen_by_ply[ply].add_if_new(payload):
            self.states_seen_count_by_ply[ply] += 1

    def total_unique_states(self) -> int:
        return self.states_seen_count

    def unique_states_by_ply(self) -> dict[int, int]:
        return {
            ply: count for ply, count in sorted(self.states_seen_count_by_ply.items())
        }


class DeepQLearner(Generic[State, Immutable]):
    def __init__(
        self,
        game: Game[State, Immutable],
        nn: NeuralNetwork[State, DQNOutput],
        target_nn: NeuralNetwork[State, DQNOutput],
        params: DeepQParameters,
        training_artifacts_folder: str,
        evaluator: Callable[[], EvaluationResult] | None = None,
    ) -> None:
        self.game = game
        self.predict_nn = nn
        self.target_nn = target_nn
        self.replay_memory: Deque[Memory] = deque([], maxlen=params.memory_size)
        self.training_artifacts_folder = training_artifacts_folder

        self.training_episodes = params.training_episodes
        self.episodes_per_model_save = params.episodes_per_model_save
        self.episodes_per_memory_save = params.episodes_per_memory_save
        self.episodes_per_stats_print = params.episodes_per_stats_print
        self.episodes_per_evaluation = params.episodes_per_evaluation
        self.state_tracker_size_bits = params.state_tracker_size_bits
        self.state_tracker_num_hashes = params.state_tracker_num_hashes

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
        self.best_evaluation_score: float | None = None
        self.unique_state_tracker = UniqueStateTracker[Immutable](
            size_bits=self.state_tracker_size_bits,
            num_hashes=self.state_tracker_num_hashes,
        )

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

    def _record_state_visit(self, state: State, ply: int) -> None:
        self.unique_state_tracker.observe(self.game.to_immutable(state), ply)

    def unique_state_count(self) -> int:
        return self.unique_state_tracker.total_unique_states()

    def unique_state_counts_by_ply(self) -> dict[int, int]:
        return self.unique_state_tracker.unique_states_by_ply()

    @staticmethod
    def _format_novelty_rate_by_ply(
        counts: dict[int, int], previous_counts: dict[int, int], episodes: int
    ) -> str:
        parts = []
        for ply, count in counts.items():
            new_states = count - previous_counts.get(ply, 0)
            novelty_rate = 100 * new_states / episodes if episodes > 0 else 0.0
            parts.append(f"{ply}:{new_states}/{episodes} ({novelty_rate:.1f}%)")
        return "[" + ", ".join(parts) + "]"

    def calculate_epsilon(self, episode: int) -> float:
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
            -self.epsilon_decay * episode
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, rem = divmod(seconds, 60)
        if minutes < 60:
            return f"{int(minutes)}m {rem:.1f}s"
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours)}h {int(minutes)}m {rem:.1f}s"

    def train(self) -> None:
        self.load_replay_memory()
        latest_ep = self.load_latest_model()

        self.steps = 0
        self.epsilon = self.calculate_epsilon(latest_ep)
        training_start = time.perf_counter()
        last_report_time = training_start
        report_rewards: List[float] = []
        report_steps = 0
        last_invalid_actions = self.invalid_action_count
        last_shortterm_replays = self.shortterm_replay_calls
        last_longterm_replays = self.longterm_replay_calls
        last_target_syncs = self.target_syncs
        last_unique_states = self.unique_state_count()
        last_unique_states_by_ply = self.unique_state_counts_by_ply()
        for i in range(latest_ep + 1, self.training_episodes + 1):
            episode_stats = self.run_game_once()
            report_rewards.append(float(episode_stats.final_reward))
            report_steps += episode_stats.steps

            self.epsilon = self.calculate_epsilon(i)

            if i % self.episodes_per_model_save == 0:
                self.predict_nn.save(f"ep_{i:07d}_model.weights.h5")

            if i % self.episodes_per_memory_save == 0:
                self.save_replay_memory(f"ep_{i:07d}_replay_memory.pkl")

            if (
                self.episodes_per_stats_print > 0
                and i % self.episodes_per_stats_print == 0
            ):
                now = time.perf_counter()
                episodes_in_window = len(report_rewards)
                current_unique_states = self.unique_state_count()
                current_unique_states_by_ply = self.unique_state_counts_by_ply()
                print(
                    "Episode"
                    f" {i}: epsilon={self.epsilon:.4f},"
                    f" avg_final_reward={np.mean(report_rewards):.4f},"
                    f" avg_steps={report_steps / episodes_in_window:.2f},"
                    f" invalid_actions={self.invalid_action_count - last_invalid_actions},"
                    f" short_replays={self.shortterm_replay_calls - last_shortterm_replays},"
                    f" long_replays={self.longterm_replay_calls - last_longterm_replays},"
                    f" target_syncs={self.target_syncs - last_target_syncs},"
                    f" window_time={self._format_duration(now - last_report_time)},"
                    f" total_time={self._format_duration(now - training_start)},"
                    f" unique_states={current_unique_states}"
                    f"(+{current_unique_states - last_unique_states})"
                )
                print(
                    "  novelty_rate_by_ply="
                    + self._format_novelty_rate_by_ply(
                        current_unique_states_by_ply,
                        last_unique_states_by_ply,
                        episodes_in_window,
                    )
                )
                report_rewards = []
                report_steps = 0
                last_invalid_actions = self.invalid_action_count
                last_shortterm_replays = self.shortterm_replay_calls
                last_longterm_replays = self.longterm_replay_calls
                last_target_syncs = self.target_syncs
                last_unique_states = current_unique_states
                last_unique_states_by_ply = current_unique_states_by_ply
                last_report_time = now

            if (
                self.evaluator is not None
                and self.episodes_per_evaluation > 0
                and i % self.episodes_per_evaluation == 0
            ):
                evaluation_start = time.perf_counter()
                evaluation = self.evaluator()
                print(
                    f"Evaluation after episode {i}: {evaluation.summary}, "
                    f"eval_time={self._format_duration(time.perf_counter() - evaluation_start)}"
                )
                if (
                    self.best_evaluation_score is None
                    or evaluation.score > self.best_evaluation_score
                ):
                    self.best_evaluation_score = evaluation.score
                    self.predict_nn.save("best_model.weights.h5")
                    print("New best evaluation checkpoint:" f" {evaluation.score:.2f}")

    def run_game_once(self) -> EpisodeStats:
        self.game.reset()
        state = self.game.state()
        episode_steps = 0

        while not self.game.check_finished(state):
            self._record_state_visit(state, ply=episode_steps)
            self.steps += 1
            episode_steps += 1

            score = self.game.calculate_reward(state)
            state_snapshot = copy.deepcopy(state)

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
            self.replay_memory.append(mem)

            # replay memory
            if (
                self.steps_to_train_shortterm > 0
                and self.steps % self.steps_to_train_shortterm == 0
            ):
                self.shortterm_replay_calls += 1
                self.train_on_replay_minibatch(np.asarray([mem], dtype=object))

            if (
                self.steps_to_train_longterm > 0
                and self.steps % self.steps_to_train_longterm == 0
                and len(self.replay_memory) > self.minibatch_size
                and len(self.replay_memory) > self.min_replay_size
            ):
                self.longterm_replay_calls += 1
                minibatch = self.rng.choice(
                    np.asarray(self.replay_memory, dtype=object),
                    size=self.minibatch_size,
                    replace=False,
                )
                self.train_on_replay_minibatch(minibatch)

            # update target network weights
            if self.steps % self.steps_per_target_update == 0:
                self.target_nn.set_weights(self.predict_nn.get_weights())
                self.target_syncs += 1

            state = next_state

        return EpisodeStats(
            final_reward=self.game.calculate_reward(state),
            steps=episode_steps,
        )

    def train_on_replay_minibatch(self, minibatch: NDArray) -> None:
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

    @staticmethod
    def _unique_states_file(replay_memory_file: str) -> str:
        if replay_memory_file.endswith("_replay_memory.pkl"):
            return replay_memory_file.replace(
                "_replay_memory.pkl", "_unique_states.pkl"
            )
        raise ValueError(f"Unexpected replay memory filename: {replay_memory_file}")

    def save_replay_memory(self, replay_memory_file: str) -> None:
        if not os.path.exists(self.training_artifacts_folder):
            print(
                "Making directory for training artifacts at:"
                f" {self.training_artifacts_folder}"
            )
            os.makedirs(self.training_artifacts_folder)

        replay_memory_path = os.path.join(
            self.training_artifacts_folder, replay_memory_file
        )
        with open(replay_memory_path, "wb") as f:
            pickle.dump(self.replay_memory, f)
        self.save_unique_states(self._unique_states_file(replay_memory_file))

    def save_unique_states(self, unique_states_file: str) -> None:
        if not os.path.exists(self.training_artifacts_folder):
            print(
                "Making directory for training artifacts at:"
                f" {self.training_artifacts_folder}"
            )
            os.makedirs(self.training_artifacts_folder)

        unique_states_path = os.path.join(
            self.training_artifacts_folder, unique_states_file
        )
        with open(unique_states_path, "wb") as f:
            pickle.dump(self.unique_state_tracker, f)

    def load_replay_memory(self) -> None:
        if not os.path.isdir(self.training_artifacts_folder):
            return

        latest = 0
        latest_replay_memory = None
        for filename in os.listdir(self.training_artifacts_folder):
            if not filename.endswith("_replay_memory.pkl"):
                continue
            f = os.path.join(self.training_artifacts_folder, filename)
            if os.path.isfile(f):
                try:
                    i = int(filename.split("_")[1])
                except ValueError:
                    continue
                if i >= latest:
                    latest = i
                    latest_replay_memory = f

        if latest_replay_memory:
            print(f"Loading replay memory from: {latest_replay_memory}")
            with open(latest_replay_memory, "rb") as file:
                self.replay_memory = pickle.load(file)
            self.load_unique_states(
                self._unique_states_file(os.path.basename(latest_replay_memory))
            )

    def load_unique_states(self, unique_states_file: str) -> None:
        unique_states_path = os.path.join(
            self.training_artifacts_folder, unique_states_file
        )
        if not os.path.isfile(unique_states_path):
            print(
                "No saved unique-state tracker found for the latest replay memory;"
                " novelty tracking will restart for this session."
            )
            return

        print(f"Loading unique-state tracker from: {unique_states_path}")
        with open(unique_states_path, "rb") as file:
            self.unique_state_tracker = pickle.load(file)
