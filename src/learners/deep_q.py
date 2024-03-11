import os
from collections import deque
from typing import Deque, Generic, NamedTuple, Tuple

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
    memory_size: int
    min_replay_size: int
    minibatch_size: int
    training_episodes: int
    steps_to_train_longterm: int
    steps_to_train_shortterm: int
    steps_per_target_update: int


Policy = NDArray  # TODO: one dimensional NDArray of arbitrary length
Reward = float


class DeepQLearner(Generic[State, Immutable]):
    def __init__(
        self,
        game: Game[State, Immutable],
        nn: NeuralNetwork[State, Policy],
        params: DeepQParameters,
    ) -> None:
        self.game = game
        self.predict_nn = nn
        self.target_nn = self.predict_nn.__class__(self.predict_nn.model_folder)
        self.memory: Deque[Tuple[State, Action, State, Reward, bool]] = deque(
            [], maxlen=params.memory_size
        )

        self.min_replay_size = params.min_replay_size
        self.minibatch_size = params.minibatch_size
        self.training_episodes = params.training_episodes
        self.steps_to_train_longterm = params.steps_to_train_longterm
        self.steps_to_train_shortterm = params.steps_to_train_shortterm
        self.steps_per_target_update = params.steps_per_target_update

        self.alpha = params.alpha
        self.gamma = params.gamma
        self.min_epsilon = params.min_epsilon
        self.max_epsilon = params.max_epsilon
        self.epsilon_decay = params.epsilon_decay

        self.rng = np.random.default_rng()

    def train(self) -> None:
        latest_ep = self.load_latest_model()
        steps = 0
        epsilon = self.max_epsilon
        for i in range(latest_ep + 1, self.training_episodes + 1):
            self.game.reset()
            state = self.game.state()

            while not self.game.check_finished(state):
                steps += 1

                score = self.game.calculate_reward(state)

                action_statuses = np.asarray(self.game.actions(state))
                if np.random.sample() < epsilon:
                    valid_actions = np.where(action_statuses == VALID)[0]
                    a = np.random.choice(valid_actions)
                else:
                    pi = self.predict_nn.predict([state])[0]
                    # get the maximum prediction of valid actions
                    a = np.argmax(pi * action_statuses)

                try:
                    next_state = self.game.apply(state, a)
                except ValueError:
                    # if there's an error for whatever reason
                    # e.g. policy not robust enough, so when masked with action statuses it produces no valid actions
                    # then choose a random valid action
                    valid_actions = np.where(action_statuses == VALID)[0]
                    a = np.random.choice(valid_actions)
                    next_state = self.game.apply(state, a)

                new_score = self.game.calculate_reward(next_state)
                game_end = self.game.check_finished(next_state)

                reward = new_score - score
                mem = (state, a, next_state, reward, game_end)
                self.memory.append(mem)

                # fit models
                if steps % self.steps_to_train_shortterm == 0:
                    self.fit_nn(np.array([mem]))

                if (
                    steps % self.steps_to_train_longterm == 0
                    and len(self.memory) > self.minibatch_size
                    and len(self.memory) > self.min_replay_size
                ):
                    minibatch = self.rng.choice(
                        np.array(self.memory), size=self.minibatch_size, replace=False
                    )
                    self.fit_nn(minibatch)

                # update target network weights
                if steps % self.steps_per_target_update == 0:
                    self.target_nn.set_weights(self.predict_nn.get_weights())

                self.predict_nn.save(f"ep_{i:07d}_model.h5")
                state = next_state

            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
                -self.epsilon_decay * i
            )
            # TODO: track efficacy of learning (e.g. play some number of games and track score)

    def fit_nn(self, minibatch: NDArray) -> None:
        # minibatch is an array converted from: List[Tuple[State, Action, State, Reward, bool]]
        states = minibatch[:, 0]
        actions = minibatch[:, 1].astype(Action)
        next_states = minibatch[:, 2]
        rewards = minibatch[:, 3]
        game_ends = minibatch[:, 4]

        qs = self.predict_nn.predict(list(states))
        next_qs = self.target_nn.predict(list(next_states))

        max_next_qs = np.where(
            game_ends, rewards, rewards + self.gamma * np.max(next_qs, axis=1)
        )
        # np.arange(len(qs)) instead of `:`?
        for i in range(len(qs)):
            qs[i][actions[i]] = (1 - self.alpha) * qs[i][
                actions[i]
            ] + self.alpha * max_next_qs[i]

        # TODO: make train type signature flexible so i don't have to convert to list?
        self.predict_nn.train(list(zip(states, qs)))

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
            self.predict_nn.load(latest_model)
            self.target_nn.load(latest_model)
        return latest
