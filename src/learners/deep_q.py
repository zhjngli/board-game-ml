from collections import deque
from typing import Deque, Generic, NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray

from games.game import Action, Game, Immutable, State
from nn.neural_network import NeuralNetwork


class DeepQParameters(NamedTuple):
    alpha: float
    gamma: float
    epsilon: float
    memory_size: int
    training_episodes: int
    steps_to_train_longterm: int
    steps_to_train_shortterm: int
    steps_per_target_update: int
    minibatch_num: int


class DeepQInput(NamedTuple, Generic[State]):
    state: State
    action: Action
    next_state: State
    reward: float


Policy = NDArray  # TODO: one dimensional NDArray of arbitrary length
Reward = float


class DeepQOutput(NamedTuple):
    policy: Policy


class DeepQLearner(Generic[State, Immutable]):
    def __init__(
        self,
        game: Game[State, Immutable],
        nn: NeuralNetwork[State, State, Policy],
        params: DeepQParameters,
    ) -> None:
        self.game = game
        self.predict_nn = nn
        self.target_nn = self.predict_nn.__class__(self.predict_nn.model_folder)
        self.memory: Deque[Tuple[State, Action, State, Reward, bool]] = deque(
            [], maxlen=params.memory_size
        )

        self.training_episodes = params.training_episodes
        self.steps_to_train_longterm = params.steps_to_train_longterm
        self.steps_to_train_shortterm = params.steps_to_train_shortterm
        self.steps_per_target_update = params.steps_per_target_update
        self.minibatch_num = params.minibatch_num

        self.alpha = params.alpha
        self.gamma = params.gamma
        self.epsilon = params.epsilon

        self.rng = np.random.default_rng()

    def train(self) -> None:
        steps = 0
        for i in range(1, self.training_episodes + 1):
            self.game.reset()
            state = self.game.state()

            while not self.game.check_finished(state):
                steps += 1

                score = self.game.calculate_reward(state)

                if np.random.sample() < self.epsilon:
                    action_statuses = self.game.actions(state)
                    valid_actions = np.where(action_statuses == 1)[0]
                    a = np.random.choice(valid_actions)
                else:
                    pi = self.predict_nn.predict([state])[0]
                    a = np.argmax(pi)

                next_state = self.game.apply(state, a)
                new_score = self.game.calculate_reward(next_state)
                game_end = self.game.check_finished(next_state)

                reward = new_score - score
                mem = (state, a, next_state, reward, game_end)
                self.memory.append(mem)

                # fit models
                if steps % self.steps_to_train_shortterm == 0:
                    self.fit_nn(np.array([mem]))

                # TODO: len(self.memory) > some other hyperparameter?
                if (
                    steps % self.steps_to_train_longterm == 0
                    and len(self.memory) > self.minibatch_num
                ):
                    minibatch = self.rng.choice(
                        np.array(self.memory), size=self.minibatch_num, replace=False
                    )
                    self.fit_nn(minibatch)

                # update target network weights
                if steps % self.steps_per_target_update == 0:
                    self.target_nn.set_weights(self.predict_nn.get_weights())

                self.predict_nn.save(f"ep_{i:07d}_model.h5")
                state = next_state

            # TODO: epsilon decay

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
