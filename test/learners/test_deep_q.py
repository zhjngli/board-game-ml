from typing import List

import numpy as np

from games.game import INVAL, P1, VALID, ActionStatus, BasicState, Game
from learners.deep_q import DeepQLearner, DeepQParameters, DQNOutput
from nn.neural_network import NeuralNetwork


class DummyState(BasicState):
    def __init__(self, board: np.ndarray, tag: str, finished: bool = False) -> None:
        super().__init__(board=board, player=P1)
        self.tag = tag
        self.finished = finished


class DummyNetwork(NeuralNetwork[DummyState, DQNOutput]):
    def __init__(self, outputs: dict[str, DQNOutput]) -> None:
        super().__init__(model_folder="")
        self.outputs = outputs
        self.training_batches: List[List[tuple[DummyState, DQNOutput]]] = []

    def train(self, data: List[tuple[DummyState, DQNOutput]]) -> None:
        self.training_batches.append(data)

    def predict(self, inputs: List[DummyState]) -> List[DQNOutput]:
        preds: List[DQNOutput] = []
        for state in inputs:
            out = self.outputs[state.tag]
            preds.append(
                DQNOutput(policy=np.array(out.policy, dtype=float), value=out.value)
            )
        return preds

    def save(self, file: str) -> None:
        pass

    def load(self, file: str) -> None:
        pass

    def set_weights(self, weights) -> None:
        pass

    def get_weights(self):
        return []

    def summary(self) -> None:
        pass


class SingleStepMaskingGame(Game[DummyState, str]):
    def __init__(self) -> None:
        self._state = DummyState(board=np.array([0, 1]), tag="start")

    def reset(self) -> None:
        self._state = DummyState(board=np.array([0, 1]), tag="start")

    def state(self) -> DummyState:
        return self._state

    @staticmethod
    def to_immutable(state: DummyState) -> str:
        return state.tag

    def num_actions(self) -> int:
        return 2

    @staticmethod
    def actions(state: DummyState) -> List[ActionStatus]:
        return [VALID, INVAL]

    @staticmethod
    def apply(state: DummyState, action: int) -> DummyState:
        if action != 0:
            raise ValueError("invalid action")
        return DummyState(board=np.array([1, 1]), tag="terminal", finished=True)

    @staticmethod
    def check_finished(state: DummyState) -> bool:
        return state.finished

    @staticmethod
    def calculate_reward(state: DummyState) -> float:
        return 0.0

    @staticmethod
    def orient_state(state: DummyState) -> DummyState:
        return state

    @staticmethod
    def symmetries_of(a: np.ndarray) -> List[np.ndarray]:
        return [a]


class InPlaceMutationGame(Game[DummyState, str]):
    def __init__(self) -> None:
        self._state = DummyState(board=np.array([0]), tag="start")

    def reset(self) -> None:
        self._state = DummyState(board=np.array([0]), tag="start")

    def state(self) -> DummyState:
        return self._state

    @staticmethod
    def to_immutable(state: DummyState) -> str:
        return state.tag

    def num_actions(self) -> int:
        return 1

    @staticmethod
    def actions(state: DummyState) -> List[ActionStatus]:
        return [VALID]

    @staticmethod
    def apply(state: DummyState, action: int) -> DummyState:
        state.board[0] = 99
        state.tag = "terminal"
        state.finished = True
        return state

    @staticmethod
    def check_finished(state: DummyState) -> bool:
        return state.finished

    @staticmethod
    def calculate_reward(state: DummyState) -> float:
        return float(state.board[0])

    @staticmethod
    def orient_state(state: DummyState) -> DummyState:
        return state

    @staticmethod
    def symmetries_of(a: np.ndarray) -> List[np.ndarray]:
        return [a]


def build_params() -> DeepQParameters:
    return DeepQParameters(
        alpha=1.0,
        gamma=1.0,
        min_epsilon=0.0,
        max_epsilon=0.0,
        epsilon_decay=0.0,
        valid_action_reward=0.0,
        memory_size=10,
        min_replay_size=0,
        minibatch_size=1,
        steps_to_train_longterm=0,
        steps_to_train_shortterm=0,
        steps_per_target_update=100,
        training_episodes=1,
        episodes_per_model_save=1,
        episodes_per_memory_save=1,
        episodes_per_stats_print=0,
        episodes_per_evaluation=0,
    )


def test_run_game_once_masks_invalid_greedy_action() -> None:
    game = SingleStepMaskingGame()
    predict_nn = DummyNetwork(
        outputs={
            "start": DQNOutput(policy=np.array([1.0, 100.0]), value=0.0),
            "terminal": DQNOutput(policy=np.array([0.0, 0.0]), value=0.0),
        }
    )
    learner = DeepQLearner(
        game=game,
        nn=predict_nn,
        target_nn=DummyNetwork(outputs={}),
        params=build_params(),
        memory_folder="",
    )

    learner.run_game_once()

    assert learner.memory[0][1] == 0


def test_run_game_once_masks_invalid_random_action() -> None:
    game = SingleStepMaskingGame()
    predict_nn = DummyNetwork(
        outputs={
            "start": DQNOutput(policy=np.array([1.0, 100.0]), value=0.0),
            "terminal": DQNOutput(policy=np.array([0.0, 0.0]), value=0.0),
        }
    )
    params = build_params()._replace(min_epsilon=1.0, max_epsilon=1.0)
    learner = DeepQLearner(
        game=game,
        nn=predict_nn,
        target_nn=DummyNetwork(outputs={}),
        params=params,
        memory_folder="",
    )
    learner.epsilon = 1.0

    learner.run_game_once()

    assert learner.memory[0][1] == 0


def test_replay_memory_masks_invalid_next_actions() -> None:
    predict_nn = DummyNetwork(
        outputs={"start": DQNOutput(policy=np.array([0.0, 0.0]), value=0.0)}
    )
    target_nn = DummyNetwork(
        outputs={"next": DQNOutput(policy=np.array([1.0, 100.0]), value=0.0)}
    )
    learner = DeepQLearner(
        game=SingleStepMaskingGame(),
        nn=predict_nn,
        target_nn=target_nn,
        params=build_params(),
        memory_folder="",
    )

    state = DummyState(board=np.array([0, 1]), tag="start")
    next_state = DummyState(board=np.array([0, 1]), tag="next")
    minibatch = np.asarray([(state, 0, next_state, 0.0, False)], dtype=object)

    learner.replay_memory(minibatch)

    trained_output = predict_nn.training_batches[0][0][1]
    assert trained_output.policy[0] == 1.0


def test_run_game_once_stores_frozen_state_snapshots() -> None:
    predict_nn = DummyNetwork(
        outputs={
            "start": DQNOutput(policy=np.array([1.0]), value=0.0),
            "terminal": DQNOutput(policy=np.array([0.0]), value=0.0),
        }
    )
    learner = DeepQLearner(
        game=InPlaceMutationGame(),
        nn=predict_nn,
        target_nn=DummyNetwork(outputs={}),
        params=build_params(),
        memory_folder="",
    )

    learner.run_game_once()

    state, _, next_state, _, _ = learner.memory[0]
    assert int(state.board[0]) == 0
    assert int(next_state.board[0]) == 99


def test_train_calls_evaluator_on_schedule() -> None:
    game = SingleStepMaskingGame()
    predict_nn = DummyNetwork(
        outputs={
            "start": DQNOutput(policy=np.array([1.0, 100.0]), value=0.0),
            "terminal": DQNOutput(policy=np.array([0.0, 0.0]), value=0.0),
        }
    )
    params = build_params()._replace(
        training_episodes=2,
        episodes_per_model_save=99,
        episodes_per_memory_save=99,
        episodes_per_evaluation=1,
    )
    evaluation_calls: List[str] = []

    def evaluator() -> str:
        evaluation_calls.append("called")
        return "ok"

    learner = DeepQLearner(
        game=game,
        nn=predict_nn,
        target_nn=DummyNetwork(outputs={}),
        params=params,
        memory_folder="",
        evaluator=evaluator,
    )

    learner.train()

    assert evaluation_calls == ["called", "called"]
