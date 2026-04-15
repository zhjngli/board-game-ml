import pathlib
from typing import NamedTuple

import numpy as np
from keras.models import Model  # type: ignore

from games.digit_party.game import DigitParty, DigitPartyPlacement, DigitPartyState
from games.digit_party.run_helpers import computer_game
from games.game import VALID
from learners.deep_q import DeepQLearner, DeepQParameters, DQNOutput, EvaluationResult
from nn.neural_network import NeuralNetwork


class DigitPartyEvaluation(NamedTuple):
    games: int
    average_pct: float
    aggregate_pct: float
    average_score: float
    average_theoretical_max: float


class BaseDigitPartyDeepQNN(NeuralNetwork[DigitPartyState, DQNOutput]):
    def __init__(
        self,
        model_folder: str,
        batch_size: int,
        epochs: int,
        normalize_scale: float = 1.0,
    ) -> None:
        super().__init__(model_folder)
        self.batch_size = batch_size
        self.epochs = epochs
        self.normalize_scale = normalize_scale
        self.model: Model

    def _model_inputs(
        self, inputs: list[DigitPartyState]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_boards = np.asarray(
            [input.board / self.normalize_scale for input in inputs], dtype=float
        )
        input_currs = np.asarray(
            [
                (input.next[0] if input.next[0] is not None else 0)
                / self.normalize_scale
                for input in inputs
            ],
            dtype=float,
        )
        input_nexts = np.asarray(
            [
                (input.next[1] if input.next[1] is not None else 0)
                / self.normalize_scale
                for input in inputs
            ],
            dtype=float,
        )
        return input_boards, input_currs, input_nexts

    def train(self, data: list[tuple[DigitPartyState, DQNOutput]]) -> None:
        inputs: list[DigitPartyState]
        outputs: list[DQNOutput]
        inputs, outputs = list(zip(*data))
        input_boards, input_currs, input_nexts = self._model_inputs(list(inputs))
        target_qs = np.asarray([output.policy for output in outputs])
        self.model.fit(
            x=[input_boards, input_currs, input_nexts],
            y=target_qs,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
            verbose=0,
        )

    def predict(self, inputs: list[DigitPartyState]) -> list[DQNOutput]:
        input_boards, input_currs, input_nexts = self._model_inputs(inputs)
        qs = self.model.predict([input_boards, input_currs, input_nexts], verbose=0)
        return [DQNOutput(policy=q, value=0.0) for q in qs]

    def save(self, file: str) -> None:
        model_path = pathlib.Path(self.model_folder)
        if not model_path.exists():
            print(f"Making directory for models at: {self.model_folder}")
            model_path.mkdir(parents=True)
        self.model.save_weights(model_path / file)

    def load(self, file: str) -> None:
        self.model.load_weights(pathlib.Path(self.model_folder) / file)

    def set_weights(self, weights) -> None:
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def summary(self) -> None:
        self.model.summary()


def greedy_digit_party_move(
    nn: NeuralNetwork[DigitPartyState, DQNOutput], state: DigitPartyState
) -> DigitPartyPlacement:
    out = nn.predict([state])[0]
    valid_actions = np.flatnonzero(np.asarray(DigitParty.actions(state)) == VALID)
    if len(valid_actions) == 0:
        raise ValueError("No valid actions are available for Digit Party evaluation")

    action = int(valid_actions[int(np.argmax(out.policy[valid_actions]))])
    n = state.board.shape[0]
    return int(action / n), int(action % n)


def evaluate_digit_party(
    nn: NeuralNetwork[DigitPartyState, DQNOutput], games: int, n: int
) -> DigitPartyEvaluation:
    game = DigitParty(n=n)
    total_score = 0.0
    total_theoretical_max = 0.0
    total_pct = 0.0

    for _ in range(games):
        game.reset()
        while not game.is_finished():
            r, c = greedy_digit_party_move(nn, game.state())
            game.place(r, c)

        score = float(game.score)
        theoretical_max = float(game.theoretical_max_score())
        total_score += score
        total_theoretical_max += theoretical_max
        total_pct += score / theoretical_max if theoretical_max > 0 else 0.0

    return DigitPartyEvaluation(
        games=games,
        average_pct=100 * total_pct / games,
        aggregate_pct=100 * total_score / total_theoretical_max,
        average_score=total_score / games,
        average_theoretical_max=total_theoretical_max / games,
    )


def digit_party_evaluation_summary(
    nn: NeuralNetwork[DigitPartyState, DQNOutput], games: int, n: int
) -> EvaluationResult:
    evaluation = evaluate_digit_party(nn=nn, games=games, n=n)
    return EvaluationResult(
        score=evaluation.average_pct,
        summary=(
            f"{evaluation.games} games, "
            f"avg_pct={evaluation.average_pct:.2f}%, "
            f"aggregate_pct={evaluation.aggregate_pct:.2f}%, "
            f"avg_score={evaluation.average_score:.2f}/"
            f"{evaluation.average_theoretical_max:.2f}"
        ),
    )


def train_digit_party_dqn(
    n: int,
    nn: NeuralNetwork[DigitPartyState, DQNOutput],
    target_nn: NeuralNetwork[DigitPartyState, DQNOutput],
    params: DeepQParameters,
    training_artifacts_folder: str,
    eval_games: int,
    final_eval_games: int,
    episodes_override: int | None = None,
) -> None:
    if episodes_override is not None:
        params = params._replace(training_episodes=episodes_override)

    nn.summary()
    deepq = DeepQLearner(
        DigitParty(n=n),
        nn,
        target_nn,
        params,
        training_artifacts_folder=training_artifacts_folder,
        evaluator=lambda: digit_party_evaluation_summary(nn=nn, games=eval_games, n=n),
    )
    deepq.train()

    print(
        "Final evaluation: "
        + digit_party_evaluation_summary(nn=nn, games=final_eval_games, n=n).summary
    )


def eval_best_digit_party_dqn(
    n: int,
    nn: NeuralNetwork[DigitPartyState, DQNOutput],
    best_model_file: str,
    games: int,
) -> None:
    if not pathlib.Path(best_model_file).exists():
        raise FileNotFoundError(
            f"Could not find trained best model at {best_model_file}"
        )

    nn.load(best_model_file)
    print(f"Loaded best model from: {best_model_file}")
    computer_game(
        DigitParty(n=n),
        games,
        lambda state: greedy_digit_party_move(nn, state),
    )
