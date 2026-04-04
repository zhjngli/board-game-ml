import pathlib
from typing import NamedTuple

import numpy as np
from keras.layers import (  # type: ignore
    Concatenate,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Reshape,
)
from keras.models import Model  # type: ignore
from keras.optimizers import Adam  # type: ignore

from games.digit_party.game import DigitParty, DigitPartyPlacement, DigitPartyState
from games.digit_party.train_deep import DP3NNParams
from games.game import VALID
from learners.deep_q import DeepQLearner, DeepQParameters, DQNOutput, EvaluationResult
from nn.neural_network import NeuralNetwork

"""Trains a 3x3 Digit Party network with a DQN-focused architecture baseline."""

# Temporary DQN baseline.
DQN_3X3_NN_PARAMS = DP3NNParams(
    conv_layers=8,
    conv_filters=14,
    dense_layers=1,
    dense_units=337,
    learning_rate=0.0009647204266707786,
    batch_size=64,
    epochs=1,
    dropout_rate=0.0,
    output_activation="linear",
)

TRAINING_EVALUATION_GAMES = 500
TRAINING_EVALUATION_INTERVAL = 2500
FINAL_EVALUATION_GAMES = 1000


class DigitPartyEvaluation(NamedTuple):
    games: int
    average_pct: float
    aggregate_pct: float
    average_score: float
    average_theoretical_max: float


class DigitParty3x3DeepQNN(NeuralNetwork[DigitPartyState, DQNOutput]):
    def __init__(self, params: DP3NNParams, model_folder: str) -> None:
        super().__init__(model_folder)
        self.params = params

        input_board = Input(shape=(3, 3), name="dp_3x3_board")
        input_curr_digit = Input(shape=(1,), name="current_digit")
        input_next_digit = Input(shape=(1,), name="next_digit")

        board = Reshape((3, 3, 1))(input_board)
        prev = board

        for _ in range(self.params.conv_layers):
            prev = Conv2D(
                filters=self.params.conv_filters,
                kernel_size=(2, 2),
                padding="same",
                activation="relu",
            )(prev)

        flat = Flatten()(prev)
        prev = Concatenate()([flat, input_curr_digit, input_next_digit])

        for _ in range(self.params.dense_layers):
            prev = Dense(self.params.dense_units, activation="relu")(prev)

        q_values = Dense(9, activation=self.params.output_activation, name="q")(prev)

        self.model = Model(
            inputs=[input_board, input_curr_digit, input_next_digit], outputs=q_values
        )
        self.model.compile(
            loss="mean_squared_error",
            optimizer=Adam(learning_rate=self.params.learning_rate),
        )

    def train(self, data: list[tuple[DigitPartyState, DQNOutput]]) -> None:
        inputs: list[DigitPartyState]
        outputs: list[DQNOutput]
        inputs, outputs = list(zip(*data))
        input_boards = np.asarray([input.board for input in inputs])
        input_currs = np.asarray(
            [input.next[0] if input.next[0] is not None else 0 for input in inputs]
        )
        input_nexts = np.asarray(
            [input.next[1] if input.next[1] is not None else 0 for input in inputs]
        )
        target_qs = np.asarray([output.policy for output in outputs])
        self.model.fit(
            x=[input_boards, input_currs, input_nexts],
            y=target_qs,
            batch_size=self.params.batch_size,
            epochs=self.params.epochs,
            shuffle=True,
            verbose=0,
        )

    def predict(self, inputs: list[DigitPartyState]) -> list[DQNOutput]:
        input_boards = np.asarray([input.board for input in inputs])
        input_currs = np.asarray(
            [input.next[0] if input.next[0] is not None else 0 for input in inputs]
        )
        input_nexts = np.asarray(
            [input.next[1] if input.next[1] is not None else 0 for input in inputs]
        )
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


def deep_q_3x3_trained_game():
    cur_dir = pathlib.Path(__file__).parent.resolve()
    nn = DigitParty3x3DeepQNN(
        params=DQN_3X3_NN_PARAMS, model_folder=f"{cur_dir}/deepq_3x3_models/"
    )
    target_nn = DigitParty3x3DeepQNN(
        params=DQN_3X3_NN_PARAMS, model_folder=f"{cur_dir}/deepq_3x3_models/"
    )
    nn.summary()
    deepq = DeepQLearner(
        DigitParty(n=3),
        nn,
        target_nn,
        DeepQParameters(
            alpha=0.1,
            gamma=0.9,
            min_epsilon=0.05,
            max_epsilon=1,
            epsilon_decay=0.0001,
            valid_action_reward=0.01,
            memory_size=100_000,
            min_replay_size=5_000,
            minibatch_size=64,
            steps_to_train_longterm=4,
            steps_to_train_shortterm=0,
            steps_per_target_update=250,
            training_episodes=100_000,
            episodes_per_model_save=5_000,
            episodes_per_memory_save=5_000,
            episodes_per_stats_print=500,
            episodes_per_evaluation=TRAINING_EVALUATION_INTERVAL,
            state_tracker_size_bits=16_777_216,
            state_tracker_num_hashes=7,
        ),
        memory_folder=f"{cur_dir}/deepq_3x3_memory/",
        evaluator=lambda: digit_party_evaluation_summary(
            nn=nn, games=TRAINING_EVALUATION_GAMES, n=3
        ),
    )
    deepq.train()

    print(
        "Final evaluation: "
        + digit_party_evaluation_summary(
            nn=nn, games=FINAL_EVALUATION_GAMES, n=3
        ).summary
    )


def main() -> None:
    deep_q_3x3_trained_game()
