import pathlib

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

from games.digit_party.deep_q_common import (
    BaseDigitPartyDeepQNN,
    eval_best_digit_party_dqn,
    train_digit_party_dqn,
)
from games.digit_party.train_deep import DP3NNParams
from learners.deep_q import DeepQParameters

"""Trains a 3x3 Digit Party network with a DQN-focused architecture baseline."""

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

DQN_3X3_PARAMS = DeepQParameters(
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
    training_episodes=200_000,
    episodes_per_model_save=5_000,
    episodes_per_memory_save=5_000,
    episodes_per_stats_print=500,
    episodes_per_evaluation=2500,
    state_tracker_size_bits=16_777_216,
    state_tracker_num_hashes=7,
)

MODELS_FOLDER = "deepq_3x3_models"
ARTIFACTS_FOLDER = "deepq_3x3_artifacts"
TRAINING_EVAL_GAMES = 500
FINAL_EVAL_GAMES = 1000


class DigitParty3x3DeepQNN(BaseDigitPartyDeepQNN):
    def __init__(self, params: DP3NNParams, model_folder: str) -> None:
        super().__init__(
            model_folder=model_folder,
            batch_size=params.batch_size,
            epochs=params.epochs,
            normalize_scale=1.0,
        )
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
        # TODO(digit-party-5x5): try broadcasting current/next digits as extra
        # 2D planes before the conv stack instead of appending them here as
        # scalars. That would let spatial filters condition on digit context
        # earlier in the network.
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


def train_3x3_dqn(episodes: int | None = None) -> None:
    cur_dir = pathlib.Path(__file__).parent.resolve()
    model_folder = f"{cur_dir}/{MODELS_FOLDER}/"
    nn = DigitParty3x3DeepQNN(params=DQN_3X3_NN_PARAMS, model_folder=model_folder)
    target_nn = DigitParty3x3DeepQNN(
        params=DQN_3X3_NN_PARAMS, model_folder=model_folder
    )
    train_digit_party_dqn(
        n=3,
        nn=nn,
        target_nn=target_nn,
        params=DQN_3X3_PARAMS,
        training_artifacts_folder=f"{cur_dir}/{ARTIFACTS_FOLDER}/",
        eval_games=TRAINING_EVAL_GAMES,
        final_eval_games=FINAL_EVAL_GAMES,
        episodes_override=episodes,
    )


def eval_3x3_best(games: int = 1000) -> None:
    cur_dir = pathlib.Path(__file__).parent.resolve()
    model_folder = f"{cur_dir}/{MODELS_FOLDER}"
    nn = DigitParty3x3DeepQNN(params=DQN_3X3_NN_PARAMS, model_folder=f"{model_folder}/")
    eval_best_digit_party_dqn(
        n=3,
        nn=nn,
        best_model_file=f"{model_folder}/best_model.weights.h5",
        games=games,
    )


def deep_q_3x3_trained_game() -> None:
    train_3x3_dqn()


def best_model_game(games: int = 1000) -> None:
    eval_3x3_best(games=games)
