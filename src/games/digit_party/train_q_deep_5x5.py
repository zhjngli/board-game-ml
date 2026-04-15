import pathlib
from typing import NamedTuple

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
from learners.deep_q import DeepQParameters


class DigitParty5x5DQNParams(NamedTuple):
    conv_layers: int
    conv_filters: int
    dense_layers: int
    dense_units: int
    learning_rate: float
    batch_size: int
    epochs: int
    output_activation: str
    normalize_scale: float


DQN_5X5_NN_PARAMS = DigitParty5x5DQNParams(
    conv_layers=8,
    conv_filters=24,
    dense_layers=1,
    dense_units=512,
    learning_rate=0.00075,
    batch_size=128,
    epochs=1,
    output_activation="linear",
    normalize_scale=9.0,
)

DQN_5X5_PARAMS = DeepQParameters(
    alpha=0.1,
    gamma=0.95,
    min_epsilon=0.05,
    max_epsilon=1,
    epsilon_decay=0.00002,
    valid_action_reward=0.01,
    memory_size=500_000,
    min_replay_size=25_000,
    minibatch_size=128,
    steps_to_train_longterm=4,
    steps_to_train_shortterm=0,
    steps_per_target_update=500,
    training_episodes=500_000,
    episodes_per_model_save=10_000,
    episodes_per_memory_save=10_000,
    episodes_per_stats_print=1_000,
    episodes_per_evaluation=5_000,
    state_tracker_size_bits=67_108_864,
    state_tracker_num_hashes=7,
)

MODELS_FOLDER = "deepq_5x5_models"
ARTIFACTS_FOLDER = "deepq_5x5_artifacts"
TRAINING_EVAL_GAMES = 250
FINAL_EVAL_GAMES = 1000


class DigitParty5x5DeepQNN(BaseDigitPartyDeepQNN):
    def __init__(self, params: DigitParty5x5DQNParams, model_folder: str) -> None:
        super().__init__(
            model_folder=model_folder,
            batch_size=params.batch_size,
            epochs=params.epochs,
            normalize_scale=params.normalize_scale,
        )
        self.params = params

        input_board = Input(shape=(5, 5), name="dp_5x5_board")
        input_curr_digit = Input(shape=(1,), name="current_digit")
        input_next_digit = Input(shape=(1,), name="next_digit")

        board = Reshape((5, 5, 1))(input_board)
        prev = board

        for _ in range(self.params.conv_layers):
            prev = Conv2D(
                filters=self.params.conv_filters,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )(prev)

        flat = Flatten()(prev)
        prev = Concatenate()([flat, input_curr_digit, input_next_digit])

        for _ in range(self.params.dense_layers):
            prev = Dense(self.params.dense_units, activation="relu")(prev)

        q_values = Dense(25, activation=self.params.output_activation, name="q")(prev)

        self.model = Model(
            inputs=[input_board, input_curr_digit, input_next_digit], outputs=q_values
        )
        self.model.compile(
            loss="mean_squared_error",
            optimizer=Adam(learning_rate=self.params.learning_rate),
        )


def train_5x5_dqn(episodes: int | None = None) -> None:
    cur_dir = pathlib.Path(__file__).parent.resolve()
    model_folder = f"{cur_dir}/{MODELS_FOLDER}/"
    nn = DigitParty5x5DeepQNN(params=DQN_5X5_NN_PARAMS, model_folder=model_folder)
    target_nn = DigitParty5x5DeepQNN(
        params=DQN_5X5_NN_PARAMS, model_folder=model_folder
    )
    train_digit_party_dqn(
        n=5,
        nn=nn,
        target_nn=target_nn,
        params=DQN_5X5_PARAMS,
        training_artifacts_folder=f"{cur_dir}/{ARTIFACTS_FOLDER}/",
        eval_games=TRAINING_EVAL_GAMES,
        final_eval_games=FINAL_EVAL_GAMES,
        episodes_override=episodes,
    )


def eval_5x5_best(games: int = 1000) -> None:
    cur_dir = pathlib.Path(__file__).parent.resolve()
    model_folder = f"{cur_dir}/{MODELS_FOLDER}"
    nn = DigitParty5x5DeepQNN(params=DQN_5X5_NN_PARAMS, model_folder=f"{model_folder}/")
    eval_best_digit_party_dqn(
        n=5,
        nn=nn,
        best_model_file=f"{model_folder}/best_model.weights.h5",
        games=games,
    )
