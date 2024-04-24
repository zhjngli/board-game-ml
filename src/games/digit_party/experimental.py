import math
import os
import pathlib
import pickle
from typing import List, NamedTuple, Tuple

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe  # type: ignore
from keras.layers import (  # type: ignore
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Reshape,
)
from keras.models import Model  # type: ignore
from keras.optimizers import Adam  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from tqdm import tqdm

from games.digit_party.game import (
    DigitParty,
    DigitPartyIR,
    DigitPartyPlacement,
    DigitPartyState,
)
from games.digit_party.run import computer_game
from games.game import VALID
from learners.deep_q import DQNOutput
from nn.neural_network import NeuralNetwork


class DP3NNParams(NamedTuple):
    conv_layers: int
    conv_filters: int
    dense_layers: int
    dense_units: int
    learning_rate: float
    batch_size: int
    epochs: int
    dropout_rate: float
    output_activation: str


class DigitParty3x3NeuralNetwork(NeuralNetwork[DigitPartyIR, DQNOutput]):
    def __init__(self, params: DP3NNParams, model_folder: str) -> None:
        super().__init__(model_folder)
        self.params = params

        input_board = Input(shape=(3, 3), name="dp_3x3_board")
        input_curr_digit = Input(shape=(1,), name="current_digit")
        input_next_digit = Input(shape=(1,), name="next_digit")
        # each layer is a 4D tensor consisting of: batch_size, board_height, board_width, num_channels
        board = Reshape((3, 3, 1))(input_board)
        prev = board

        # setup conv layers
        for _ in range(self.params.conv_filters):
            # normalize along channels axis
            conv = Activation("relu")(
                BatchNormalization(axis=3)(
                    Conv2D(
                        filters=self.params.conv_filters,
                        kernel_size=(2, 2),
                        padding="same",
                    )(prev)
                )
            )
            prev = conv

        flat = Flatten()(prev)
        concat = Concatenate()([flat, input_curr_digit, input_next_digit])
        prev = concat

        # setup dense layers
        for _ in range(self.params.dense_layers):
            # TODO: divide units by 2 each layer?
            dense = Dropout(rate=self.params.dropout_rate)(
                Activation("relu")(
                    BatchNormalization(axis=1)(Dense(self.params.dense_units)(prev))
                )
            )
            prev = dense

        # policy, guessing the value of each valid action at the input state
        # TODO: was softmax before
        pi = Dense(9, activation=self.params.output_activation, name="pi")(prev)
        v = Dense(1, activation=self.params.output_activation, name="v")(prev)

        self.model = Model(
            inputs=[input_board, input_curr_digit, input_next_digit], outputs=[pi, v]
        )
        self.model.compile(
            loss=["mean_squared_error", "mean_squared_error"],
            optimizer=Adam(learning_rate=self.params.learning_rate),
            metrics={"pi": ["accuracy", "mse"], "v": ["accuracy", "mse"]},
        )
        self.model.summary()

    def train(self, data: List[Tuple[DigitPartyIR, DQNOutput]]) -> None:
        inputs: List[DigitPartyIR]
        outputs: List[DQNOutput]
        inputs, outputs = list(zip(*data))
        input_boards = np.asarray([input.board for input in inputs])
        input_currs = np.asarray(
            [input.next[0] if input.next[0] is not None else 0 for input in inputs]
        )
        input_nexts = np.asarray(
            [input.next[1] if input.next[1] is not None else 0 for input in inputs]
        )
        target_pis = np.asarray([output.policy for output in outputs])
        target_vs = np.asarray([output.value for output in outputs])
        print("boards: ", len(input_boards), input_boards)
        print("currs: ", len(input_currs), input_currs)
        print("nexts: ", len(input_nexts), input_nexts)
        print("pis: ", len(target_pis), target_pis)
        print("vs: ", len(target_vs), target_vs)
        self.model.fit(
            x=[input_boards, input_currs, input_nexts],
            y=[target_pis, target_vs],
            batch_size=self.params.batch_size,
            epochs=self.params.epochs,
            shuffle=True,
        )

    def predict(self, inputs: List[DigitPartyIR]) -> List[DQNOutput]:
        input_boards = np.asarray([input.board for input in inputs])
        input_currs = np.asarray(
            [input.next[0] if input.next[0] is not None else 0 for input in inputs]
        )
        input_nexts = np.asarray(
            [input.next[1] if input.next[1] is not None else 0 for input in inputs]
        )
        pis, vs = self.model.predict(
            [input_boards, input_currs, input_nexts], verbose=0
        )
        return [DQNOutput(policy=pi, value=v) for pi, v in zip(pis, vs)]

    def save(self, file: str) -> None:
        if not os.path.exists(self.model_folder):
            print(f"Making directory for models at: {self.model_folder}")
            os.makedirs(self.model_folder)
        model_path = os.path.join(self.model_folder, file)
        self.model.save_weights(model_path)

    def load(self, file: str) -> None:
        model_path = os.path.join(self.model_folder, file)
        self.model.load_weights(model_path)

    def set_weights(self, weights) -> None:
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()


orig_nn_params: DP3NNParams = DP3NNParams(
    conv_layers=4,
    conv_filters=5,  # one for each digit, 0-4
    dense_layers=4,
    dense_units=256,
    learning_rate=0.1,
    batch_size=64,
    epochs=10,
    dropout_rate=0.3,
    output_activation="linear",
)


def bayesian_optimization() -> None:
    cur_dir = pathlib.Path(__file__).parent.resolve()
    space = {
        "num_conv_layers": hp.uniformint("num_conv_layers", 0, 10),
        "num_conv_filters": hp.uniformint("num_conv_filters", 1, 64),
        "num_dense_layers": hp.uniformint("num_dense_layers", 0, 10),
        "num_dense_units": hp.qloguniform(
            "num_dense_units", np.log(1), np.log(2048), 1
        ),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.0001), np.log(0.1)),
        "batch_size": hp.qloguniform("batch_size", np.log(1), np.log(512), 1),
        "epochs": hp.uniformint("epochs", 10, 50),
        "dropout_rate": hp.uniform("dropout_rate", 0, 0.5),
        "output_activation": hp.choice("output_activation", ["linear", "relu"]),
    }

    params_file = f"{cur_dir}/opt.pkl"
    if os.path.isfile(params_file):
        with open(params_file, "rb") as file:
            params_to_val_hist = pickle.load(file)
    else:
        params_to_val_hist = []

    training_data: List[Tuple[DigitPartyIR, DQNOutput]] = []
    for i in tqdm(range(10), desc="loading chunks for training data"):
        with open(f"{cur_dir}/chunked_simple_q_data/{i:04d}_chunk.pkl", "rb") as file:
            q_table: dict[DigitPartyIR, dict[DigitPartyPlacement, float]] = pickle.load(
                file
            )
        # convert q_table to states -> policies
        for state, actions in tqdm(
            q_table.items(), desc=f"converting chunk {i} to nn ins/outs"
        ):
            pi = np.zeros(len(actions))
            for (r, c), q in actions.items():
                a = r * 3 + c
                pi[a] = q
            v = DigitParty.calc_score(state)
            training_data.append((state, DQNOutput(policy=pi, value=v)))

    def objective(params):
        # force batch_size to be a power of 2
        batch_size = int(2 ** round(math.log2(params["batch_size"])))
        dp_params = DP3NNParams(
            conv_layers=params["num_conv_layers"],
            conv_filters=params["num_conv_filters"],
            dense_layers=params["num_dense_layers"],
            dense_units=int(
                params["num_dense_units"]
            ),  # qloguniform doesn't return int for some reason
            learning_rate=params["learning_rate"],
            batch_size=batch_size,
            epochs=params["epochs"],
            dropout_rate=params["dropout_rate"],
            output_activation=params["output_activation"],
        )
        model = DigitParty3x3NeuralNetwork(
            params=dp_params, model_folder=f"{cur_dir}/opt_models/"
        )

        inputs: List[DigitPartyIR]
        outputs: List[DQNOutput]
        inputs, outputs = list(zip(*training_data))
        input_boards = np.asarray([input.board for input in inputs])
        input_currs = np.asarray(
            [input.next[0] if input.next[0] is not None else 0 for input in inputs]
        )
        input_nexts = np.asarray(
            [input.next[1] if input.next[1] is not None else 0 for input in inputs]
        )
        target_pis = np.asarray([output.policy for output in outputs])
        target_vs = np.asarray([output.value for output in outputs])

        (
            board_train,
            board_val,
            curr_train,
            curr_val,
            next_train,
            next_val,
            pi_train,
            pi_val,
            v_train,
            v_val,
        ) = train_test_split(
            input_boards,
            input_currs,
            input_nexts,
            target_pis,
            target_vs,
            test_size=0.2,
            random_state=42,
        )
        history = model.model.fit(
            [board_train, curr_train, next_train],
            [pi_train, v_train],
            batch_size=batch_size,
            epochs=params["epochs"],
            validation_data=([board_val, curr_val, next_val], [pi_val, v_val]),
            verbose=0,
        )

        val_loss_history = history.history["val_loss"]
        params_to_val_hist.append({"params": params, "val_loss": val_loss_history})

        return {
            "loss": val_loss_history[-1],
            "status": STATUS_OK,
            "params": params,
            "val_loss_hist": val_loss_history,
        }

    with open(params_file, "wb") as f:
        pickle.dump(params_to_val_hist, f)

    max_evals = 100
    trials_file = f"{cur_dir}/trials.pkl"
    if os.path.isfile(trials_file):
        with open(trials_file, "rb") as pickle_file:
            trials = pickle.load(pickle_file)
    else:
        trials = Trials()

    with open(trials_file, "wb") as f:
        pickle.dump(trials, f)

    for i in range(max_evals):
        try:
            best = fmin(
                objective,
                space=space,
                algo=tpe.suggest,
                max_evals=len(trials.trials) + 1,
                trials=trials,
            )
            with open(trials_file, "wb") as trials_pkl:
                pickle.dump(trials, trials_pkl)

            print(f"best params on trial {i}: {best}")
            print(f"params: {trials.best_trial['result']['params']}")
            print(f"loss: {trials.best_trial['result']['loss']}")
            print(f"hist: {trials.best_trial['result']['val_loss_hist']}")

        except Exception as e:
            print("Exception occurred:", e)
            with open(trials_file, "rb") as read_trials:
                trials = pickle.load(read_trials)


def deep_q_3x3_chunk_trained_game() -> None:  # noqa: C901
    cur_dir = pathlib.Path(__file__).parent.resolve()

    # code to chunk the full data from q-3x3.pkl
    # with open(f"{cur_dir}/q-3x3.pkl", "rb") as file:
    #     q_table: dict[DigitPartyIR, dict[DigitPartyPlacement, float]] = pickle.load(
    #         file
    #     )

    # keys = list(q_table.keys())
    # chunks = np.array_split(np.arange(len(keys)), 1000)
    # for i, chunk in enumerate(chunks):
    #     chunked = {keys[ix]: q_table[keys[ix]] for ix in chunk}
    #     with open(f"{cur_dir}/chunked_simple_q_data/{i:04d}_chunk.pkl", "wb") as file:
    #         pickle.dump(chunked, file)

    model_folder = f"{cur_dir}/experimental3x3_models/"
    nn = DigitParty3x3NeuralNetwork(params=orig_nn_params, model_folder=model_folder)
    latest = 0
    latest_model = None
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    for filename in os.listdir(model_folder):
        f = os.path.join(model_folder, filename)
        if os.path.isfile(f):
            try:
                i = int(
                    filename.split(".")[0].split("_")[-1]
                )  # simple_q_data_incremental_{i:04d}.weights.h5
            except ValueError:
                # any other model
                continue
            if i >= latest:
                latest = i
                latest_model = f

    if latest_model:
        print(f"loading {latest_model}")
        nn.load(latest_model)

    for i in tqdm(range(latest + 1, 300), desc="training on each chunk"):
        with open(f"{cur_dir}/chunked_simple_q_data/{i:04d}_chunk.pkl", "rb") as file:
            q_table: dict[DigitPartyIR, dict[DigitPartyPlacement, float]] = pickle.load(
                file
            )
        # convert q_table to states -> policies
        training_data: List[Tuple[DigitPartyIR, DQNOutput]] = []
        for state, actions in tqdm(
            q_table.items(), desc=f"converting chunk {i} to nn ins/outs"
        ):
            pi = np.zeros(len(actions))
            for (r, c), q in actions.items():
                a = r * 3 + c
                pi[a] = q
            v = DigitParty.calc_score(state)
            training_data.append((state, DQNOutput(policy=pi, value=v)))

        nn.train(training_data)
        nn.save(f"simple_q_data_incremental_{i:04d}.weights.h5")

    deep_play_digit_party(100, 3, nn)


def deep_q_3x3_full_trained_game() -> None:
    cur_dir = pathlib.Path(__file__).parent.resolve()
    model_folder = f"{cur_dir}/experimental3x3_models/"
    nn = DigitParty3x3NeuralNetwork(params=orig_nn_params, model_folder=model_folder)

    # could also load full data at once instead of loading each chunk
    training_data: List[Tuple[DigitPartyIR, DQNOutput]] = []
    for i in tqdm(range(0, 1000), desc="loading chunks"):
        with open(f"{cur_dir}/chunked_simple_q_data/{i:04d}_chunk.pkl", "rb") as file:
            q_table: dict[DigitPartyIR, dict[DigitPartyPlacement, float]] = pickle.load(
                file
            )
        for state, actions in tqdm(
            q_table.items(), desc=f"converting chunk {i} to nn ins/outs"
        ):
            pi = np.zeros(len(actions))
            for (r, c), q in actions.items():
                a = r * 3 + c
                pi[a] = q
            v = DigitParty.calc_score(state)
            training_data.append((state, DQNOutput(policy=pi, value=v)))

    nn.train(training_data)
    nn.save("simple_q_data.weights.h5")

    deep_play_digit_party(100, 3, nn)


def deep_play_digit_party(games: int, n: int, nn: NeuralNetwork) -> None:
    g = DigitParty(n=n)
    random = 0
    prediction = 0

    def deepq_play(state: DigitPartyState) -> DigitPartyPlacement:
        nonlocal random, prediction
        out = nn.predict([DigitParty.to_immutable(state)])[0]
        pi = out.policy
        action_statuses = np.asarray(g.actions(state))
        valid_actions = np.where(action_statuses == VALID)[0]
        a = valid_actions[np.argmax(pi[valid_actions])]
        # a = np.argmax(pi)
        if not np.isin(valid_actions, a).any():
            a = np.random.choice(valid_actions)
            print("######### CHOSE RANDOM ACTION", a)
            random += 1
        else:
            prediction += 1

        n = state.board.shape[0]
        r = int(a / n)
        c = int(a % n)
        return r, c

    computer_game(g, games, deepq_play)
    print(f"random: {random}")
    print(f"prediction: {prediction}")
