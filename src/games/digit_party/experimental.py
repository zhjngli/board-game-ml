import os
import pathlib
import pickle
from typing import List, Tuple

import numpy as np
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


class DigitParty3x3NeuralNetwork(NeuralNetwork[DigitPartyIR, DQNOutput]):
    # TODO: maybe 1 channel for each digit? means i would need to reshape the input
    # TODO: or maybe it's 1 filter for each digit?
    NUM_CHANNELS = 1
    DROPOUT_RATE = 0.3
    LEARN_RATE = 0.1
    BATCH_SIZE = 64
    EPOCHS = 10

    def __init__(self, model_folder: str) -> None:
        super().__init__(model_folder)

        input_board = Input(shape=(3, 3), name="dp_3x3_board")
        input_curr_digit = Input(shape=(1,), name="current_digit")
        input_next_digit = Input(shape=(1,), name="next_digit")
        # each layer is a 4D tensor consisting of: batch_size, board_height, board_width, num_channels
        board = Reshape((3, 3, self.NUM_CHANNELS))(input_board)
        # normalize along channels axis
        conv1 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(filters=1, kernel_size=(2, 2), padding="same")(board)
            )
        )
        conv2 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(filters=1, kernel_size=(2, 2), padding="same")(conv1)
            )
        )
        conv3 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(filters=1, kernel_size=(2, 2), padding="valid")(conv2)
            )
        )
        conv4 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(filters=1, kernel_size=(2, 2), padding="valid")(conv3)
            )
        )
        flat = Flatten()(conv4)
        concat = Concatenate()([flat, input_curr_digit, input_next_digit])
        dense1 = Dropout(rate=self.DROPOUT_RATE)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(256)(concat)))
        )
        dense2 = Dropout(rate=self.DROPOUT_RATE)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(128)(dense1)))
        )
        dense3 = Dropout(rate=self.DROPOUT_RATE)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(64)(dense2)))
        )
        dense4 = Dropout(rate=self.DROPOUT_RATE)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(32)(dense3)))
        )

        # policy, guessing the value of each valid action at the input state
        # TODO: was softmax before
        pi = Dense(9, activation="linear", name="pi")(dense4)
        v = Dense(1, activation="linear", name="v")(dense4)

        self.model = Model(
            inputs=[input_board, input_curr_digit, input_next_digit], outputs=[pi, v]
        )
        self.model.compile(
            loss=["mean_squared_error", "mean_squared_error"],
            optimizer=Adam(learning_rate=self.LEARN_RATE),
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
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
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
    nn = DigitParty3x3NeuralNetwork(model_folder=model_folder)
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
    nn = DigitParty3x3NeuralNetwork(model_folder=model_folder)

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
