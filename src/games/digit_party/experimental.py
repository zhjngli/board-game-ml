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
from learners.deep_q import Policy
from nn.neural_network import NeuralNetwork


class DigitParty3x3NeuralNetwork(NeuralNetwork[DigitPartyIR, Policy]):
    NUM_CHANNELS = 1
    DROPOUT_RATE = 0.3
    LEARN_RATE = 0.05
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
        pi = Dense(9, activation="relu", name="pi")(dense3)
        # TODO: add an output representing the score of the input board?

        self.model = Model(
            inputs=[input_board, input_curr_digit, input_next_digit], outputs=[pi]
        )
        self.model.compile(
            loss=["mean_squared_error"],
            optimizer=Adam(learning_rate=self.LEARN_RATE),
            metrics=["accuracy"],
        )
        self.model.summary()

    def train(self, data: List[Tuple[DigitPartyIR, Policy]]) -> None:
        inputs: List[DigitPartyIR]
        outputs: List[Policy]
        inputs, outputs = list(zip(*data))
        input_boards = np.asarray([input.board for input in inputs])
        input_currs = np.asarray(
            [input.next[0] if input.next[0] is not None else 0 for input in inputs]
        )
        input_nexts = np.asarray(
            [input.next[1] if input.next[1] is not None else 0 for input in inputs]
        )
        policies = np.asarray(outputs)
        print("boards: ", len(input_boards), input_boards)
        print("currs: ", len(input_currs), input_currs)
        print("nexts: ", len(input_nexts), input_nexts)
        print("pis: ", len(policies), policies)
        self.model.fit(
            x=[input_boards, input_currs, input_nexts],
            y=policies,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            shuffle=True,
        )

    def predict(self, inputs: List[DigitPartyIR]) -> List[Policy]:
        input_boards = np.asarray([input.board for input in inputs])
        input_currs = np.asarray(
            [input.next[0] if input.next[0] is not None else 0 for input in inputs]
        )
        input_nexts = np.asarray(
            [input.next[1] if input.next[1] is not None else 0 for input in inputs]
        )
        pis = self.model.predict([input_boards, input_currs, input_nexts], verbose=0)
        return pis

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


def deep_q_3x3_trained_game() -> None:
    cur_dir = pathlib.Path(__file__).parent.resolve()
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

    for i in tqdm(range(latest + 1, 100), desc="training on each chunk"):
        with open(f"{cur_dir}/chunked_simple_q_data/{i:04d}_chunk.pkl", "rb") as file:
            q_table: dict[DigitPartyIR, dict[DigitPartyPlacement, float]] = pickle.load(
                file
            )
        # convert q_table to states -> policies
        training_data: List[Tuple[DigitPartyIR, Policy]] = []
        for state, actions in tqdm(
            q_table.items(), desc=f"converting chunk {i} to nn ins/outs"
        ):
            policy = np.zeros(len(actions))
            for (r, c), q in actions.items():
                a = r * 3 + c
                policy[a] = q
            training_data.append((state, policy))

        nn.train(training_data)
        nn.save(f"simple_q_data_incremental_{i:04d}.weights.h5")

    g = DigitParty(n=3)
    random = 0
    prediction = 0

    def deepq_play(state: DigitPartyState) -> DigitPartyPlacement:
        nonlocal random, prediction
        pi = nn.predict([DigitParty.to_immutable(state)])[0]
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

    computer_game(g, 100, deepq_play)
    print(f"random: {random}")
    print(f"prediction: {prediction}")
