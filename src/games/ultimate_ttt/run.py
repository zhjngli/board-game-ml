import os
import pathlib
from typing import Deque, List, Tuple

import numpy as np
from keras.layers import (  # type: ignore
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Reshape,
)
from keras.models import Model  # type: ignore

# from keras.optimizers import Adam  # type: ignore
# keras 2.11+ optimizer Adam runs slowly on M1/M2, so use legacy
from keras.optimizers.legacy import Adam  # type: ignore
from typing_extensions import override

from games.game import P1, P2, VALID
from games.ultimate_ttt.ultimate import (
    Location,
    Section,
    UltimateBoard,
    UltimateIR,
    UltimateState,
    UltimateTicTacToe,
    ir_to_state,
)
from learners.alpha_zero.alpha_zero import A0Parameters, AlphaZero
from learners.alpha_zero.monte_carlo_tree_search import MCTSParameters
from learners.monte_carlo import MonteCarloLearner
from learners.trainer import Trainer
from nn.neural_network import NeuralNetwork, Policy, Value


def human_game() -> None:
    g = UltimateTicTacToe()
    p = 0
    t = "XO"

    while not g.is_finished():
        print(f"\n{g.show()}\n")

        if g.active_nonant is None:
            print(
                "Last active section is finished, or it's a new game. You can freely choose a section to play in."
            )
            sec = input(f"player {p + 1} ({t[p]}) choose a section: ").strip()
            try:
                RC = sec.split(",")[:2]
                R = int(RC[0])
                C = int(RC[1])
            except (ValueError, IndexError):
                print(
                    "can't read your section input. input as comma separated, e.g. 1,1"
                )
                continue
        else:
            # gets past mypy error. if this happens, error will happen downstream when trying to play at (-1, -1)
            (R, C) = g.active_nonant if g.active_nonant is not None else (-1, -1)
            print(
                f"The last active section is {R, C}, you are forced to play in that section."
            )

        loc = input(
            f"player {p + 1} ({t[p]}) choose a location in section {R, C}: "
        ).strip()
        try:
            rc = loc.split(",")[:2]
            r = int(rc[0])
            c = int(rc[1])
        except (ValueError, IndexError):
            print("can't read your location input. input as comma separated, e.g. 1,1")
            continue

        try:
            g.play((R, C), (r, c))
        except ValueError as e:
            print(f"\n\n{str(e)}")
            continue

        p += 1
        p %= 2

    print(f"{g.show()}\n")
    print("game over!")


class UltimateMonteCarloTrainer(UltimateTicTacToe, Trainer):
    def __init__(self, p1: MonteCarloLearner, p2: MonteCarloLearner) -> None:
        super().__init__()
        self.p1 = p1
        self.p2 = p2

    @override
    def train(self, episodes=1000) -> None:
        super().train(episodes)

        self.p1.save_policy()
        self.p2.save_policy()

    def train_once(self) -> None:
        while not self.is_finished():
            r, c = self.p1.choose_action(
                UltimateTicTacToe.immutable_representation(self.state())
            )
            self.play(r, c)
            self.p1.add_state(UltimateTicTacToe.immutable_representation(self.state()))

            if self.is_finished():
                break

            r, c = self.p2.choose_action(
                UltimateTicTacToe.immutable_representation(self.state())
            )
            self.play(r, c)
            self.p2.add_state(UltimateTicTacToe.immutable_representation(self.state()))

            if self.is_finished():
                break

        self.give_rewards()
        self.reset()
        self.p1.reset_states()
        self.p2.reset_states()

    def give_rewards(self) -> None:
        # TODO: how might changing these rewards affect behavior?
        if self.win(P1):
            self.p1.propagate_reward(1)
            self.p2.propagate_reward(0)
        elif self.win(P2):
            self.p1.propagate_reward(0)
            self.p2.propagate_reward(1)
        elif self.board_filled():
            self.p1.propagate_reward(0.1)
            self.p2.propagate_reward(0.5)
        else:
            raise Exception("giving rewards when game's not over. something's wrong!")


class UltimateMonteCarloLearner(
    MonteCarloLearner[UltimateIR, Tuple[Section, Location]]
):
    def get_actions_from_state(
        self, state: UltimateIR
    ) -> List[Tuple[Section, Location]]:
        valid_actions = UltimateTicTacToe.actions(ir_to_state(state))
        return [
            UltimateTicTacToe.from_action(a)
            for a in range(len(valid_actions))
            if valid_actions[a] == VALID
        ]

    def apply(self, ir: UltimateIR, action: Tuple[Section, Location]) -> UltimateIR:
        (sec, loc) = action
        a = UltimateTicTacToe.to_action(sec, loc)
        s = ir_to_state(ir)
        return UltimateTicTacToe.immutable_representation(UltimateTicTacToe.apply(s, a))


MCP1_POLICY = "src/games/ultimate_ttt/mcp1.pkl"
MCP2_POLICY = "src/games/ultimate_ttt/mcp2.pkl"


def monte_carlo_trained_game():
    computer1 = UltimateMonteCarloLearner(policy_file=MCP1_POLICY)
    computer2 = UltimateMonteCarloLearner(policy_file=MCP2_POLICY)
    g = UltimateMonteCarloTrainer(p1=computer1, p2=computer2)
    g.train(episodes=1000)

    while not g.is_finished():
        print(f"\n{g.show()}\n")
        sec, loc = computer1.choose_action(
            UltimateTicTacToe.immutable_representation(g.state()), exploit=True
        )
        g.play(sec, loc)
        print(f"computer X plays at section {sec} location {loc}")
        if g.is_finished():
            break

        print(f"\n{g.show()}\n")
        sec, loc = computer2.choose_action(
            UltimateTicTacToe.immutable_representation(g.state()), exploit=True
        )
        g.play(sec, loc)
        print(f"computer O plays at section {sec} location {loc}")

    print(g.show())
    print("\ngame over!")


class UltimateNeuralNetwork(NeuralNetwork[UltimateState]):
    NUM_CHANNELS = 1
    DROPOUT_RATE = 0.3
    LEARN_RATE = 0.01
    BATCH_SIZE = 64
    EPOCHS = 10

    def __init__(self, model_folder: str) -> None:
        self.model_folder = model_folder
        # no 4d conv layer so reshape input to 9x9
        input = Input(
            shape=(3, 3, 3, 3), name="UltimateBoardInput"
        )  # TODO: batch size? defaults to None I think.
        # each layer is a 4D tensor consisting of: batch_size, board_height, board_width, num_channels
        board = Reshape((9, 9, self.NUM_CHANNELS))(input)
        # normalize along channels axis
        conv1 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(filters=self.NUM_CHANNELS, kernel_size=(3, 3), padding="valid")(
                    board
                )
            )
        )
        conv2 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(filters=self.NUM_CHANNELS, kernel_size=(3, 3), padding="valid")(
                    conv1
                )
            )
        )
        conv3 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(filters=self.NUM_CHANNELS, kernel_size=(3, 3), padding="valid")(
                    conv2
                )
            )
        )
        flat = Flatten()(conv3)
        dense1 = Dropout(rate=self.DROPOUT_RATE)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(1024)(flat)))
        )
        dense2 = Dropout(rate=self.DROPOUT_RATE)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(512)(dense1)))
        )

        # policy, guessing the value of each valid action at the input state
        pi = Dense(UltimateTicTacToe.num_actions(), activation="softmax", name="pi")(
            dense2
        )
        # value, guessing the value of the input state
        v = Dense(1, activation="tanh", name="v")(dense2)

        self.model = Model(inputs=input, outputs=[pi, v])
        self.model.compile(
            loss=["categorical_crossentropy", "mean_squared_error"],
            optimizer=Adam(learning_rate=self.LEARN_RATE),
        )
        self.model.summary()

    def train(self, data: List[Deque[Tuple[UltimateBoard, Policy, Value]]]) -> None:
        input_boards, target_pis, target_vs = list(zip(*data))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.model.fit(
            x=input_boards,
            y=[target_pis, target_vs],
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
        )

    def predict(self, state: UltimateState) -> Tuple[Policy, Value]:
        inputs = np.asarray([state.board])  # array of inputs, so add 1 to the dimension
        pis, vs = self.model.predict(inputs)
        # TODO: can i choose different predictions here?
        return pis[0], vs[0]

    def save(self, file: str) -> None:
        if not os.path.exists(self.model_folder):
            print(f"Making directory for models at: {self.model_folder}")
            os.makedirs(self.model_folder)
        model_path = os.path.join(self.model_folder, file)
        self.model.save_weights(model_path)

    def load(self, file: str) -> None:
        model_path = os.path.join(self.model_folder, file)
        self.model.load_weights(model_path)


def alpha_zero_trained_game():
    a0 = AlphaZero(
        UltimateTicTacToe(),
        UltimateNeuralNetwork(
            model_folder=f"{pathlib.Path(__file__).parent.resolve()}/a0_nn_models/"
        ),
        A0Parameters(
            temp_threshold=11,
            pit_games=100,
            pit_threshold=0.55,
            training_episodes=100,
            training_games_per_episode=100,
            training_queue_length=10,
            training_hist_max_len=20,
        ),
        MCTSParameters(
            num_searches=40,
            cpuct=1,
            epsilon=1e-4,
        ),
    )
    a0.train()
