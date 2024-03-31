import os
import pathlib
from typing import List, Tuple

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
from keras.optimizers import Adam  # type: ignore
from typing_extensions import override

from games.game import P1, P2, VALID, Action, State
from games.ultimate_ttt.ultimate import (
    Location,
    Section,
    UltimateIR,
    UltimateTicTacToe,
    ir_to_state,
)
from learners.alpha_zero.alpha_zero import (
    A0NNInput,
    A0NNOutput,
    A0Parameters,
    AlphaZero,
)
from learners.alpha_zero.monte_carlo_tree_search import (
    MCTSParameters,
    MonteCarloTreeSearch,
)
from learners.monte_carlo import MonteCarloLearner
from learners.trainer import Trainer
from nn.neural_network import NeuralNetwork


def human_play(p: int, g: UltimateTicTacToe) -> Tuple[Section, Location]:
    t = "XO"
    if g.active_nonant is None:
        print(
            "Last active section is finished, or it's a new game. You can freely choose a section to play in."
        )
        sec = input(f"player {p + 1} ({t[p]}) choose a section: ").strip()
        try:
            RC = sec.split(",")[:2]
            R = int(RC[0])
            C = int(RC[1])
        except (ValueError, IndexError) as e:
            print("can't read your section input. input as comma separated, e.g. 1,1")
            raise e
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
    except (ValueError, IndexError) as e:
        print("can't read your location input. input as comma separated, e.g. 1,1")
        raise e

    return ((R, C), (r, c))


def human_game() -> None:
    g = UltimateTicTacToe()
    p = 0

    while not g.is_finished():
        print(f"\n{g.show()}\n")

        try:
            (R, C), (r, c) = human_play(p, g)
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
            r, c = self.p1.choose_action(UltimateTicTacToe.to_immutable(self.state()))
            self.play(r, c)
            self.p1.add_state(UltimateTicTacToe.to_immutable(self.state()))

            if self.is_finished():
                break

            r, c = self.p2.choose_action(UltimateTicTacToe.to_immutable(self.state()))
            self.play(r, c)
            self.p2.add_state(UltimateTicTacToe.to_immutable(self.state()))

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
        return UltimateTicTacToe.to_immutable(UltimateTicTacToe.apply(s, a))


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
            UltimateTicTacToe.to_immutable(g.state()), exploit=True
        )
        g.play(sec, loc)
        print(f"computer X plays at section {sec} location {loc}")
        if g.is_finished():
            break

        print(f"\n{g.show()}\n")
        sec, loc = computer2.choose_action(
            UltimateTicTacToe.to_immutable(g.state()), exploit=True
        )
        g.play(sec, loc)
        print(f"computer O plays at section {sec} location {loc}")

    print(g.show())
    print("\ngame over!")


class UltimateNeuralNetwork(NeuralNetwork[A0NNInput, A0NNOutput]):
    NUM_CHANNELS = 1
    DROPOUT_RATE = 0.3
    LEARN_RATE = 0.01
    BATCH_SIZE = 64
    EPOCHS = 10

    def __init__(self, model_folder: str) -> None:
        super().__init__(model_folder)

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
            metrics={"pi": ["accuracy"], "v": ["accuracy", "mse"]},
        )
        self.model.summary()

    def train(self, data: List[Tuple[A0NNInput, A0NNOutput]]) -> None:
        inputs: List[A0NNInput]
        outputs: List[A0NNOutput]
        inputs, outputs = list(zip(*data))
        input_boards = np.asarray([input.board for input in inputs])
        target_pis = np.asarray([output.policy for output in outputs])
        target_vs = np.asarray([output.value for output in outputs])
        self.model.fit(
            x=input_boards,
            y=[target_pis, target_vs],
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            shuffle=True,
        )

    def predict(self, inputs: List[A0NNInput]) -> List[A0NNOutput]:
        boards = np.asarray([i.board for i in inputs])
        pis, vs = self.model.predict(boards, verbose=0)
        return [A0NNOutput(policy=pi, value=v) for pi, v in zip(pis, vs)]

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


def alpha_zero_trained_game():
    cur_dir = pathlib.Path(__file__).parent.resolve()
    a0 = AlphaZero(
        UltimateTicTacToe(),
        UltimateNeuralNetwork(model_folder=f"{cur_dir}/a0_nn_models/"),
        A0Parameters(
            temp_threshold=11,
            pit_games=20,
            pit_threshold=0.55,
            training_episodes=200,
            training_games_per_episode=10,
            training_queue_length=10000,
            training_hist_max_len=50,
        ),
        MCTSParameters(
            num_searches=100,
            cpuct=1,
            epsilon=1e-4,
        ),
        training_examples_folder=f"{cur_dir}/a0_training_examples/",
    )
    a0.train()

    g = UltimateTicTacToe()
    params = MCTSParameters(
        num_searches=100,
        cpuct=1,
        epsilon=1e-4,
    )
    nn1 = UltimateNeuralNetwork(model_folder=f"{cur_dir}/a0_nn_models/")
    nn1.load("best_model.weights.h5")
    nn2 = UltimateNeuralNetwork(model_folder=f"{cur_dir}/a0_nn_models/")
    nn2.load("best_model.weights.h5")
    mcts1 = MonteCarloTreeSearch(g, nn1, params)
    mcts2 = MonteCarloTreeSearch(g, nn2, params)

    def play1(s: State) -> Action:
        return int(np.argmax(mcts1.action_probabilities(s, temperature=0)))

    def play2(s: State) -> Action:
        return int(np.argmax(mcts2.action_probabilities(s, temperature=0)))

    while not g.is_finished():
        print(f"\n{g.show()}\n")
        sec, loc = UltimateTicTacToe.from_action(play1(g.state()))
        g.play(sec, loc)
        print(f"computer X plays at section {sec} location {loc}")
        if g.is_finished():
            break

        print(f"\n{g.show()}\n")
        sec, loc = UltimateTicTacToe.from_action(play2(g.state()))
        g.play(sec, loc)
        print(f"computer O plays at section {sec} location {loc}")

    print(g.show())
    print("\ngame over!")


def vs_alpha_zero_game():
    cur_dir = pathlib.Path(__file__).parent.resolve()
    params = MCTSParameters(
        num_searches=100,
        cpuct=1,
        epsilon=1e-4,
    )
    g = UltimateTicTacToe()
    nn = UltimateNeuralNetwork(model_folder=f"{cur_dir}/a0_nn_models/")
    nn.load("best_model.h5")
    mcts = MonteCarloTreeSearch(g, nn, params)

    def nn_play(s: State) -> Action:
        return int(np.argmax(mcts.action_probabilities(s, temperature=0)))

    while not g.is_finished():
        print(f"\n{g.show()}\n")

        try:
            (R, C), (r, c) = human_play(0, g)
            g.play((R, C), (r, c))
        except ValueError as e:
            print(f"\n\n{str(e)}")
            continue

        if g.is_finished():
            break

        print(f"\n{g.show()}\n")
        sec, loc = UltimateTicTacToe.from_action(nn_play(g.state()))
        g.play(sec, loc)
        print(f"computer O plays at section {sec} location {loc}")

    print(g.show())
    print("\ngame over!")
