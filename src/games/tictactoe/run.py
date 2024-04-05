import os
import pathlib
from typing import Callable, List, Optional, Tuple

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
from tqdm import tqdm
from typing_extensions import override

from games.game import P1, P2
from games.tictactoe.tictactoe import (
    Empty,
    TicTacToe,
    TicTacToeIR,
    TicTacToeState,
    tile_char,
)
from learners.alpha_zero.alpha_zero import A0Parameters, AlphaZero
from learners.alpha_zero.monte_carlo_tree_search import (
    MCTSParameters,
    MonteCarloTreeSearch,
)
from learners.alpha_zero.types import A0NNInput, A0NNOutput
from learners.monte_carlo import MonteCarloLearner
from learners.q import SimpleQLearner
from learners.trainer import Trainer
from nn.neural_network import NeuralNetwork


def human_game() -> None:
    g = TicTacToe()
    p = 0
    play = [g.play1, g.play2]
    while not g.is_finished():
        print()
        print(g.show())
        print()

        coord = input(f"player {p + 1} choose a spot: ").strip()
        try:
            rc = coord.split(",")[:2]
            r = int(rc[0])
            c = int(rc[1])
        except (ValueError, IndexError):
            print("can't read your coordinate input")
            continue

        try:
            play[p](r, c)
        except ValueError as e:
            print(str(e))
            continue

        p += 1
        p %= 2

    print(g.show())
    print("game over!")


class TicTacToeMonteCarloTrainer(TicTacToe, Trainer):
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
            r, c = self.p1.choose_action(self.to_immutable(self.state()))
            self.play1(r, c)
            self.p1.add_state(self.to_immutable(self.state()))

            if self.is_finished():
                break

            r, c = self.p2.choose_action(self.to_immutable(self.state()))
            self.play2(r, c)
            self.p2.add_state(self.to_immutable(self.state()))

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
            raise RuntimeError(
                "giving rewards when game's not over. something's wrong!"
            )


class TicTacToeMonteCarloLearner(MonteCarloLearner[TicTacToeIR, Tuple[int, int]]):
    def get_actions_from_state(self, state: TicTacToeIR) -> List[Tuple[int, int]]:
        return [
            (i, j) for i in range(3) for j in range(3) if state.board[i][j] == Empty
        ]

    def apply(self, state: TicTacToeIR, action: Tuple[int, int]) -> TicTacToeIR:
        r, c = action
        a = r * 3 + c
        return TicTacToe.applyIR(state, a)


class TicTacToeQTrainer(TicTacToe, Trainer):
    def __init__(self, p1: SimpleQLearner, p2: SimpleQLearner) -> None:
        super().__init__()
        self.p1 = p1
        self.p2 = p2

    @override
    def train(self, episodes=10000) -> None:
        super().train(episodes)

        self.p1.save_policy()
        self.p2.save_policy()

    # TODO: this training is pretty dumb, doesn't work well
    def train_once(self) -> None:
        while not self.is_finished():
            ir = self.to_immutable(self.state())
            action = self.p1.choose_action(ir)
            r, c = action
            self.play1(r, c)

            if self.is_finished():
                break

            ir = self.to_immutable(self.state())
            action = self.p2.choose_action(ir)
            r, c = action
            self.play2(r, c)

        if self.win(P1):
            self.p1.update_q_value(ir, action, 1, self.to_immutable(self.state()))
            self.p2.update_q_value(ir, action, -1, self.to_immutable(self.state()))
        elif self.win(P2):
            self.p1.update_q_value(ir, action, -1, self.to_immutable(self.state()))
            self.p2.update_q_value(ir, action, 1, self.to_immutable(self.state()))
        elif self.board_filled():
            self.p1.update_q_value(ir, action, -0.1, self.to_immutable(self.state()))
            self.p2.update_q_value(ir, action, 0, self.to_immutable(self.state()))
        else:
            raise Exception("giving rewards when game's not over. something's wrong!")
        self.reset()


class TicTacToeQLearner(SimpleQLearner[TicTacToeIR, Tuple[int, int]]):
    def default_action_q_values(self) -> dict[Tuple[int, int], float]:
        actions = {}
        for r in range(3):
            for c in range(3):
                actions[(r, c)] = 0.0
        return actions

    def get_actions_from_state(self, state: TicTacToeIR) -> List[Tuple[int, int]]:
        return [
            (i, j) for i in range(3) for j in range(3) if state.board[i][j] == Empty
        ]


def _computer_play(
    g: TicTacToe, computer: Callable[[TicTacToeState], Tuple[int, int]], verbose=True
):
    if verbose:
        print(f"\n{g.show()}\n")

    s = g.state()
    p = tile_char(s.player)
    r, c = computer(s)
    g.play(r, c)

    if verbose:
        print(f"\ncomputer {p} plays at {r, c}!")


def _human_play(g: TicTacToe):
    print(f"\n{g.show()}\n")

    coord = input("please choose a spot to play: ").strip()
    try:
        rc = coord.split(",")[:2]
        r = int(rc[0])
        c = int(rc[1])
    except (ValueError, IndexError):
        raise ValueError("can't read your coordinate input, try again")

    try:
        g.play(r, c)
    except ValueError as e:
        raise e


def _trained_game(  # noqa: C901
    g: TicTacToe,
    computer1: Callable[[TicTacToeState], Tuple[int, int]],
    computer2: Callable[[TicTacToeState], Tuple[int, int]],
):
    player = input(
        "play as player 1 or 2? or choose 0 to spectate computers play. "
    ).strip()
    try:
        p = int(player)
    except ValueError:
        print("unrecognized input, defaulting to player 1!")
        p = 1

    if p == 1:
        pass
    elif p == 2 or p == 0:
        _computer_play(g, computer1)
    else:
        print("unrecognized input, defaulting to player 1!")

    while not g.is_finished():
        if p == 0:
            _computer_play(g, computer2)
        else:
            try:
                _human_play(g)
            except ValueError as e:
                print(str(e))
                continue

        if g.is_finished():
            break

        computer = computer1 if p == 0 or p == 2 else computer2
        _computer_play(g, computer)

    print(f"\n{g.show()}\n")
    print("game over!")
    if g.win(P1):
        print("X won!")
    elif g.win(P2):
        print("O won!")
    else:
        print("tie!")


def _many_games(
    g: TicTacToe,
    computer1: Callable[[TicTacToeState], Tuple[int, int]],
    computer2: Callable[[TicTacToeState], Tuple[int, int]],
    games: int,
    desc: Optional[str] = None,
    verbose: bool = False,
):
    x_wins = 0
    o_wins = 0
    ties = 0
    if desc is None:
        desc = f"playing {games} computer games"
    for _ in tqdm(range(games), desc=desc):
        while not g.is_finished():
            _computer_play(g, computer1, verbose=verbose)
            if g.is_finished():
                break
            _computer_play(g, computer2, verbose=verbose)

        if g.win(P1):
            x_wins += 1
        elif g.win(P2):
            o_wins += 1
        else:
            ties += 1
        g.reset()

    print(f"played {games} games")
    print(f"x won {x_wins} times")
    print(f"o won {o_wins} times")
    print(f"{ties} ties")


MCP1_POLICY = "src/games/tictactoe/mcp1.pkl"
MCP2_POLICY = "src/games/tictactoe/mcp2.pkl"


def monte_carlo_trained_game(training_episodes=0):
    computer1 = TicTacToeMonteCarloLearner(policy_file=MCP1_POLICY)
    computer2 = TicTacToeMonteCarloLearner(policy_file=MCP2_POLICY)
    g = TicTacToeMonteCarloTrainer(p1=computer1, p2=computer2)
    g.train(episodes=training_episodes)

    def play1(s: TicTacToeState) -> Tuple[int, int]:
        return computer1.choose_action(TicTacToe.to_immutable(s), exploit=True)

    def play2(s: TicTacToeState) -> Tuple[int, int]:
        return computer2.choose_action(TicTacToe.to_immutable(s), exploit=True)

    _trained_game(g, play1, play2)


def monte_carlo_many_games(games=10000):
    computer1 = TicTacToeMonteCarloLearner(policy_file=MCP1_POLICY)
    computer2 = TicTacToeMonteCarloLearner(policy_file=MCP2_POLICY)
    g = TicTacToeMonteCarloTrainer(p1=computer1, p2=computer2)

    def play1(s: TicTacToeState) -> Tuple[int, int]:
        return computer1.choose_action(TicTacToe.to_immutable(s), exploit=True)

    def play2(s: TicTacToeState) -> Tuple[int, int]:
        return computer2.choose_action(TicTacToe.to_immutable(s), exploit=True)

    _many_games(g, play1, play2, games, desc="monte carlo versus games")


QP1_POLICY = "src/games/tictactoe/qp1.pkl"
QP2_POLICY = "src/games/tictactoe/qp2.pkl"


def q_trained_game(training_episodes=0):
    computer1 = TicTacToeQLearner(q_pickle=QP1_POLICY)
    computer2 = TicTacToeQLearner(q_pickle=QP2_POLICY)
    g = TicTacToeQTrainer(p1=computer1, p2=computer2)
    g.train(episodes=training_episodes)

    def play1(s: TicTacToeState) -> Tuple[int, int]:
        return computer1.choose_action(TicTacToe.to_immutable(s), exploit=True)

    def play2(s: TicTacToeState) -> Tuple[int, int]:
        return computer2.choose_action(TicTacToe.to_immutable(s), exploit=True)

    _trained_game(g, play1, play2)


def q_many_games(games=10000):
    computer1 = TicTacToeQLearner(q_pickle=QP1_POLICY)
    computer2 = TicTacToeQLearner(q_pickle=QP2_POLICY)
    g = TicTacToeQTrainer(p1=computer1, p2=computer2)

    def play1(s: TicTacToeState) -> Tuple[int, int]:
        return computer1.choose_action(TicTacToe.to_immutable(s), exploit=True)

    def play2(s: TicTacToeState) -> Tuple[int, int]:
        return computer2.choose_action(TicTacToe.to_immutable(s), exploit=True)

    _many_games(g, play1, play2, games, desc="simple q versus games")


class TTTNeuralNetwork(NeuralNetwork[A0NNInput, A0NNOutput]):
    NUM_CHANNELS = 1
    DROPOUT_RATE = 0.3
    LEARN_RATE = 0.01
    BATCH_SIZE = 64
    EPOCHS = 10

    # TODO: more layers to the network?
    def __init__(self, model_folder: str) -> None:
        super().__init__(model_folder)

        input = Input(shape=(3, 3), name="ttt_board")
        # each layer is a 4D tensor consisting of: batch_size, board_height, board_width, num_channels
        board = Reshape((3, 3, self.NUM_CHANNELS))(input)
        # normalize along channels axis
        conv1 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(filters=self.NUM_CHANNELS, kernel_size=(2, 2), padding="same")(
                    board
                )
            )
        )
        conv2 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(filters=self.NUM_CHANNELS, kernel_size=(2, 2), padding="valid")(
                    conv1
                )
            )
        )
        flat = Flatten()(conv2)
        dense1 = Dropout(rate=self.DROPOUT_RATE)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(256)(flat)))
        )
        dense2 = Dropout(rate=self.DROPOUT_RATE)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(128)(dense1)))
        )

        # policy, guessing the value of each valid action at the input state
        pi = Dense(TicTacToe.num_actions(), activation="softmax", name="pi")(dense2)
        # value, guessing the value of the input state
        v = Dense(1, activation="tanh", name="v")(dense2)

        self.model = Model(inputs=input, outputs=[pi, v])
        self.model.compile(
            loss=["categorical_crossentropy", "mean_squared_error"],
            optimizer=Adam(learning_rate=self.LEARN_RATE),
            metrics={"pi": ["accuracy", "categorical_crossentropy"], "v": ["mse"]},
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
    mcts_params = MCTSParameters(
        num_searches=100,
        cpuct=1,
        epsilon=1e-4,
    )
    a0 = AlphaZero(
        TicTacToe(),
        TTTNeuralNetwork(model_folder=f"{cur_dir}/a0_nn_models/"),
        A0Parameters(
            temp_threshold=1,
            pit_games=100,
            pit_threshold=0.55,
            training_episodes=150,
            training_games_per_episode=100,
            training_queue_length=10000,
            training_hist_max_len=20,
        ),
        mcts_params,
        training_examples_folder=f"{cur_dir}/a0_training_examples/",
    )
    a0.train()

    g = TicTacToe()
    nn = TTTNeuralNetwork(model_folder=f"{cur_dir}/a0_nn_models/")
    nn.load("best_model.weights.h5")
    mcts = MonteCarloTreeSearch(g, nn, mcts_params)

    def play(s: TicTacToeState) -> Tuple[int, int]:
        return TicTacToe.from_action(
            int(np.argmax(mcts.action_probabilities(s, temperature=0)))
        )

    _trained_game(g, play, play)


def alpha_zero_many_games(games=1000):
    cur_dir = pathlib.Path(__file__).parent.resolve()
    g = TicTacToe()
    params = MCTSParameters(
        num_searches=100,
        cpuct=1,
        epsilon=1e-4,
    )
    nn = TTTNeuralNetwork(model_folder=f"{cur_dir}/a0_nn_models/")
    nn.load("best_model.weights.h5")
    mcts = MonteCarloTreeSearch(g, nn, params)

    def play(s: TicTacToeState) -> Tuple[int, int]:
        return TicTacToe.from_action(
            int(np.argmax(mcts.action_probabilities(s, temperature=0)))
        )

    _many_games(g, play, play, games)


def a0_vs_mc_games(games=1000):
    cur_dir = pathlib.Path(__file__).parent.resolve()
    g = TicTacToe()
    params = MCTSParameters(
        num_searches=100,
        cpuct=1,
        epsilon=1e-4,
    )
    nn = TTTNeuralNetwork(model_folder=f"{cur_dir}/a0_nn_models/")
    nn.load("best_model.weights.h5")
    mcts = MonteCarloTreeSearch(g, nn, params)

    def a0_play(s: TicTacToeState) -> Tuple[int, int]:
        return TicTacToe.from_action(
            int(np.argmax(mcts.action_probabilities(s, temperature=0)))
        )

    computer1 = TicTacToeMonteCarloLearner(policy_file=MCP1_POLICY)
    computer2 = TicTacToeMonteCarloLearner(policy_file=MCP2_POLICY)

    def mc_play1(s: TicTacToeState) -> Tuple[int, int]:
        return computer1.choose_action(TicTacToe.to_immutable(s), exploit=True)

    def mc_play2(s: TicTacToeState) -> Tuple[int, int]:
        return computer2.choose_action(TicTacToe.to_immutable(s), exploit=True)

    half = int(games / 2)
    _many_games(
        g,
        a0_play,
        mc_play2,
        half,
        desc=f"P1: alpha zero. P2: monte carlo. games: {half}",
    )
    _many_games(
        g,
        mc_play1,
        a0_play,
        half,
        desc=f"P1: monte carlo. P2: alpha zero. games: {half}",
    )
