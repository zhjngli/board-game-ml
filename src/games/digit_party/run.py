import os
import pathlib
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
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

# from keras.optimizers import Adam  # type: ignore
# keras 2.11+ optimizer Adam runs slowly on M1/M2, so use legacy
from keras.optimizers.legacy import Adam  # type: ignore
from matplotlib import pyplot as plt
from typing_extensions import override

from games.digit_party.game import (
    DigitParty,
    DigitPartyIR,
    DigitPartyPlacement,
    DigitPartyState,
    Empty,
)
from games.game import VALID
from learners.deep_q import DeepQLearner, DeepQParameters, Policy
from learners.q import SimpleQLearner
from learners.trainer import Trainer
from nn.neural_network import NeuralNetwork


def human_game() -> None:
    x = input("hi this is digit party. what game size? (default 5): ").strip()
    if x == "":
        n = 5
    else:
        n = int(x)

    ds = input(
        "do you want to input a series of digits? (for testing, default random): "
    )
    if ds == "":
        game = DigitParty(n=n, digits=None)
    else:
        game = DigitParty(
            n=n, digits=list(map(lambda s: int(s.strip()), ds.split(",")))
        )

    while not game.is_finished():
        print(game.show_board())
        print(f"current score: {game.score}")
        curr_digit, next_digit = game.next_digits()
        print(f"current digit: {curr_digit}")
        print(f"next digit: {next_digit}")
        print()
        coord = input(
            "give me 0-indexed row col coords from the top left to place the current"
            " digit (delimit with ','): "
        ).strip()
        print()

        try:
            rc = coord.split(",")[:2]
            r = int(rc[0])
            c = int(rc[1])
        except (ValueError, IndexError):
            print("can't read your coordinate input")
            continue

        try:
            game.place(r, c)
        except ValueError as e:
            print(str(e))

    print(game.show_board())
    print("game finished!")
    print(f"your score: {game.score}")
    print(f"theoretical max score: {game.theoretical_max_score()}")
    print(f"% of total: {100 * game.score / game.theoretical_max_score()}")


def computer_game(
    g: DigitParty,
    num_games: int,
    play: Callable[[DigitPartyState], DigitPartyPlacement],
) -> None:
    g.reset()
    while not g.is_finished():
        print(f"\n{g.show_board()}\n")
        curr_digit, next_digit = g.next_digits()
        print(f"current digit: {curr_digit}")
        print(f"next digit: {next_digit}")
        r, c = play(g.state())
        g.place(r, c)
        print(f"\ncomputer plays {curr_digit} at ({r}, {c})!")

    print(g.show_board())
    print("game finished!")
    print(f"computer score: {g.score}")
    print(f"theoretical max score: {g.theoretical_max_score()}")
    print(f"% of total: {100 * g.score / g.theoretical_max_score():.2f}")

    # run a lot of games to get some data on the performance
    g.reset()

    score = 0
    theoretical_max = 0
    percent_per_game = 0.0
    percentages = []
    for e in range(1, num_games + 1):
        while not g.is_finished():
            r, c = play(g.state())
            g.place(r, c)

        score += g.score
        t_max = g.theoretical_max_score()
        theoretical_max += t_max
        percent_per_game += g.score / t_max
        percentages.append(100 * g.score / t_max)
        g.reset()

        if e % 1000 == 0:
            print(f"Episode {e}/{num_games}")

    percent = score / theoretical_max
    print(f"played {num_games} games")
    print(f"achieved {100 * percent:.2f}% or {score}/{theoretical_max}")

    df = pd.DataFrame({"percentages": percentages})
    mean = df["percentages"].mean()
    median = df["percentages"].median()
    mode = df["percentages"].mode().values[0]

    print(f"averaged {mean:.2f}% of theoretical max")
    print(f"median: {median:.2f}%")
    print(f"mode: {mode:.2f}%")

    plt.hist(df, bins=50)
    plt.xticks(range(0, 101, 2))
    plt.locator_params(axis="x", nbins=100)
    plt.title("games played per percent score")
    plt.xlabel("percent score")
    plt.ylabel("number of games")
    plt.axvline(mean, color="r", linestyle="--", label="Mean")
    plt.axvline(median, color="g", linestyle="-", label="Median")
    plt.axvline(mode, color="b", linestyle="-", label="Mode")
    plt.legend(
        {
            f"Mean: {mean:.2f}%": mean,
            f"Median: {median:.2f}%": median,
            f"Mode: {mode:.2f}%": mode,
        }
    )
    plt.grid()
    plt.show()


class DigitPartyQLearner(SimpleQLearner[DigitPartyIR, DigitPartyPlacement]):
    def __init__(
        self, n: int, q_pickle: str = "", alpha=0.1, gamma=0.9, epsilon=0.1
    ) -> None:
        super().__init__(q_pickle, alpha, gamma, epsilon)
        self.n = n

    def default_action_q_values(self) -> dict[DigitPartyPlacement, float]:
        actions = {}
        for r in range(self.n):
            for c in range(self.n):
                actions[(r, c)] = 0.0
        return actions

    def get_actions_from_state(self, state: DigitPartyIR) -> List[DigitPartyPlacement]:
        r = len(state.board)
        c = len(state.board[0])
        return [
            (i, j) for i in range(r) for j in range(c) if Empty == state.board[i][j]
        ]


class DigitPartyQTrainer(DigitParty, Trainer):
    def __init__(
        self, player: DigitPartyQLearner, n: int = 5, digits: List[int] | None = None
    ) -> None:
        super().__init__(n, digits)
        self.player = player

    @override
    def train(self, episodes=10000) -> None:
        super().train(episodes)
        self.player.save_policy()

    def train_once(self) -> None:
        while not self.is_finished():
            curr_score = self.score
            ir = self.to_immutable(self.state())
            action = self.player.choose_action(ir)

            r, c = action
            self.place(r, c)
            new_score = self.score

            self.player.update_q_value(
                ir,
                action,
                new_score - curr_score,
                self.to_immutable(self.state()),
            )

        self.reset()


def q_trained_game(game_size: int, num_games: int) -> None:
    # there's too many states in default digit party, so naive q learning is inexhaustive and doesn't work well
    # we can train a 3x3 game reasonably well, but it's very memory inefficient, since it needs to keep track
    # of all possible digit party states. after 20 million games, the policy file is about 5 GB
    # for a 2x2 game, the result is trivially 100%
    q = DigitPartyQLearner(
        game_size,
        q_pickle=f"src/games/digit_party/q-{game_size}x{game_size}-test.pkl",
        epsilon=0.5,
    )
    g = DigitPartyQTrainer(player=q, n=game_size)
    g.train(episodes=0)

    def q_play(state: DigitPartyState) -> DigitPartyPlacement:
        return q.choose_action(g.to_immutable(state), exploit=True)

    computer_game(g, num_games, q_play)


# each digit party needs its own neural network I guess since the board is a different shape
# not sure if this even makes sense to parametrize since I think for LARGE boards the NN
# architecture would be very different, with more layers, etc
class DigitParty3x3NeuralNetwork(NeuralNetwork[DigitPartyState, Policy]):
    NUM_CHANNELS = 1
    DROPOUT_RATE = 0.5
    LEARN_RATE = 0.01
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
                Conv2D(filters=1, kernel_size=(2, 2), padding="valid")(board)
            )
        )
        conv2 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(filters=1, kernel_size=(2, 2), padding="valid")(conv1)
            )
        )
        flat = Flatten()(conv2)
        concat = Concatenate()([flat, input_curr_digit, input_next_digit])
        dense1 = Dropout(rate=self.DROPOUT_RATE)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(256)(concat)))
        )
        dense2 = Dropout(rate=self.DROPOUT_RATE)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(128)(dense1)))
        )

        # policy, guessing the value of each valid action at the input state
        pi = Dense(9, activation="softmax", name="pi")(dense2)
        # TODO: add an output representing the score of the input board?

        self.model = Model(
            inputs=[input_board, input_curr_digit, input_next_digit], outputs=[pi]
        )
        self.model.compile(
            loss=["categorical_crossentropy"],
            optimizer=Adam(learning_rate=self.LEARN_RATE),
            metrics=["accuracy"],
        )
        self.model.summary()

    def train(self, data: List[Tuple[DigitPartyState, Policy]]) -> None:
        inputs: List[DigitPartyState]
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
        self.model.fit(
            x=[input_boards, input_currs, input_nexts],
            y=policies,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            shuffle=True,
        )

    def predict(self, inputs: List[DigitPartyState]) -> List[Policy]:
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


def deep_q_3x3_trained_game():
    cur_dir = pathlib.Path(__file__).parent.resolve()
    nn = DigitParty3x3NeuralNetwork(model_folder=f"{cur_dir}/deepq_3x3_models/")
    deepq = DeepQLearner(
        DigitParty(n=3),
        nn,
        DeepQParameters(
            alpha=0.01,
            gamma=0.618,
            min_epsilon=0.5,
            max_epsilon=1,
            epsilon_decay=0.01,
            memory_size=100_000,
            min_replay_size=1000,
            minibatch_size=32,
            training_episodes=10000,  # 9 * eps total steps
            episodes_per_model_save=100,
            episodes_per_memory_save=100,
            steps_to_train_longterm=1000,  # (steps / 9) episodes before training longterm
            steps_to_train_shortterm=1,
            steps_per_target_update=1000,  # (steps / 9) episodes before updating target network
        ),
        memory_folder=f"{cur_dir}/deepq_3x3_memory/",
    )
    deepq.train()

    g = DigitParty(n=3)
    random = 0
    prediction = 0

    def deepq_play(state: DigitPartyState) -> DigitPartyPlacement:
        nonlocal random
        nonlocal prediction
        print(f"board:\n{state.board}")
        print(f"next: {state.next}")
        pi = nn.predict([state])
        print(f"pi: {pi}")
        action_statuses = np.asarray(g.actions(state))
        valid_actions = np.where(action_statuses == VALID)[0]
        a = np.argmax(pi * action_statuses)
        if not np.isin(valid_actions, a).any():
            # TODO: might be an issue with my model, not the implementation?
            # policy not robust enough, so when masked with action statuses it produces no valid actions
            a = np.random.choice(valid_actions)
            print(f"######## CHOSE RANDOM ACTION {a}")
            random += 1
        else:
            print(f"nn chose action {a}")
            prediction += 1

        n = state.board.shape[0]
        r = int(a / n)
        c = int(a % n)
        return r, c

    computer_game(g, 1000, deepq_play)
    print("random actions: ", random)
    print("predicted actions: ", prediction)


class DigitParty5x5NeuralNetwork(NeuralNetwork[DigitPartyState, Policy]):
    NUM_CHANNELS = 1
    DROPOUT_RATE = 0.5
    LEARN_RATE = 0.01
    BATCH_SIZE = 64
    EPOCHS = 10

    def __init__(self, model_folder: str) -> None:
        super().__init__(model_folder)

        input_board = Input(shape=(5, 5), name="dp_5x5_board")
        input_curr_digit = Input(shape=(1,), name="current_digit")
        input_next_digit = Input(shape=(1,), name="next_digit")
        # each layer is a 4D tensor consisting of: batch_size, board_height, board_width, num_channels
        board = Reshape((5, 5, self.NUM_CHANNELS))(input_board)
        # normalize along channels axis
        conv1 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(filters=1, kernel_size=(3, 3), padding="valid")(board)
            )
        )
        conv2 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(filters=1, kernel_size=(3, 3), padding="valid")(conv1)
            )
        )
        flat = Flatten()(conv2)
        concat = Concatenate()([flat, input_curr_digit, input_next_digit])
        dense1 = Dropout(rate=self.DROPOUT_RATE)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(256)(concat)))
        )
        dense2 = Dropout(rate=self.DROPOUT_RATE)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(128)(dense1)))
        )

        # policy, guessing the value of each valid action at the input state
        pi = Dense(25, activation="softmax", name="pi")(dense2)
        # TODO: add an output representing the score of the input board?

        self.model = Model(
            inputs=[input_board, input_curr_digit, input_next_digit], outputs=[pi]
        )
        self.model.compile(
            loss=["categorical_crossentropy"],
            optimizer=Adam(learning_rate=self.LEARN_RATE),
            metrics=["accuracy"],
        )
        self.model.summary()

    def train(self, data: List[Tuple[DigitPartyState, Policy]]) -> None:
        inputs: List[DigitPartyState]
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
        self.model.fit(
            x=[input_boards, input_currs, input_nexts],
            y=policies,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            shuffle=True,
        )

    def predict(self, inputs: List[DigitPartyState]) -> List[Policy]:
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


def deep_q_5x5_trained_game():
    cur_dir = pathlib.Path(__file__).parent.resolve()
    nn = DigitParty5x5NeuralNetwork(model_folder=f"{cur_dir}/deepq_5x5_models/")
    deepq = DeepQLearner(
        DigitParty(n=5),
        nn,
        DeepQParameters(
            alpha=0.001,
            gamma=0.618,
            min_epsilon=0.5,
            max_epsilon=1,
            epsilon_decay=0.01,
            memory_size=100_000,
            min_replay_size=1000,
            minibatch_size=32,
            training_episodes=3000,  # 25 * eps total steps
            episodes_per_model_save=100,
            steps_to_train_longterm=1000,  # (steps / 25) episodes before training longterm
            steps_to_train_shortterm=1,
            steps_per_target_update=1000,  # (steps / 25) episodes before updating target network
        ),
    )
    deepq.train()

    g = DigitParty(n=5)
    random = 0
    prediction = 0

    def deepq_play(state: DigitPartyState) -> DigitPartyPlacement:
        nonlocal random
        nonlocal prediction
        print(f"board:\n{state.board}")
        print(f"next: {state.next}")
        pi = nn.predict([state])
        print(f"pi: {pi}")
        action_statuses = np.asarray(g.actions(state))
        valid_actions = np.where(action_statuses == VALID)[0]
        a = np.argmax(pi * action_statuses)
        if not np.isin(valid_actions, a).any():
            # TODO: might be an issue with my model, not the implementation?
            # policy not robust enough, so when masked with action statuses it produces no valid actions
            a = np.random.choice(valid_actions)
            print(f"######## CHOSE RANDOM ACTION {a}")
            random += 1
        else:
            print(f"nn chose action {a}")
            prediction += 1

        n = state.board.shape[0]
        r = int(a / n)
        c = int(a % n)
        return r, c

    computer_game(g, 1000, deepq_play)
    print("random actions: ", random)
    print("predicted actions: ", prediction)
