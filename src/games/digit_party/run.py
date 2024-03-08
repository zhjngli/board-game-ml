from typing import List

import pandas as pd
from matplotlib import pyplot as plt
from typing_extensions import override

from games.digit_party.game import DigitParty, DigitPartyIR, DigitPartyPlacement, Empty
from learners.q import SimpleQLearner
from learners.trainer import Trainer


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


def q_trained_game(game_size: int) -> None:
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

    while not g.is_finished():
        print(f"\n{g.show_board()}\n")
        curr_digit, next_digit = g.next_digits()
        print(f"current digit: {curr_digit}")
        print(f"next digit: {next_digit}")
        r, c = q.choose_action(g.to_immutable(g.state()), exploit=True)
        g.place(r, c)
        print(f"\ncomputer plays {curr_digit} at ({r}, {c})!")

    print(g.show_board())
    print("game finished!")
    print(f"computer score: {g.score}")
    print(f"theoretical max score: {g.theoretical_max_score()}")
    print(f"% of total: {100 * g.score / g.theoretical_max_score():.2f}")

    # run a lot of games to get some data on the performance
    games = 100_000
    g.reset()

    score = 0
    theoretical_max = 0
    percent_per_game = 0.0
    percentages = []
    for e in range(1, games + 1):
        while not g.is_finished():
            r, c = q.choose_action(g.to_immutable(g.state()), exploit=True)
            g.place(r, c)

        score += g.score
        t_max = g.theoretical_max_score()
        theoretical_max += t_max
        percent_per_game += g.score / t_max
        percentages.append(100 * g.score / t_max)
        g.reset()

        if e % 1000 == 0:
            print(f"Episode {e}/{games}")

    percent = score / theoretical_max
    print(f"played {games} games")
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
    plt.title("games played per percent score (untrained)")
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
