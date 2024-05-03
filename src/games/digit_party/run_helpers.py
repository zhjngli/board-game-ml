from typing import Callable

import pandas as pd
from matplotlib import pyplot as plt

from games.digit_party.game import DigitParty, DigitPartyPlacement, DigitPartyState


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
