import argparse

from games.digit_party import deep_train as digit_party_exp
from games.digit_party import run as digit_party
from games.random_walk import random_walk
from games.tictactoe import run as tictactoe
from games.ultimate_ttt import run as ultimate_ttt


def _define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="board_game_ml",
        description="Training computer agents to play simple board games",
    )
    subparsers = parser.add_subparsers()

    sub_random_walk = subparsers.add_parser(
        "random_walk", aliases=["rw"], help="run random walk"
    )
    sub_random_walk.set_defaults(function=random_walk.main)

    sub_tictactoe = subparsers.add_parser(
        "tictactoe", aliases=["ttt"], help="run tic tac toe"
    )
    sub_tictactoe.set_defaults(function=tictactoe.main)

    sub_ultimate_ttt = subparsers.add_parser(
        "ultimate_ttt", aliases=["u", "ult"], help="run ultiamte tic tac toe"
    )
    sub_ultimate_ttt.set_defaults(function=ultimate_ttt.main)

    sub_digit_party = subparsers.add_parser(
        "digit_party", aliases=["dp"], help="run digit party"
    )
    sub_digit_party.set_defaults(function=digit_party.main)

    sub_digit_party_exp = subparsers.add_parser(
        "digit_party_experimental",
        aliases=["dpe", "dp_exp"],
        help="run digit party experimental",
    )
    sub_digit_party_exp.set_defaults(function=digit_party_exp.main)

    return parser


def run() -> None:
    parser = _define_parser()
    args = parser.parse_args()
    args.function()
