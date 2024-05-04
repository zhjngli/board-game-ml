import argparse

from games.digit_party import train_deep as digit_party_deep
from games.digit_party import train_q_deep as digit_party_deep_q
from games.digit_party import train_q_simple as digit_party_simple_q
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
        "digit_party_simple_q",
        aliases=["dpq", "dp_simple_q"],
        help="run digit party with simple q learning",
    )
    sub_digit_party.set_defaults(function=digit_party_simple_q.main)

    sub_digit_party_deep = subparsers.add_parser(
        "digit_party_deep",
        aliases=["dpd", "dp_deep"],
        help="run digit party with deep learning (NOT q learning. uses data from simple q learning to train)",
    )
    sub_digit_party_deep.set_defaults(function=digit_party_deep.main)

    sub_digit_party_deep_q = subparsers.add_parser(
        "digit_party_deep_q",
        aliases=["dpdq", "dp_deep_q"],
        help="run digit party with deep q learning",
    )
    sub_digit_party_deep_q.set_defaults(function=digit_party_deep_q.main)

    return parser


def run() -> None:
    parser = _define_parser()
    args = parser.parse_args()
    args.function()
