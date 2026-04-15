from types import SimpleNamespace
from typing import Protocol, cast

from games.digit_party.train_q_deep_3x3 import eval_3x3_best, train_3x3_dqn
from games.digit_party.train_q_deep_5x5 import eval_5x5_best, train_5x5_dqn


class DeepQCLIArgs(Protocol):
    size: str
    mode: str
    games: int
    episodes: int | None


deep_q_3x3_trained_game = train_3x3_dqn
best_model_game = eval_3x3_best


def main(args: DeepQCLIArgs | None = None) -> None:
    if args is None:
        args = cast(
            DeepQCLIArgs,
            SimpleNamespace(size="3", mode="eval-best", games=1000, episodes=None),
        )

    if args.size == "3":
        if args.mode == "train":
            train_3x3_dqn(episodes=args.episodes)
        else:
            eval_3x3_best(games=args.games)
        return

    if args.mode == "train":
        train_5x5_dqn(episodes=args.episodes)
    else:
        eval_5x5_best(games=args.games)
