from typing import List

from typing_extensions import override

from games.digit_party.game import (
    DigitParty,
    DigitPartyIR,
    DigitPartyPlacement,
    DigitPartyState,
    Empty,
)
from games.digit_party.run_helpers import computer_game, human_game
from learners.q import SimpleQLearner
from learners.trainer import Trainer


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


def q_trained_game(game_size: int, num_episodes: int, num_games: int) -> None:
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
    g.train(episodes=num_episodes)

    def q_play(state: DigitPartyState) -> DigitPartyPlacement:
        return q.choose_action(g.to_immutable(state), exploit=True)

    computer_game(g, num_games, q_play)


def main() -> None:
    human_game()
    q_trained_game(game_size=3, num_episodes=20_000_000, num_games=100_000)
