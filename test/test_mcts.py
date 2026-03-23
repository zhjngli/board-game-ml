import numpy as np

from games.game import P1
from games.ultimate_ttt.ultimate import UltimateState, UltimateTicTacToe
from learners.alpha_zero.monte_carlo_tree_search import (
    MCTSParameters,
    MonteCarloTreeSearch,
)


class MockNN:
    """Mock NN that returns uniform policy and 0 value."""

    def predict(self, inputs):
        n = UltimateTicTacToe.num_actions()
        return [type("Out", (), {"policy": np.ones(n) / n, "value": 0.0})()]

    def set_weights(self, w):
        pass

    def get_weights(self):
        return []


def test_mcts_handles_zero_reward_terminal():
    """MCTS should correctly identify terminal states even when reward is 0 (draw)."""
    params = MCTSParameters(
        num_searches=5,
        cpuct=1,
        epsilon=1e-4,
        dirichlet_alpha=0,
        dirichlet_epsilon=0,
    )
    game = UltimateTicTacToe()
    mcts = MonteCarloTreeSearch(game, MockNN(), params)

    # Create a filled board that's a draw (reward=0)
    b = np.asarray(
        [
            [
                [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                [[0, -1, 0], [0, -1, 0], [0, -1, 0]],
                [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            ],
            [
                [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]],
                [[1, -1, 1], [1, -1, 1], [-1, 1, -1]],
                [[0, 0, -1], [0, 0, -1], [0, 0, -1]],
            ],
            [
                [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                [[0, -1, 0], [0, -1, 0], [0, -1, 0]],
                [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            ],
        ]
    )
    state = UltimateState(board=b, player=P1, active_nonant=None)
    assert UltimateTicTacToe.check_finished(state)

    # search should return the negated reward (0) without crashing
    result = mcts.search(state)
    assert result == 0  # -0 == 0
