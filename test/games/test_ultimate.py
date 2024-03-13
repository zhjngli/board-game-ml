import numpy as np

from games.game import P1, P2
from games.ultimate_ttt.ultimate import UltimateState, UltimateTicTacToe


def test_tie():
    b = np.asarray(
        [
            [
                [[1, 0, 0], [1, 0, 0], [1, 0, 0]],  # <- this line is nonant 1
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
    assert not UltimateTicTacToe._is_win(P1, b)
    assert not UltimateTicTacToe._is_win(P2, b)
    assert UltimateTicTacToe._is_board_filled(b)
    assert UltimateTicTacToe.check_finished(state)


def test_p1_3_in_a_row():
    b = np.asarray(
        [
            [
                [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                [[0, -1, 0], [0, -1, 0], [0, -1, 0]],
                [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            ],
            [
                [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]],
                [[1, -1, 1], [-1, -1, 1], [-1, 1, 1]],
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
    assert UltimateTicTacToe._3_in_a_row(P1, b)
    assert UltimateTicTacToe._is_win(P1, b)
    assert not UltimateTicTacToe._is_win(P2, b)
    assert UltimateTicTacToe._is_board_filled(b)
    assert UltimateTicTacToe.check_finished(state)


def test_p2_win_by_mini_wins():
    b = np.asarray(
        [
            [
                [[1, -1, 1], [1, -1, 1], [-1, 1, -1]],
                [[1, -1, 1], [1, -1, 1], [-1, 1, -1]],
                [[1, -1, 1], [1, -1, 1], [-1, 1, -1]],
            ],
            [
                [[1, -1, 1], [1, -1, 1], [-1, 1, -1]],
                [[1, -1, 1], [1, -1, 1], [-1, 1, -1]],
                [[1, -1, 1], [1, -1, 1], [-1, 1, -1]],
            ],
            [
                [[1, -1, 1], [1, -1, 1], [-1, 1, -1]],
                [[0, -1, 0], [0, -1, 0], [0, -1, 0]],  # <- the only p2 win in nonant 8
                [[1, -1, 1], [1, -1, 1], [-1, 1, -1]],
            ],
        ]
    )
    state = UltimateState(board=b, player=P1, active_nonant=None)
    assert not UltimateTicTacToe._is_win(P1, b)
    assert UltimateTicTacToe._is_win(P2, b)
    assert not UltimateTicTacToe._3_in_a_row(P2, b)
    assert UltimateTicTacToe._is_board_filled(b)
    assert UltimateTicTacToe.check_finished(state)
