import numpy as np

from games.game import P1, P2
from games.ultimate_ttt.ultimate import UltimateState, UltimateTicTacToe


def test_unfinished():
    b = np.asarray(
        [
            [  # technically p1 has 2 mini wins, but board not filled
                [[1, 0, 0], [1, 0, 0], [1, 0, 0]],  # <- this line is nonant 1
                [[0, -1, 0], [0, -1, 0], [0, -1, 0]],
                [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            ],
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
        ]
    )
    state = UltimateState(board=b, player=P1, active_nonant=None)
    assert not UltimateTicTacToe._is_win(P1, b)
    assert not UltimateTicTacToe._is_win(P2, b)
    assert not UltimateTicTacToe._is_board_filled(b)
    assert not UltimateTicTacToe.check_finished(state)


def test_tie():
    b = np.asarray(
        [
            [
                [[1, 0, 0], [1, 0, 0], [1, 0, 0]],  # p1 win
                [[0, -1, 0], [0, -1, 0], [0, -1, 0]],  # p2 win
                [[0, 0, 1], [0, 0, 1], [0, 0, 1]],  # p1 win
            ],
            [
                [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]],  # p2 win
                [[1, -1, 1], [1, -1, 1], [-1, 1, -1]],  # tie
                [[0, 0, -1], [0, 0, -1], [0, 0, -1]],  # p2 win
            ],
            [
                [[1, 0, 0], [1, 0, 0], [1, 0, 0]],  # p1 win
                [[0, -1, 0], [0, -1, 0], [0, -1, 0]],  # p2 win
                [[0, 0, 1], [0, 0, 1], [0, 0, 1]],  # p1 win
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


def test_p2_3_in_a_row():
    b = np.asarray(
        [
            [
                [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                [[0, -1, 0], [0, -1, 0], [0, -1, 0]],
                [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            ],
            [
                [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]],
                [[-1, 1, -1], [1, 1, -1], [1, -1, -1]],
                [[0, 0, -1], [0, 0, -1], [0, 0, -1]],
            ],
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
        ]
    )
    state = UltimateState(board=b, player=P1, active_nonant=None)
    assert not UltimateTicTacToe._3_in_a_row(P1, b)
    assert not UltimateTicTacToe._is_win(P1, b)
    assert UltimateTicTacToe._3_in_a_row(P2, b)
    assert UltimateTicTacToe._is_win(P2, b)
    assert not UltimateTicTacToe._is_board_filled(b)
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


def test_to_nn_input_shape():
    b = np.zeros((3, 3, 3, 3))
    state = UltimateState(board=b, player=P1, active_nonant=None)
    nn_input = UltimateTicTacToe.to_nn_input(state)
    assert nn_input.shape == (9, 9, 2)


def test_to_nn_input_spatial_layout():
    """board[R][C][r][c] should map to grid[R*3+r][C*3+c] in channel 0."""
    b = np.zeros((3, 3, 3, 3))
    b[0][0][2][1] = 1  # section (0,0), cell (2,1)
    b[1][2][0][0] = -1  # section (1,2), cell (0,0)
    state = UltimateState(board=b, player=P1, active_nonant=None)
    nn_input = UltimateTicTacToe.to_nn_input(state)
    # channel 0: board
    assert nn_input[0 * 3 + 2][0 * 3 + 1][0] == 1  # (2, 1) in 9x9
    assert nn_input[1 * 3 + 0][2 * 3 + 0][0] == -1  # (3, 6) in 9x9


def test_to_nn_input_active_nonant_mask():
    b = np.zeros((3, 3, 3, 3))
    state = UltimateState(board=b, player=P1, active_nonant=(1, 2))
    nn_input = UltimateTicTacToe.to_nn_input(state)
    mask = nn_input[:, :, 1]
    # Only section (1,2) should be 1: rows 3-5, cols 6-8
    assert mask[3][6] == 1
    assert mask[5][8] == 1
    assert mask[0][0] == 0  # section (0,0) should be 0
    assert mask[4][4] == 0  # section (1,1) should be 0
    assert np.sum(mask) == 9  # exactly one 3x3 section lit up


def test_to_nn_input_free_choice_mask():
    b = np.zeros((3, 3, 3, 3))
    state = UltimateState(board=b, player=P1, active_nonant=None)
    nn_input = UltimateTicTacToe.to_nn_input(state)
    mask = nn_input[:, :, 1]
    assert np.all(mask == 1)  # all 1s when free choice


def test_draw_reward_is_zero():
    """Draws should return 0, not 0.1."""
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
    assert UltimateTicTacToe.calculate_reward(state) == 0
