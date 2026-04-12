import numpy as np

from games.game import P1
from games.tictactoe.tictactoe import TicTacToe, TicTacToeState


def test_to_nn_input_returns_board():
    board = np.array([[1, 0, -1], [0, 1, 0], [-1, 0, 1]])
    state = TicTacToeState(board=board, player=P1)
    nn_input = TicTacToe.to_nn_input(state)
    np.testing.assert_array_equal(nn_input, board)


def test_training_symmetries_includes_identity():
    board = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    state = TicTacToeState(board=board, player=P1)
    nn_input = TicTacToe.to_nn_input(state)
    policy = np.array([0.5, 0.1, 0.1, 0.1, 0.0, 0.05, 0.05, 0.05, 0.05])

    syms = TicTacToe.training_symmetries(nn_input, policy)
    # Should have 8 symmetries (4 rotations x 2 mirror states, rot by 4 = identity)
    assert len(syms) == 8

    # Check identity is present (rotation by 4 = identity)
    found_identity = False
    for inp, pol in syms:
        if np.array_equal(inp, nn_input) and np.array_equal(pol, policy):
            found_identity = True
            break
    assert found_identity


def test_training_symmetries_preserves_shape():
    board = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    state = TicTacToeState(board=board, player=P1)
    nn_input = TicTacToe.to_nn_input(state)
    policy = np.array([0.5, 0.1, 0.1, 0.1, 0.0, 0.05, 0.05, 0.05, 0.05])

    syms = TicTacToe.training_symmetries(nn_input, policy)
    for inp, pol in syms:
        assert inp.shape == (3, 3)
        assert pol.shape == (9,)
