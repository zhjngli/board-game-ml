from typing import NamedTuple

from numpy.typing import NDArray

from games.game import Board

# common output types
Policy = NDArray  # TODO: one dimensional NDArray of arbitrary length
Value = float


# TODO: in the current scheme, (Board, Policy, Value), only captures the type of games s.t.
# the neural network input is the board, and the output is the policy and value.
# Other games can potentially have other input that's not fully captured in the board.
# We should be able to extend input/output to capture state relevant to the specific game.
class A0NNInput(NamedTuple):
    board: Board


class A0NNOutput(NamedTuple):
    policy: Policy
    value: Value
