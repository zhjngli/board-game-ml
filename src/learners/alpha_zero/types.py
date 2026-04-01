from typing import NamedTuple, TypeVar

from numpy.typing import NDArray

from games.game import Board

# Generic NN input type — each game defines its own
# TODO: could use a stronger bound, e.g. TypeVar("NNInput", bound=NDArray)
NNInput = TypeVar("NNInput")

# common output types
Policy = NDArray
Value = float


# Deprecated: kept for backward pickle compatibility with old .pkl training files.
# New code should use game-specific types via Game.to_nn_input().
class A0NNInput(NamedTuple):
    board: Board


class A0NNOutput(NamedTuple):
    policy: Policy
    value: Value
