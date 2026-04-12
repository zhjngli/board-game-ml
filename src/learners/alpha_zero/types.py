from typing import NamedTuple

from numpy.typing import NDArray

from games.game import Board

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
