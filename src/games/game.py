from abc import ABC, abstractmethod
from typing import Generic, List, Literal, TypeVar

from numpy.typing import NDArray

Board = NDArray
Player = Literal[1, -1]
P1: Player = 1
P2: Player = -1


class BasicState:
    """
    A basic state for 2 player games. Easily extensible with other fields.
    """

    def __init__(self, board: Board, player: Player) -> None:
        self.board = board
        self.player = player


State = TypeVar("State", bound=BasicState)
Action = int
ImmutableRepresentation = TypeVar("ImmutableRepresentation")  # TODO: enforce hashable

ActionStatus = Literal[1, 0]
VALID: ActionStatus = 1
INVAL: ActionStatus = 0

# TODO: new type for certain reward values?
P1WIN = 1
P2WIN = -1


def switch_player(p: Player) -> Player:
    return P1 if p == P2 else P2


class Game(ABC, Generic[State, ImmutableRepresentation]):
    @abstractmethod
    def immutable_representation(self, state: State) -> ImmutableRepresentation:
        """
        Returns an immutable (hashable) representation of the given game state.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def state(self) -> State:
        pass

    @abstractmethod
    def oriented_state(self, state: State) -> State:
        """
        Returns a state that is from the perspective of player 1, no matter the input state.
        This helps with a constant input for training the neural network.
        """
        pass

    @abstractmethod
    def finished(self, state: State) -> bool:
        pass

    @abstractmethod
    def reward(self, state: State) -> float:
        """
        A reward to pass to the agent at the given state.
        """
        pass

    @abstractmethod
    def num_actions(self) -> int:
        """
        The number of possible actions for a given game.
        For games with a board like tic tac toe, this is straightforward.
        For games such as photosynthesis, we need to have a mapping between the action and the move.
        """
        pass

    @abstractmethod
    def actions(self, state: State) -> List[ActionStatus]:
        """
        Get all the actions available at the given state. 1 represents a valid action and 0 represents invalid.
        """
        pass

    @abstractmethod
    def apply(self, state: State, action: Action) -> State:
        """
        Apply an action to the given state.
        """
        pass

    @abstractmethod
    def symmetries(self, a: NDArray) -> List[NDArray]:
        """
        List the symmetries of the current board or policy.
        """
        pass
