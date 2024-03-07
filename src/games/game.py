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
Immutable = TypeVar("Immutable")  # TODO: enforce hashable/immutable

ActionStatus = Literal[1, 0]
VALID: ActionStatus = 1
INVAL: ActionStatus = 0

# TODO: new type for certain reward values?
P1WIN = 1
P2WIN = -1


def switch_player(p: Player) -> Player:
    return P1 if p == P2 else P2


class Game(ABC, Generic[State, Immutable]):
    @abstractmethod
    def reset(self) -> None:
        """
        Resets the game.
        """
        pass

    @abstractmethod
    def state(self) -> State:
        """
        Returns the current state of the game.
        """
        pass

    @staticmethod
    @abstractmethod
    def to_immutable(state: State) -> Immutable:
        """
        Returns an immutable (hashable) representation of the given game state.
        """
        pass

    @abstractmethod
    def num_actions(self) -> int:
        """
        The number of possible actions for a given game.
        For games with a board like tic tac toe, this is straightforward.
        For games such as photosynthesis, we need to have a mapping between the action and the move.

        For some games, this can be a static method, e.g. TicTacToe. The board is always constant.
        For games like Digit Party, this won't be a static method since it depends on game construction (rows and cols)
        """
        pass

    @staticmethod
    @abstractmethod
    def actions(state: State) -> List[ActionStatus]:
        """
        Get all the actions available at the given state. 1 represents a valid action and 0 represents invalid.
        """
        pass

    @staticmethod
    @abstractmethod
    def apply(state: State, action: Action) -> State:
        """
        Apply an action to the given state, and return the new state
        """
        pass

    @staticmethod
    @abstractmethod
    def check_finished(state: State) -> bool:
        """
        Checks whether the given state is a finished game.
        """
        pass

    @staticmethod
    @abstractmethod
    def calculate_reward(state: State) -> float:
        """
        A reward to pass to the agent at the given state.
        """
        pass

    @staticmethod
    @abstractmethod
    def orient_state(state: State) -> State:
        """
        Returns a state that is from the perspective of player 1, no matter the input state.
        This helps with a constant input for training the neural network.
        """
        pass

    @staticmethod
    @abstractmethod
    def symmetries_of(a: NDArray) -> List[NDArray]:
        """
        List the symmetries of the given board or policy.
        """
        pass
