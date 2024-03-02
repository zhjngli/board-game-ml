from abc import ABC, abstractmethod
from typing import Generic, List, Tuple

from numpy.typing import NDArray

from games.game import Board, State

Policy = NDArray  # TODO: one dimensional NDArray of arbitrary length
Value = float


class NeuralNetwork(ABC, Generic[State]):
    @abstractmethod
    def train(self, data: List[Tuple[Board, Policy, Value]]) -> None:
        pass

    @abstractmethod
    def predict(self, state: State) -> Tuple[Policy, Value]:
        pass

    @abstractmethod
    def save(self, file: str) -> None:
        pass

    @abstractmethod
    def load(self, file: str) -> None:
        pass
