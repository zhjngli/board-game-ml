from abc import ABC, abstractmethod
from typing import Deque, Generic, List, Tuple

from numpy.typing import NDArray

from games.game import Board, State

Policy = NDArray  # TODO: one dimensional NDArray of arbitrary length
Value = float


class NeuralNetwork(ABC, Generic[State]):
    @abstractmethod
    def train(self, data: List[Deque[Tuple[Board, Policy, float]]]):
        pass

    @abstractmethod
    def predict(self, state: State) -> Tuple[Policy, Value]:
        pass

    @abstractmethod
    def save(self, path: str, file: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str, file: str) -> None:
        pass
