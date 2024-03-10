from abc import ABC, abstractmethod
from typing import Generic, List, Tuple, TypeVar

State = TypeVar("State")
Input = TypeVar("Input")
Output = TypeVar("Output")


class NeuralNetwork(ABC, Generic[State, Input, Output]):
    def __init__(self, model_folder: str) -> None:
        self.model_folder = model_folder

    @abstractmethod
    def train(self, data: List[Tuple[Input, Output]]) -> None:
        pass

    @abstractmethod
    def predict(self, inputs: List[State]) -> List[Output]:
        pass

    @abstractmethod
    def save(self, file: str) -> None:
        pass

    @abstractmethod
    def load(self, file: str) -> None:
        pass

    @abstractmethod
    def set_weights(self, weights) -> None:
        pass

    @abstractmethod
    def get_weights(self):
        pass
