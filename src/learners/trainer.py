from abc import ABC, abstractmethod


class Trainer(ABC):
    def train(self, episodes: int) -> None:
        for e in range(1, episodes + 1):
            self.train_once()

            if e % 1000 == 0:
                print(f"Episode {e}/{episodes}")

    @abstractmethod
    def train_once(self) -> None:
        """
        One episode of training your agent
        """
        pass
