import os
import pickle
import random
from abc import abstractmethod
from collections import defaultdict
from typing import Generic, List, Tuple, TypeVar

StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")


class DefaultQLearner(Generic[StateType, ActionType]):
    """
    A default Q Learner that uses default dictionaries to track the Q table. Works for simple games.
    For complex games, we probably need to serialize the states/actions so learning can be more efficient.
    """

    def __init__(self, q_pickle: str = "", alpha=0.1, gamma=0.9, epsilon=0.1) -> None:
        self.q_pickle = q_pickle
        self.q_table: defaultdict[StateType, dict[ActionType, float]] = defaultdict(
            self.default_action_q_values
        )
        if self.q_pickle and os.path.isfile(self.q_pickle):
            with open(self.q_pickle, "rb") as file:
                self.q_table = pickle.load(file)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    @abstractmethod
    def default_action_q_values(self) -> dict[ActionType, float]:
        """
        Gets an enumeration of all actions to a default q-value (probably 0.0). Assumes that the possible actions are always the same at each state.
        """
        pass

    def choose_action(self, state: StateType, exploit: bool = False) -> ActionType:
        if random.uniform(0, 1) < self.epsilon and not exploit:
            actions = self.get_actions_from_state(state)
            return random.choice(actions)
        else:
            legal_actions = self.get_actions_from_state(state)

            # max() takes the first item if there are ties, so sometimes we can get stuck in a cycle of always choosing one action
            actions_q_vals: List[Tuple[ActionType, float]] = [
                (a, q) for (a, q) in self.q_table[state].items() if a in legal_actions
            ]
            (_, best_q) = max(actions_q_vals, key=lambda x: x[1])
            best_actions: List[ActionType] = [
                a for (a, q) in actions_q_vals if q == best_q
            ]
            return random.choice(best_actions)

    def update_q_value(
        self, state: StateType, action: ActionType, reward: float, next_state: StateType
    ) -> None:
        random_action = random.choice(self.get_actions_from_state(state))
        best_next_action = max(
            self.q_table[next_state],
            key=self.q_table[next_state].__getitem__,
            default=random_action,
        )
        current_q_value = self.q_table[state][action]
        next_q_value = self.q_table[next_state][best_next_action]
        self.q_table[state][action] = (
            1 - self.alpha
        ) * current_q_value + self.alpha * (reward + self.gamma * next_q_value)

    @abstractmethod
    def get_actions_from_state(self, state: StateType) -> List[ActionType]:
        """
        Gets a list of legal actions from a state
        """
        pass

    def train(self, episodes: int = 1000) -> None:
        for e in range(1, episodes + 1):
            self.train_once()

            if e % 100 == 0:
                print(f"Episode {e}/{episodes}")

        if self.q_pickle:
            with open(self.q_pickle, "wb") as file:
                pickle.dump(self.q_table, file)

    @abstractmethod
    def train_once(self) -> None:
        """
        Override this function for the specific game you want to train on!
        """
        pass
