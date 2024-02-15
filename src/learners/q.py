import os
import pickle
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Generic, List, Tuple, TypeVar

StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")


class SimpleQLearner(Generic[StateType, ActionType], ABC):
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
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # reward decay rate
        self.epsilon = epsilon  # explore rate

    @abstractmethod
    def default_action_q_values(self) -> dict[ActionType, float]:
        """
        Gets an enumeration of all actions to a default q-value (probably 0.0). Assumes that the possible actions are always the same at each state.
        """
        pass

    # TODO: maybe extract choose_action to a learner or agent class
    def choose_action(self, state: StateType, exploit: bool = False) -> ActionType:
        legal_actions = self.get_actions_from_state(state)
        if random.uniform(0, 1) < self.epsilon and not exploit:
            return random.choice(legal_actions)
        else:
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
        next_actions = self.get_actions_from_state(next_state)
        if next_actions:
            random_action = random.choice(self.get_actions_from_state(next_state))
            best_next_action = max(
                self.q_table[next_state],
                key=self.q_table[next_state].__getitem__,
                default=random_action,
            )
            next_q_value = self.q_table[next_state][best_next_action]
        else:
            # if next_state is a terminal state (game end), then the best next q value is...?
            # TODO: does 0.0 work? if so, what does it say about how the reward function should be structured?
            next_q_value = 0.0
        current_q_value = self.q_table[state][action]
        self.q_table[state][action] = (
            1 - self.alpha
        ) * current_q_value + self.alpha * (reward + self.gamma * next_q_value)

    @abstractmethod
    def get_actions_from_state(self, state: StateType) -> List[ActionType]:
        """
        Gets a list of legal actions from a state
        """
        pass

    def save_policy(self) -> None:
        if self.q_pickle:
            with open(self.q_pickle, "wb") as file:
                pickle.dump(self.q_table, file)
