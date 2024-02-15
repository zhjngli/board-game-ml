import os
import pickle
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Generic, List, Tuple, TypeVar

StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")


class MonteCarloLearner(Generic[StateType, ActionType], ABC):
    """
    A Monte Carlo Learner that uses default dictionaries to track the Q table. Works for simple games.
    Monte Carlo is needed in episodic environments, where rewards are only received at the end of the game.
    """

    def __init__(
        self, policy_file: str = "", alpha=0.2, gamma=0.9, epsilon=0.3
    ) -> None:
        self.policy_file = policy_file
        self.state_values: defaultdict[StateType, float] = defaultdict(float)
        if self.policy_file and os.path.isfile(self.policy_file):
            with open(self.policy_file, "rb") as file:
                self.state_values = pickle.load(file)
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # value decay rate
        self.epsilon = epsilon  # explore rate
        self.states: List[StateType] = []

    @abstractmethod
    def get_actions_from_state(self, state: StateType) -> List[ActionType]:
        """
        Gets a list of legal actions from a state
        """
        pass

    @abstractmethod
    def apply(self, state: StateType, action: ActionType) -> StateType:
        """
        Applies an action to a state and returns the next state
        """
        pass

    def choose_action(self, state: StateType, exploit: bool = False) -> ActionType:
        legal_actions = self.get_actions_from_state(state)
        if random.uniform(0, 1) < self.epsilon and not exploit:
            action = random.choice(legal_actions)
        else:
            # max() takes the first item if there are ties, so sometimes we can get stuck in a cycle of always choosing one action
            action_values: List[Tuple[ActionType, float]] = [
                # TODO: maybe default should be configurable, otherwise a default 0 sets a condition on the reward function
                (a, self.state_values.get(self.apply(state, a), 0))
                for a in legal_actions
            ]
            (_, best_q) = max(action_values, key=lambda x: x[1])
            best_actions: List[ActionType] = [
                a for (a, q) in action_values if q == best_q
            ]
            action = random.choice(best_actions)
        return action

    def propagate_reward(self, reward: float) -> None:
        for s in reversed(self.states):
            if self.state_values.get(s) is None:
                self.state_values[s] = 0
            self.state_values[s] += self.alpha * (
                self.gamma * reward - self.state_values[s]
            )
            reward = self.state_values[s]

    def add_state(self, state: StateType) -> None:
        self.states.append(state)

    def reset_states(self) -> None:
        self.states = []

    def save_policy(self) -> None:
        if self.policy_file:
            with open(self.policy_file, "wb") as file:
                pickle.dump(self.state_values, file)
