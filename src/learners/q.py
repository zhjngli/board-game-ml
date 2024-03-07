import os
import pickle
import random
from collections import defaultdict
from typing import Generic, List, NamedTuple, Tuple

from games.game import VALID, Action, Game, Immutable, State


class SimpleQParameters(NamedTuple):
    alpha: float  # learning rate
    gamma: float  # reward decay rate
    epsilon: float  # explore rate


DefaultQParams = SimpleQParameters(alpha=0.1, gamma=0.9, epsilon=0.1)


class SimpleQLearner(Generic[State, Immutable]):
    """
    A default Q Learner that uses default dictionaries to track the Q table. Works for simple games.
    For complex games, we probably need to serialize the states/actions so learning can be more efficient.
    """

    def __init__(
        self,
        game: Game[State, Immutable],
        q_pickle: str = "",
        params: SimpleQParameters = DefaultQParams,
    ) -> None:
        self.game = game

        self.q_pickle = q_pickle
        self.q_table: defaultdict[Immutable, dict[Action, float]] = defaultdict(
            self.default_action_q_values  # TODO: pickle can't find this if refactored
        )
        if self.q_pickle and os.path.isfile(self.q_pickle):
            with open(self.q_pickle, "rb") as file:
                self.q_table = pickle.load(file)

        self.alpha = params.alpha
        self.gamma = params.gamma
        self.epsilon = params.epsilon

    def default_action_q_values(self) -> dict[Action, float]:
        """
        Gets an enumeration of all actions to a default q-value of 0.0. Assumes that the possible actions are always the same at each state.
        """
        d = {}
        for a in range(self.game.num_actions()):
            d[a] = 0.0
        return d

    def choose_action(self, ir: Immutable, exploit: bool = False) -> Action:
        legal_actions = self.get_actions_from_state(ir)
        if random.uniform(0, 1) < self.epsilon and not exploit:
            return random.choice(legal_actions)
        else:
            # max() takes the first item if there are ties, so sometimes we can get stuck in a cycle of always choosing one action
            actions_q_vals: List[Tuple[Action, float]] = [
                (a, q) for (a, q) in self.q_table[ir].items() if a in legal_actions
            ]
            (_, best_q) = max(actions_q_vals, key=lambda x: x[1])
            best_actions: List[Action] = [a for (a, q) in actions_q_vals if q == best_q]
            return random.choice(best_actions)

    def update_q_value(
        self, ir: Immutable, action: Action, reward: float, next_ir: Immutable
    ) -> None:
        next_actions = self.get_actions_from_state(next_ir)
        if next_actions:
            random_action = random.choice(self.get_actions_from_state(next_ir))
            best_next_action = max(
                self.q_table[next_ir],
                key=self.q_table[next_ir].__getitem__,
                default=random_action,
            )
            next_q_value = self.q_table[next_ir][best_next_action]
        else:
            # if next_state is a terminal state (game end), then the best next q value is...?
            # TODO: does 0.0 work? if so, what does it say about how the reward function should be structured?
            next_q_value = 0.0
        current_q_value = self.q_table[ir][action]
        self.q_table[ir][action] = (1 - self.alpha) * current_q_value + self.alpha * (
            reward + self.gamma * next_q_value
        )

    def get_actions_from_state(self, ir: Immutable) -> List[Action]:
        """
        Gets a list of legal actions from a state
        """
        legals = []
        actions = self.game.actions(self.game.from_immutable(ir))
        for a, status in enumerate(actions):
            if status == VALID:
                legals.append(a)
        return legals

    def save_policy(self) -> None:
        if self.q_pickle:
            with open(self.q_pickle, "wb") as file:
                pickle.dump(self.q_table, file)
