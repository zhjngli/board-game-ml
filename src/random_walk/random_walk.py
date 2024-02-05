import pickle
from enum import Enum, auto
from typing import List

from q_learners.default import DefaultQLearner


class Action(Enum):
    L = auto()
    R = auto()
    N = auto()


class RandomWalk:
    """
    A random walk game where the goal is to score some number of points.
    At the right bound, 1 point is scored, at the left bound, 1 point is lost.
    It's also possible to just stand there, infinitely accumulating points if applicable.
    """

    def __init__(self, left=-6, right=6, goal=10) -> None:
        self.i = 0
        self.l_bound = left
        self.r_bound = right
        self.goal = goal
        self.score = 0

    def step(self, action: Action) -> None:
        if action == Action.L:
            self.i -= 1
        elif action == Action.R:
            self.i += 1

        if self.at_left():
            self.score -= 1
        elif self.at_right():
            self.score += 1

    def at_left(self) -> bool:
        return self.i == self.l_bound

    def at_right(self) -> bool:
        return self.i == self.r_bound

    def get_state(self) -> int:
        return self.i

    def finished(self) -> bool:
        return self.score >= self.goal

    def show(self) -> str:
        return f"random walk at {self.i} with score {self.score}"


class RandomWalkQLearner(DefaultQLearner[int, Action]):
    def default_action_q_values(self) -> dict[Action, float]:
        actions = {}
        for a in Action:
            actions[a] = 0.0
        return actions

    def get_actions_from_state(self, state: int) -> List[Action]:
        return [Action.L, Action.R, Action.N]

    def train_once(self) -> None:
        w = RandomWalk()

        while not w.finished():
            state = w.get_state()

            action = self.choose_action(state)
            w.step(action)

            new_state = w.get_state()

            # reward minimizing distance to right bound
            new_r_dist = abs(w.r_bound - new_state)
            old_r_dist = abs(w.r_bound - state)
            reward = old_r_dist - new_r_dist

            if w.at_right():
                reward += 1

            self.update_q_value(state, action, reward, new_state)


def trained_game() -> None:
    pkl_file = "src/random_walk/q.pkl"
    q = RandomWalkQLearner(epsilon=0.5, q_pickle=pkl_file)
    q.train(episodes=1000)
    g = RandomWalk()

    while not g.finished():
        print(f"\n{g.show()}\n")

        action = q.choose_action(g.get_state(), exploit=True)
        g.step(action)

        print(f"computer takes {action} action")

    print(f"\nrandom walk ended! {g.show()}\n")

    with open(pkl_file, "rb") as file:
        q_table = pickle.load(file)
        for i in range(g.l_bound, g.r_bound + 1):
            print("state: ", i)
            for a, r in q_table[i].items():
                print(a, " has reward ", r)
            print()
