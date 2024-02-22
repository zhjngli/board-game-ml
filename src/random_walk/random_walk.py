import pickle
from enum import Enum, auto
from typing import List

from typing_extensions import override

from learners.monte_carlo import MonteCarloLearner
from learners.q import SimpleQLearner
from learners.trainer import Trainer


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

    def reset(self) -> None:
        self.score = 0
        self.i = 0

    @staticmethod
    def apply(state: int, action: Action) -> int:
        delta = 0
        if action == Action.L:
            delta = -1
        elif action == Action.R:
            delta = 1

        return state + delta

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


class RandomWalkQLearner(SimpleQLearner[int, Action]):
    def default_action_q_values(self) -> dict[Action, float]:
        actions = {}
        for a in Action:
            actions[a] = 0.0
        return actions

    def get_actions_from_state(self, state: int) -> List[Action]:
        return [Action.L, Action.R, Action.N]


class RandomWalkQTrainer(RandomWalk, Trainer):
    def __init__(self, player: RandomWalkQLearner, left=-6, right=6, goal=10) -> None:
        super().__init__(left, right, goal)
        self.player = player

    @override
    def train(self, episodes=10000) -> None:
        super().train(episodes)
        self.player.save_policy()

    def train_once(self) -> None:
        while not self.finished():
            state = self.get_state()

            action = self.player.choose_action(state)
            self.step(action)

            new_state = self.get_state()

            # reward minimizing distance to right bound
            new_r_dist = abs(self.r_bound - new_state)
            old_r_dist = abs(self.r_bound - state)
            reward = old_r_dist - new_r_dist

            if self.at_right():
                reward += 1

            self.player.update_q_value(state, action, reward, new_state)

        self.reset()


def q_trained_game() -> None:
    pkl_file = "src/random_walk/q.pkl"
    q = RandomWalkQLearner(epsilon=0.5, q_pickle=pkl_file)
    g = RandomWalkQTrainer(player=q)
    g.train()

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


class RandomWalkMonteCarloLearner(MonteCarloLearner[int, Action]):
    def get_actions_from_state(self, state: int) -> List[Action]:
        return [Action.L, Action.R, Action.N]

    def apply(self, state: int, action: Action) -> int:
        return RandomWalk.apply(state, action)


class RandomWalkMonteCarloTrainer(RandomWalk):
    # changing the defaults in the init will simplify monte carlo a lot more as compared to q learning
    def __init__(self, player: MonteCarloLearner, left=-3, right=3, goal=1) -> None:
        super().__init__(left, right, goal)
        self.player = player

    def train(self, episodes=1000) -> None:
        for e in range(1, episodes + 1):
            self.train_once()

            if e % 100 == 0:
                print(f"Episode {e}/{episodes}")

        self.player.save_policy()

    def train_once(self) -> None:
        while not self.finished():
            a = self.player.choose_action(self.get_state())
            self.step(a)
            self.player.add_state(self.get_state())

        self.player.propagate_reward(1)
        self.reset()
        self.player.reset_states()


def monte_carlo_trained_game(training_episodes=10000):
    policy_pkl = "src/random_walk/monte_carlo_player.pkl"
    p = RandomWalkMonteCarloLearner(policy_file=policy_pkl)
    g = RandomWalkMonteCarloTrainer(p)
    g.train(episodes=training_episodes)

    while not g.finished():
        print(f"{g.show()}\n")

        action = p.choose_action(g.get_state())
        g.step(action)

        print(f"computer takes {action} action")

    print(f"\nrandom walk ended! {g.show()}\n")

    with open(policy_pkl, "rb") as file:
        state_values = pickle.load(file)
        for i in range(g.l_bound * 3, g.r_bound * 3 + 1):
            print(f"state {i} has value {state_values[i]}")
