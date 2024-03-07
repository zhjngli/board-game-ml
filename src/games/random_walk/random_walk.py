from typing import List, NamedTuple

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from games.game import P1, Action, ActionStatus, BasicState, Board, Game, Player
from learners.q import SimpleQLearner, SimpleQParameters
from learners.trainer import Trainer


class RandomWalkState(BasicState):
    def __init__(
        self,
        board: Board,
        player: Player,
        i: int,
        left: int,
        right: int,
        goal: int,
        score: int,
    ) -> None:
        super().__init__(board, player)  # unused
        self.i = i
        self.left = left
        self.right = right
        self.goal = goal
        self.score = score


class RandomWalkIR(NamedTuple):
    i: int
    left: int
    right: int
    goal: int
    score: int


class RandomWalk(Game[RandomWalkState, RandomWalkIR]):
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

    def state(self) -> RandomWalkState:
        return RandomWalkState(
            board=np.asarray([]),
            player=P1,
            i=self.i,
            left=self.l_bound,
            right=self.r_bound,
            goal=self.goal,
            score=self.score,
        )

    @staticmethod
    def orient_state(state: RandomWalkState) -> RandomWalkState:
        return state  # no need to orient

    @staticmethod
    def symmetries_of(a: NDArray) -> List[NDArray]:
        return [a]

    @staticmethod
    def to_immutable(state: RandomWalkState) -> RandomWalkIR:
        return RandomWalkIR(
            i=state.i,
            left=state.left,
            right=state.right,
            goal=state.goal,
            score=state.score,
        )

    @staticmethod
    def from_immutable(ir: RandomWalkIR) -> RandomWalkState:
        return RandomWalkState(
            board=np.asarray([]),
            player=P1,
            i=ir.i,
            left=ir.left,
            right=ir.right,
            goal=ir.goal,
            score=ir.score,
        )

    @staticmethod
    def action_to_char(a: Action) -> str:
        if a == 0:
            return "L"
        elif a == 1:
            return "N"
        elif a == 2:
            return "R"
        else:
            raise ValueError(f"Invalid action {a}")

    @staticmethod
    def from_action(a: Action) -> int:
        if a == 0:
            return -1
        elif a == 1:
            return 0
        elif a == 2:
            return 1
        else:
            raise ValueError(f"Invalid action {a}")

    def num_actions(self) -> int:
        return 3  # move left, move right, stand

    @staticmethod
    def actions(state: RandomWalkState) -> List[ActionStatus]:
        return [1, 1, 1]  # all actions are always valid

    @staticmethod
    def apply(state: RandomWalkState, action: Action) -> RandomWalkState:
        delta = RandomWalk.from_action(action)
        return RandomWalkState(
            board=state.board,
            player=state.player,
            i=state.i + delta,
            left=state.left,
            right=state.right,
            goal=state.goal,
            score=state.score,
        )

    def step(self, action: Action) -> None:
        delta = RandomWalk.from_action(action)
        self.i += delta

        if self.at_left():
            self.score -= 1
        elif self.at_right():
            self.score += 1

    def at_left(self) -> bool:
        return self.i == self.l_bound

    def at_right(self) -> bool:
        return self.i == self.r_bound

    @staticmethod
    def check_finished(state: RandomWalkState) -> bool:
        return state.score >= state.goal

    def finished(self) -> bool:
        return RandomWalk.check_finished(self.state())

    @staticmethod
    def calculate_reward(state: RandomWalkState) -> float:
        return float(state.score)

    def show(self) -> str:
        return f"random walk at {self.i} with score {self.score}"


# class RandomWalkQLearner(SimpleQLearner[int, Action]):
#     def default_action_q_values(self) -> dict[Action, float]:
#         actions = {}
#         for a in Action:
#             actions[a] = 0.0
#         return actions

#     def get_actions_from_state(self, state: int) -> List[Action]:
#         return [Action.L, Action.R, Action.N]


class RandomWalkQTrainer(RandomWalk, Trainer):
    def __init__(self, player: SimpleQLearner, left=-6, right=6, goal=10) -> None:
        super().__init__(left, right, goal)
        self.player = player

    @override
    def train(self, episodes=10000) -> None:
        super().train(episodes)
        self.player.save_policy()

    def train_once(self) -> None:
        while not self.finished():
            ir = RandomWalk.to_immutable(self.state())

            action = self.player.choose_action(ir)
            self.step(action)

            new_ir = RandomWalk.to_immutable(self.state())

            # reward minimizing distance to right bound
            new_r_dist = abs(self.r_bound - new_ir.i)
            old_r_dist = abs(self.r_bound - ir.i)
            reward = old_r_dist - new_r_dist

            if self.at_right():
                reward += 1

            self.player.update_q_value(ir, action, reward, new_ir)

        self.reset()


def q_trained_game() -> None:
    pkl_file = "src/games/random_walk/q.pkl"
    q = SimpleQLearner(
        game=RandomWalk(),
        q_pickle=pkl_file,
        params=SimpleQParameters(alpha=0.1, gamma=0.9, epsilon=0.5),
    )
    t = RandomWalkQTrainer(player=q)
    t.train(episodes=0)

    while not t.finished():
        print(f"\n{t.show()}\n")

        action = q.choose_action(RandomWalk.to_immutable(t.state()), exploit=True)
        t.step(action)

        print(f"computer takes {RandomWalk.action_to_char(action)} action")

    print(f"\nrandom walk ended! {t.show()}\n")

    # with open(pkl_file, "rb") as file:
    #     q_table = pickle.load(file)
    #     for i in range(t.l_bound, t.r_bound + 1):
    #         print("state: ", i)
    #         for a, r in q_table[i].items():
    #             print(type(r))
    #             print(RandomWalk.action_to_char(a), " has reward ", r)
    #         print()


# class RandomWalkMonteCarloLearner(MonteCarloLearner[int, Action]):
#     def get_actions_from_state(self, state: int) -> List[Action]:
#         return [Action.L, Action.R, Action.N]

#     def apply(self, state: int, action: Action) -> int:
#         return RandomWalk.apply(state, action)


# class RandomWalkMonteCarloTrainer(RandomWalk):
#     # changing the defaults in the init will simplify monte carlo a lot more as compared to q learning
#     def __init__(self, player: MonteCarloLearner, left=-3, right=3, goal=1) -> None:
#         super().__init__(left, right, goal)
#         self.player = player

#     def train(self, episodes=1000) -> None:
#         for e in range(1, episodes + 1):
#             self.train_once()

#             if e % 100 == 0:
#                 print(f"Episode {e}/{episodes}")

#         self.player.save_policy()

#     def train_once(self) -> None:
#         while not self.finished():
#             a = self.player.choose_action(self.get_state())
#             self.step(a)
#             self.player.add_state(self.get_state())

#         self.player.propagate_reward(1)
#         self.reset()
#         self.player.reset_states()


# def monte_carlo_trained_game(training_episodes=10000):
#     policy_pkl = "src/games/random_walk/monte_carlo_player.pkl"
#     p = RandomWalkMonteCarloLearner(policy_file=policy_pkl)
#     g = RandomWalkMonteCarloTrainer(p)
#     g.train(episodes=training_episodes)

#     while not g.finished():
#         print(f"{g.show()}\n")

#         action = p.choose_action(g.get_state())
#         g.step(action)

#         print(f"computer takes {action} action")

#     print(f"\nrandom walk ended! {g.show()}\n")

#     with open(policy_pkl, "rb") as file:
#         state_values = pickle.load(file)
#         for i in range(g.l_bound * 3, g.r_bound * 3 + 1):
#             print(f"state {i} has value {state_values[i]}")
