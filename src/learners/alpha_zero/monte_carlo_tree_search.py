import math
from abc import ABC
from typing import Dict, Generic, List, NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray

from games.game import Action, ActionStatus, Game, ImmutableRepresentation, State
from nn.neural_network import NeuralNetwork


class MCTSParameters(NamedTuple):
    num_searches: int
    cpuct: float
    epsilon: float


class MonteCarloTreeSearch(ABC, Generic[State, ImmutableRepresentation]):
    def __init__(
        self,
        game: Game[State, ImmutableRepresentation],
        nn: NeuralNetwork,
        params: MCTSParameters,
    ) -> None:
        # q values for state-action pair
        self.q: Dict[Tuple[ImmutableRepresentation, Action], float] = {}
        # number of times state-action pair was visited
        self.nsa: Dict[Tuple[ImmutableRepresentation, Action], int] = {}
        # number of times state was visited
        self.ns: Dict[ImmutableRepresentation, int] = {}
        # value at game end, or 0 if the game is not finished yet
        self.evs: Dict[ImmutableRepresentation, float] = {}
        # valid actions at a game state
        self.vas: Dict[ImmutableRepresentation, List[ActionStatus]] = {}
        # the action policies at a game state
        self.ps: Dict[ImmutableRepresentation, NDArray] = {}

        self.game = game
        self.nn = nn

        self.num_searches = params.num_searches
        self.cpuct = params.cpuct
        self.epsilon = params.epsilon

    def action_probabilities(self, state: State, temperature: float) -> List[float]:
        # TODO: output NDArray instead of list, might have more optimized calculations?
        for _ in range(self.num_searches):
            self.search(state)

        ir: ImmutableRepresentation = self.game.immutable_of(state)
        sa_visits = [
            self.nsa[(ir, a)] if (ir, a) in self.nsa else 0
            for a in range(self.game.num_actions())
        ]

        if temperature == 0:
            bests = np.array(np.argwhere(sa_visits == np.max(sa_visits))).flatten()
            best = np.random.choice(bests)
            probs = [0.0] * len(sa_visits)
            probs[best] = 1
            return probs

        visits = [n ** (1 / temperature) for n in sa_visits]
        total_visits = sum(visits)
        # probability of each action weighted by how many times the state has been visited
        if total_visits == 0:
            return [1 / self.game.num_actions()] * self.game.num_actions()
        return [v / total_visits for v in visits]

    def search(self, state: State) -> float:
        ir = self.game.immutable_of(state)

        if ir not in self.evs:
            self.evs[ir] = (
                self.game.calculate_reward(state)
                if self.game.check_finished(state)
                else 0
            )
        if self.evs[ir] != 0:
            return -self.evs[ir]

        if ir not in self.ps:
            self.ps[ir], v = self.nn.predict(state)
            valids = self.game.actions(state)
            self.ps[ir] = self.ps[ir] * valids
            policy_sum = np.sum(self.ps[ir])
            if policy_sum > 0:
                self.ps[ir] /= policy_sum
            else:
                # the nn doesn't know anything about this state yet, so just use all the valid moves as the available actions
                self.ps[ir] = self.ps[ir] + valids
                self.ps[ir] /= np.sum(self.ps[ir])

            self.vas[ir] = valids
            self.ns[ir] = 0
            return -v

        valids = self.vas[ir]
        best_u = -float("inf")
        best_a = -1

        # find the action with the highest upper confidence bound u
        # u(s, a) = q(s, a) + c_puct * pi(s, a) * sqrt(sum all actions b: (N(s, b)) / (1 + N(s, a))
        for a in range(self.game.num_actions()):
            if valids[a]:
                if (ir, a) in self.q:
                    u = self.q[(ir, a)] + self.cpuct * self.ps[ir][a] * math.sqrt(
                        self.ns[ir]
                    ) / (1 + self.nsa[(ir, a)])
                else:
                    u = (
                        self.cpuct
                        * self.ps[ir][a]
                        * math.sqrt(self.ns[ir] + self.epsilon)
                    )  # TODO: how does epsilon change the upper confidence bound

                if u > best_u:
                    best_u = u
                    best_a = a

        a = best_a
        next_s = self.game.apply(state, a)
        next_s = self.game.orient_state(next_s)

        v = self.search(next_s)

        if (ir, a) in self.q:
            self.q[(ir, a)] = (self.nsa[(ir, a)] * self.q[(ir, a)] + v) / (
                self.nsa[(ir, a)] + 1
            )
            self.nsa[(ir, a)] += 1
        else:
            self.q[(ir, a)] = v
            self.nsa[(ir, a)] = 1

        self.ns[ir] += 1
        return -v
