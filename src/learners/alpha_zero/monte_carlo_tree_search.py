import math
from abc import ABC
from typing import Dict, Generic, List, NamedTuple, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from games.game import Action, ActionStatus, Game, Immutable, NNInput, State
from learners.alpha_zero.types import A0NNOutput
from nn.neural_network import NeuralNetwork


class MCTSParameters(NamedTuple):
    num_searches: int
    cpuct: float
    epsilon: float
    dirichlet_alpha: float = 0
    dirichlet_epsilon: float = 0


class MonteCarloTreeSearch(ABC, Generic[State, Immutable]):
    def __init__(
        self,
        game: Game[State, Immutable],
        nn: NeuralNetwork[NNInput, A0NNOutput],
        params: MCTSParameters,
    ) -> None:
        # q values for state-action pair
        self.q: Dict[Tuple[Immutable, Action], float] = {}
        # number of times state-action pair was visited
        self.nsa: Dict[Tuple[Immutable, Action], int] = {}
        # number of times state was visited
        self.ns: Dict[Immutable, int] = {}
        # reward value at terminal states
        self.evs: Dict[Immutable, float] = {}
        # set of terminal states (for detecting game-over independent of reward value)
        self.terminals: Set[Immutable] = set()
        # valid actions at a game state
        self.vas: Dict[Immutable, List[ActionStatus]] = {}
        # the action policies at a game state
        self.ps: Dict[Immutable, NDArray] = {}

        self.game = game
        self.nn = nn

        self.num_searches = params.num_searches
        self.cpuct = params.cpuct
        self.epsilon = params.epsilon
        self.dirichlet_alpha = params.dirichlet_alpha
        self.dirichlet_epsilon = params.dirichlet_epsilon

    def action_probabilities(self, state: State, temperature: float) -> List[float]:
        # TODO: output NDArray instead of list, might have more optimized calculations?
        for _ in range(self.num_searches):
            self.search(state, is_root=True)

        ir: Immutable = self.game.to_immutable(state)
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

        # probability of each action weighted by how many times the state has been visited
        visits = [n ** (1 / temperature) for n in sa_visits]
        total_visits = sum(visits)
        if total_visits == 0:
            return [1 / self.game.num_actions()] * self.game.num_actions()
        return [v / total_visits for v in visits]

    def _apply_dirichlet_noise(self, ir: Immutable, valids: List[ActionStatus]) -> None:
        """Mix Dirichlet noise into the policy at the root node for exploration."""
        valid_indices = [a for a in range(len(self.ps[ir])) if valids[a]]
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_indices))
        valid_noise = np.zeros_like(self.ps[ir])
        for i, idx in enumerate(valid_indices):
            valid_noise[idx] = noise[i]
        self.ps[ir] = (1 - self.dirichlet_epsilon) * self.ps[
            ir
        ] + self.dirichlet_epsilon * valid_noise

    def _expand_leaf(self, state: State, ir: Immutable, is_root: bool) -> float:
        """Expand a new leaf node: run NN, mask/normalize policy, optionally add noise."""
        nn_input = self.game.to_nn_input(state)
        out = self.nn.predict([nn_input])[0]
        self.ps[ir] = out.policy
        v = out.value

        valids = self.game.actions(state)
        self.ps[ir] = self.ps[ir] * valids
        policy_sum = np.sum(self.ps[ir])
        if policy_sum > 0:
            self.ps[ir] /= policy_sum
        else:
            self.ps[ir] = self.ps[ir] + valids
            self.ps[ir] /= np.sum(self.ps[ir])

        if is_root and self.dirichlet_alpha > 0:
            self._apply_dirichlet_noise(ir, valids)

        self.vas[ir] = valids
        self.ns[ir] = 0
        return -v

    def search(self, state: State, is_root: bool = False) -> float:
        ir = self.game.to_immutable(state)

        # Terminal detection — uses self.terminals set, not reward value
        if ir not in self.terminals:
            if self.game.check_finished(state):
                self.terminals.add(ir)
                self.evs[ir] = self.game.calculate_reward(state)
        if ir in self.terminals:
            return -self.evs[ir]

        if ir not in self.ps:
            return self._expand_leaf(state, ir, is_root)

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

        v = self.search(next_s)  # is_root defaults to False

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
