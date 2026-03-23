# UTTT AlphaZero Training Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Ultimate Tic-Tac-Toe AlphaZero training so the neural network can actually learn to play the game.

**Architecture:** Refactor `A0NNInput` to a generic type so each game defines its own NN input. Add `active_nonant` as a spatial mask channel to UTTT's NN input. Redesign the UTTT neural network with more capacity. Add Dirichlet noise to MCTS for exploration. Fix MCTS terminal detection bug.

**Tech Stack:** Python 3.11, TensorFlow/Keras, NumPy, pipenv

**Spec:** `docs/superpowers/specs/2026-03-22-uttt-alpha-zero-training-design.md`

---

### Task 1: Refactor base types and Game abstract class

**Files:**
- Modify: `src/learners/alpha_zero/types.py`
- Modify: `src/games/game.py`
- Test: `test/test_simple.py` (existing tests still pass)

- [ ] **Step 1: Update `types.py` — add `NNInput` TypeVar, keep `A0NNInput` as deprecated stub**

In `src/learners/alpha_zero/types.py`, add a `TypeVar` and keep the old class for pickle compat:

```python
from typing import NamedTuple, TypeVar

from numpy.typing import NDArray

from games.game import Board

# Generic NN input type — each game defines its own
NNInput = TypeVar("NNInput")

# common output types
Policy = NDArray
Value = float


# Deprecated: kept for backward pickle compatibility with old .pkl training files.
# New code should use game-specific types via Game.to_nn_input().
class A0NNInput(NamedTuple):
    board: Board


class A0NNOutput(NamedTuple):
    policy: Policy
    value: Value
```

- [ ] **Step 2: Update `game.py` — add `to_nn_input` and `training_symmetries`, remove `symmetries_of`**

In `src/games/game.py`, add the `NNInput` TypeVar import, add `to_nn_input` and `training_symmetries` abstract methods, remove `symmetries_of`:

```python
from abc import ABC, abstractmethod
from typing import Generic, List, Literal, Tuple, TypeVar

from numpy.typing import NDArray

Board = NDArray
Player = Literal[1, -1]
P1: Player = 1
P2: Player = -1

NNInput = TypeVar("NNInput")


class BasicState:
    """
    A basic state for 2 player games. Easily extensible with other fields.
    """

    def __init__(self, board: Board, player: Player) -> None:
        self.board = board
        self.player = player


State = TypeVar("State", bound=BasicState)
Action = int
Immutable = TypeVar("Immutable")

ActionStatus = Literal[1, 0]
VALID: ActionStatus = 1
INVAL: ActionStatus = 0

P1WIN = 1
P2WIN = -1


def switch_player(p: Player) -> Player:
    return P1 if p == P2 else P2


class Game(ABC, Generic[State, Immutable]):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def state(self) -> State:
        pass

    @staticmethod
    @abstractmethod
    def to_immutable(state: State) -> Immutable:
        pass

    @abstractmethod
    def num_actions(self) -> int:
        pass

    @staticmethod
    @abstractmethod
    def actions(state: State) -> List[ActionStatus]:
        pass

    @staticmethod
    @abstractmethod
    def apply(state: State, action: Action) -> State:
        pass

    @staticmethod
    @abstractmethod
    def check_finished(state: State) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def calculate_reward(state: State) -> float:
        pass

    @staticmethod
    @abstractmethod
    def orient_state(state: State) -> State:
        pass

    @staticmethod
    @abstractmethod
    def to_nn_input(state: State) -> NDArray:
        """
        Convert a game state to the NN input tensor.
        Each game defines what the NN sees (e.g., board only, board + active_nonant mask).
        """
        pass

    @staticmethod
    @abstractmethod
    def training_symmetries(
        nn_input: NDArray, policy: NDArray
    ) -> List[Tuple[NDArray, NDArray]]:
        """
        Return paired (nn_input, policy) symmetries for training data augmentation.
        MUST include the identity (un-transformed) pair.
        """
        pass
```

- [ ] **Step 3: Run tests to verify nothing is broken at the type level**

Run: `pipenv run pytest test -v`
Expected: Existing tests pass (they don't import `symmetries_of` directly). If there are import-time failures from game classes not implementing the new abstract methods yet, that's expected — we fix those in the next tasks.

- [ ] **Step 4: Commit**

```bash
git add src/learners/alpha_zero/types.py src/games/game.py
git commit -m "refactor: add NNInput TypeVar, to_nn_input, training_symmetries to Game base class"
```

---

### Task 2: Update TicTacToe to implement new abstract methods

**Files:**
- Modify: `src/games/tictactoe/tictactoe.py:196-207` (replace `symmetries_of`)
- Test: `test/test_tictactoe.py` (new)

- [ ] **Step 1: Write tests for TTT `to_nn_input` and `training_symmetries`**

Create `test/test_tictactoe.py`:

```python
import numpy as np

from games.game import P1, P2
from games.tictactoe.tictactoe import TicTacToe, TicTacToeState


def test_to_nn_input_returns_board():
    board = np.array([[1, 0, -1], [0, 1, 0], [-1, 0, 1]])
    state = TicTacToeState(board=board, player=P1)
    nn_input = TicTacToe.to_nn_input(state)
    np.testing.assert_array_equal(nn_input, board)


def test_training_symmetries_includes_identity():
    board = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    state = TicTacToeState(board=board, player=P1)
    nn_input = TicTacToe.to_nn_input(state)
    policy = np.array([0.5, 0.1, 0.1, 0.1, 0.0, 0.05, 0.05, 0.05, 0.05])

    syms = TicTacToe.training_symmetries(nn_input, policy)
    # Should have 8 symmetries (4 rotations x 2 mirror states, rot by 4 = identity)
    assert len(syms) == 8

    # Check identity is present (rotation by 4 = identity)
    found_identity = False
    for inp, pol in syms:
        if np.array_equal(inp, nn_input) and np.array_equal(pol, policy):
            found_identity = True
            break
    assert found_identity


def test_training_symmetries_preserves_shape():
    board = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    state = TicTacToeState(board=board, player=P1)
    nn_input = TicTacToe.to_nn_input(state)
    policy = np.array([0.5, 0.1, 0.1, 0.1, 0.0, 0.05, 0.05, 0.05, 0.05])

    syms = TicTacToe.training_symmetries(nn_input, policy)
    for inp, pol in syms:
        assert inp.shape == (3, 3)
        assert pol.shape == (9,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pipenv run pytest test/test_tictactoe.py -v`
Expected: FAIL — `to_nn_input` and `training_symmetries` not defined yet.

- [ ] **Step 3: Implement `to_nn_input` and `training_symmetries` on TicTacToe, remove `symmetries_of`**

In `src/games/tictactoe/tictactoe.py`, remove `symmetries_of` (lines 196-207) and add:

```python
@staticmethod
def to_nn_input(state: TicTacToeState) -> NDArray:
    return state.board

@staticmethod
def training_symmetries(
    nn_input: NDArray, policy: NDArray
) -> List[Tuple[NDArray, NDArray]]:
    syms: List[Tuple[NDArray, NDArray]] = []
    board = nn_input.reshape((3, 3))
    pol = policy.reshape((3, 3))
    for i in range(1, 5):
        for mirror in [True, False]:
            b = np.rot90(board, i)
            p = np.rot90(pol, i)
            if mirror:
                b = np.fliplr(b)
                p = np.fliplr(p)
            syms.append((b.reshape(nn_input.shape), p.reshape(policy.shape)))
    return syms
```

Also add `Tuple` to the typing imports at the top of the file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pipenv run pytest test/test_tictactoe.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/games/tictactoe/tictactoe.py test/test_tictactoe.py
git commit -m "feat: implement to_nn_input and training_symmetries for TicTacToe"
```

---

### Task 3: Update UTTT to implement new abstract methods

**Files:**
- Modify: `src/games/ultimate_ttt/ultimate.py:399-413` (replace `symmetries_of`), line 439 (draw reward)
- Test: `test/games/test_ultimate.py` (extend)

- [ ] **Step 1: Write tests for UTTT `to_nn_input`**

Add to `test/games/test_ultimate.py`:

```python
def test_to_nn_input_shape():
    b = np.zeros((3, 3, 3, 3))
    state = UltimateState(board=b, player=P1, active_nonant=None)
    nn_input = UltimateTicTacToe.to_nn_input(state)
    assert nn_input.shape == (9, 9, 2)


def test_to_nn_input_spatial_layout():
    """board[R][C][r][c] should map to grid[R*3+r][C*3+c] in channel 0."""
    b = np.zeros((3, 3, 3, 3))
    b[0][0][2][1] = 1  # section (0,0), cell (2,1)
    b[1][2][0][0] = -1  # section (1,2), cell (0,0)
    state = UltimateState(board=b, player=P1, active_nonant=None)
    nn_input = UltimateTicTacToe.to_nn_input(state)
    # channel 0: board
    assert nn_input[0 * 3 + 2][0 * 3 + 1][0] == 1  # (2, 1) in 9x9
    assert nn_input[1 * 3 + 0][2 * 3 + 0][0] == -1  # (3, 6) in 9x9


def test_to_nn_input_active_nonant_mask():
    b = np.zeros((3, 3, 3, 3))
    state = UltimateState(board=b, player=P1, active_nonant=(1, 2))
    nn_input = UltimateTicTacToe.to_nn_input(state)
    mask = nn_input[:, :, 1]
    # Only section (1,2) should be 1: rows 3-5, cols 6-8
    assert mask[3][6] == 1
    assert mask[5][8] == 1
    assert mask[0][0] == 0  # section (0,0) should be 0
    assert mask[4][4] == 0  # section (1,1) should be 0
    assert np.sum(mask) == 9  # exactly one 3x3 section lit up


def test_to_nn_input_free_choice_mask():
    b = np.zeros((3, 3, 3, 3))
    state = UltimateState(board=b, player=P1, active_nonant=None)
    nn_input = UltimateTicTacToe.to_nn_input(state)
    mask = nn_input[:, :, 1]
    assert np.all(mask == 1)  # all 1s when free choice


def test_draw_reward_is_zero():
    """Draws should return 0, not 0.1."""
    b = np.asarray(
        [
            [
                [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                [[0, -1, 0], [0, -1, 0], [0, -1, 0]],
                [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            ],
            [
                [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]],
                [[1, -1, 1], [1, -1, 1], [-1, 1, -1]],
                [[0, 0, -1], [0, 0, -1], [0, 0, -1]],
            ],
            [
                [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                [[0, -1, 0], [0, -1, 0], [0, -1, 0]],
                [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            ],
        ]
    )
    state = UltimateState(board=b, player=P1, active_nonant=None)
    assert UltimateTicTacToe.check_finished(state)
    assert UltimateTicTacToe.calculate_reward(state) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pipenv run pytest test/games/test_ultimate.py -v`
Expected: FAIL — `to_nn_input` not defined, draw reward still 0.1.

- [ ] **Step 3: Implement `to_nn_input` on UltimateTicTacToe**

In `src/games/ultimate_ttt/ultimate.py`, add after the existing `orient_state` method:

```python
@staticmethod
def to_nn_input(state: UltimateState) -> NDArray:
    board_plane = state.board.transpose(0, 2, 1, 3).reshape(9, 9)
    nonant_mask = np.zeros((9, 9))
    if state.active_nonant is not None:
        R, C = state.active_nonant
        nonant_mask[R * 3 : (R + 1) * 3, C * 3 : (C + 1) * 3] = 1
    else:
        nonant_mask[:] = 1
    return np.stack([board_plane, nonant_mask], axis=-1)
```

Also add `Tuple` to the typing imports.

- [ ] **Step 4: Implement `training_symmetries`, remove `symmetries_of`**

Remove `symmetries_of` (lines 399-413) and add:

```python
@staticmethod
def training_symmetries(
    nn_input: NDArray, policy: NDArray
) -> List[Tuple[NDArray, NDArray]]:
    syms: List[Tuple[NDArray, NDArray]] = []
    pol = policy.reshape((3, 3, 3, 3))
    for i in range(1, 5):
        for mirror in [True, False]:
            # Rotate (9,9,2) input on spatial axes — both channels rotate together
            inp = np.rot90(nn_input, i, axes=(0, 1))
            # Rotate policy in (3,3,3,3) space — inner and outer boards
            p = np.rot90(np.rot90(pol, i), i, (2, 3))
            if mirror:
                inp = np.flip(inp, axis=1)
                p = np.flip(np.flip(p, axis=1), axis=3)
            syms.append((inp, p.reshape(policy.shape)))
    return syms
```

- [ ] **Step 5: Change draw reward to 0**

In `src/games/ultimate_ttt/ultimate.py`, change `calculate_reward` (line 439):

From: `return 0.1`
To: `return 0`

- [ ] **Step 6: Run tests to verify they pass**

Run: `pipenv run pytest test/games/test_ultimate.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/games/ultimate_ttt/ultimate.py test/games/test_ultimate.py
git commit -m "feat: implement to_nn_input with active_nonant mask and training_symmetries for UTTT"
```

---

### Task 4: Update Digit Party to implement new abstract methods

**Files:**
- Modify: `src/games/digit_party/game.py:348-358` (replace `symmetries_of`)

- [ ] **Step 1: Replace `symmetries_of` with `to_nn_input` and `training_symmetries`**

In `src/games/digit_party/game.py`, remove `symmetries_of` (lines 348-358) and add:

```python
@staticmethod
def to_nn_input(state: DigitPartyState) -> NDArray:
    return state.board

@staticmethod
def training_symmetries(
    nn_input: NDArray, policy: NDArray
) -> List[Tuple[NDArray, NDArray]]:
    return [(nn_input, policy)]
```

Add `Tuple` to the typing imports at the top of the file.

- [ ] **Step 2: Run all tests**

Run: `pipenv run pytest test -v`
Expected: PASS — all games now implement the new abstract methods.

- [ ] **Step 3: Commit**

```bash
git add src/games/digit_party/game.py
git commit -m "feat: implement to_nn_input and training_symmetries for Digit Party"
```

---

### Task 5: Refactor MCTS — terminal detection fix, Dirichlet noise, generic NNInput

**Files:**
- Modify: `src/learners/alpha_zero/monte_carlo_tree_search.py`
- Test: `test/test_mcts.py` (new)

- [ ] **Step 1: Write test for MCTS terminal detection with reward=0**

Create `test/test_mcts.py`:

```python
import numpy as np

from games.game import P1
from games.ultimate_ttt.ultimate import UltimateState, UltimateTicTacToe
from learners.alpha_zero.monte_carlo_tree_search import (
    MCTSParameters,
    MonteCarloTreeSearch,
)


class MockNN:
    """Mock NN that returns uniform policy and 0 value."""

    def predict(self, inputs):
        n = UltimateTicTacToe.num_actions()
        return [type("Out", (), {"policy": np.ones(n) / n, "value": 0.0})()]

    def set_weights(self, w):
        pass

    def get_weights(self):
        return []


def test_mcts_handles_zero_reward_terminal():
    """MCTS should correctly identify terminal states even when reward is 0 (draw)."""
    params = MCTSParameters(
        num_searches=5,
        cpuct=1,
        epsilon=1e-4,
        dirichlet_alpha=0,
        dirichlet_epsilon=0,
    )
    game = UltimateTicTacToe()
    mcts = MonteCarloTreeSearch(game, MockNN(), params)

    # Create a filled board that's a draw (reward=0)
    b = np.asarray(
        [
            [
                [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                [[0, -1, 0], [0, -1, 0], [0, -1, 0]],
                [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            ],
            [
                [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]],
                [[1, -1, 1], [1, -1, 1], [-1, 1, -1]],
                [[0, 0, -1], [0, 0, -1], [0, 0, -1]],
            ],
            [
                [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                [[0, -1, 0], [0, -1, 0], [0, -1, 0]],
                [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            ],
        ]
    )
    state = UltimateState(board=b, player=P1, active_nonant=None)
    assert UltimateTicTacToe.check_finished(state)

    # search should return the negated reward (0) without crashing
    result = mcts.search(state)
    assert result == 0  # -0 == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pipenv run pytest test/test_mcts.py::test_mcts_handles_zero_reward_terminal -v`
Expected: FAIL — current code continues past terminal state with reward=0.

- [ ] **Step 3: Rewrite `monte_carlo_tree_search.py`**

Replace the full content of `src/learners/alpha_zero/monte_carlo_tree_search.py`:

```python
import math
from abc import ABC
from typing import Dict, Generic, List, NamedTuple, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from games.game import Action, ActionStatus, Game, Immutable, State
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
        nn: NeuralNetwork,
        params: MCTSParameters,
    ) -> None:
        self.q: Dict[Tuple[Immutable, Action], float] = {}
        self.nsa: Dict[Tuple[Immutable, Action], int] = {}
        self.ns: Dict[Immutable, int] = {}
        self.evs: Dict[Immutable, float] = {}
        self.terminals: Set[Immutable] = set()
        self.vas: Dict[Immutable, List[ActionStatus]] = {}
        self.ps: Dict[Immutable, NDArray] = {}

        self.game = game
        self.nn = nn

        self.num_searches = params.num_searches
        self.cpuct = params.cpuct
        self.epsilon = params.epsilon
        self.dirichlet_alpha = params.dirichlet_alpha
        self.dirichlet_epsilon = params.dirichlet_epsilon

    def action_probabilities(self, state: State, temperature: float) -> List[float]:
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

        visits = [n ** (1 / temperature) for n in sa_visits]
        total_visits = sum(visits)
        if total_visits == 0:
            return [1 / self.game.num_actions()] * self.game.num_actions()
        return [v / total_visits for v in visits]

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

            # Dirichlet noise at root only
            if is_root and self.dirichlet_alpha > 0:
                valid_indices = [a for a in range(len(self.ps[ir])) if valids[a]]
                noise = np.random.dirichlet(
                    [self.dirichlet_alpha] * len(valid_indices)
                )
                valid_noise = np.zeros_like(self.ps[ir])
                for i, idx in enumerate(valid_indices):
                    valid_noise[idx] = noise[i]
                self.ps[ir] = (
                    (1 - self.dirichlet_epsilon) * self.ps[ir]
                    + self.dirichlet_epsilon * valid_noise
                )

            self.vas[ir] = valids
            self.ns[ir] = 0
            return -v

        valids = self.vas[ir]
        best_u = -float("inf")
        best_a = -1

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
                    )

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
```

- [ ] **Step 4: Run tests**

Run: `pipenv run pytest test/test_mcts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/learners/alpha_zero/monte_carlo_tree_search.py test/test_mcts.py
git commit -m "feat: fix MCTS terminal detection, add Dirichlet noise, use game.to_nn_input()"
```

---

### Task 6: Refactor AlphaZero — generic NNInput, separate MCTS params, training_symmetries

**Files:**
- Modify: `src/learners/alpha_zero/alpha_zero.py`

- [ ] **Step 1: Update `A0Parameters` to include both MCTS param sets, remove `m_params` constructor arg**

In `src/learners/alpha_zero/alpha_zero.py`, update `A0Parameters`:

```python
class A0Parameters(NamedTuple):
    temp_threshold: int
    pit_games: int
    pit_threshold: float
    training_episodes: int
    training_games_per_episode: int
    training_queue_length: int
    training_hist_max_len: int
    thread_max_workers: int
    training_mcts_params: MCTSParameters
    eval_mcts_params: MCTSParameters
```

- [ ] **Step 2: Update `AlphaZero.__init__` — remove `m_params` arg, use params fields**

Update the constructor to read MCTS params from `A0Parameters` instead of a separate arg:

```python
class AlphaZero(ABC, Generic[State, Immutable]):
    def __init__(
        self,
        create_game: Callable[[], Game[State, Immutable]],
        create_nn: Callable[[], NeuralNetwork],
        params: A0Parameters,
        training_examples_folder: str,
    ) -> None:
        self.create_game = create_game
        self.create_nn = create_nn
        self.nn = create_nn()
        self.pn = create_nn()
        self.training_history: List[Deque[Tuple]] = []
        self.training_examples_folder = training_examples_folder

        self.training_mcts_params = params.training_mcts_params
        self.eval_mcts_params = params.eval_mcts_params

        self.temperature_threshold = params.temp_threshold
        self.pit_games = params.pit_games
        self.pit_threshold = params.pit_threshold
        self.training_episodes = params.training_episodes
        self.training_games_per_episode = params.training_games_per_episode
        self.training_queue_length = params.training_queue_length
        self.training_hist_max_len = params.training_hist_max_len
        self.thread_max_workers = params.thread_max_workers
```

- [ ] **Step 3: Update `train_once` — use `to_nn_input` and `training_symmetries`, training MCTS params**

```python
def train_once(self) -> List[Tuple]:
    game = self.create_game()
    game.reset()

    nn = self.create_nn()
    nn.set_weights(self.nn.get_weights())
    m = MonteCarloTreeSearch(self.create_game(), nn, self.training_mcts_params)

    training_data: List[Tuple[NDArray, Player, NDArray, Optional[float]]] = []
    state = game.state()
    player = state.player

    turn = 0
    while not game.check_finished(state):
        turn += 1
        oriented_state = game.orient_state(state)
        temperature = 1 if turn < self.temperature_threshold else 0
        pi = m.action_probabilities(oriented_state, temperature)

        nn_input = game.to_nn_input(oriented_state)
        syms = game.training_symmetries(nn_input, np.asarray(pi))
        for inp, p in syms:
            training_data.append((inp, player, p, None))

        action = np.random.choice(len(pi), p=pi)
        state = game.apply(state, action)
        player = state.player

    reward = game.calculate_reward(state)
    return [
        (x[0], A0NNOutput(policy=x[2], value=reward * ((-1) ** (x[1] != player))))
        for x in training_data
    ]
```

- [ ] **Step 4: Update `train` — fix type annotations for training history**

Update the `self_play_data` deque and `training_data` references to use generic `Tuple` instead of `Tuple[A0NNInput, A0NNOutput]`:

```python
self_play_data: Deque[Tuple] = deque(
    [], maxlen=self.training_queue_length
)
```

- [ ] **Step 5: Update `pit` — use eval MCTS params**

Change lines 149-150 from:
```python
prev_mtcs = MonteCarloTreeSearch(self.create_game(), self.pn, self.m_params)
candidate = MonteCarloTreeSearch(self.create_game(), self.nn, self.m_params)
```
To:
```python
prev_mtcs = MonteCarloTreeSearch(self.create_game(), self.pn, self.eval_mcts_params)
candidate = MonteCarloTreeSearch(self.create_game(), self.nn, self.eval_mcts_params)
```

- [ ] **Step 6: Clean up imports**

Update imports at the top: remove `Board` import from `games.game` (no longer needed), remove `A0NNInput` from `types` import (only `A0NNOutput` needed). Add `NDArray` import from `numpy.typing`.

- [ ] **Step 7: Run all tests**

Run: `pipenv run pytest test -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/learners/alpha_zero/alpha_zero.py
git commit -m "feat: refactor AlphaZero to use training_symmetries, to_nn_input, separate MCTS params"
```

---

### Task 7: Update TTT run.py — fix AlphaZero instantiation

**Files:**
- Modify: `src/games/tictactoe/run.py:621-654`

- [ ] **Step 1: Update `alpha_zero_trained_game` to use new `A0Parameters` with both MCTS param sets**

In `src/games/tictactoe/run.py`, the AlphaZero instantiation at line 636-654 changes. Remove the separate `m_params` arg and embed both MCTS param sets in `A0Parameters`:

```python
def alpha_zero_trained_game():
    cur_dir = pathlib.Path(__file__).parent.resolve()
    a0 = AlphaZero(
        TicTacToe,
        lambda: TTTNeuralNetwork(
            params=opt_nn_params, model_folder=f"{cur_dir}/a0_nn_models/"
        ),
        A0Parameters(
            temp_threshold=1,
            pit_games=20,
            pit_threshold=0.55,
            training_episodes=102,
            training_games_per_episode=10,
            training_queue_length=10000,
            training_hist_max_len=20,
            thread_max_workers=8,
            training_mcts_params=training_mcts_params,
            eval_mcts_params=mcts_params,
        ),
        training_examples_folder=f"{cur_dir}/a0_training_examples/",
    )
    a0.train()
    # ... rest of function unchanged
```

- [ ] **Step 2: Update NN `predict` to accept generic input instead of `A0NNInput`**

In `TTTNeuralNetwork.predict` (line 475-478), change the input handling. Currently it expects `List[A0NNInput]` and accesses `.board`. Now it receives raw NDArrays from `to_nn_input`:

```python
def predict(self, inputs):
    boards = np.asarray([i if isinstance(i, np.ndarray) else i.board for i in inputs])
    pis, vs = self.model.predict(boards, verbose=0)
    return [A0NNOutput(policy=pi, value=v) for pi, v in zip(pis, vs)]
```

Also update `train` similarly — the input tuples now contain raw NDArrays instead of `A0NNInput`:

```python
def train(self, data):
    inputs, outputs = list(zip(*data))
    input_boards = np.asarray(
        [i if isinstance(i, np.ndarray) else i.board for i in inputs]
    )
    target_pis = np.asarray([output.policy for output in outputs])
    target_vs = np.asarray([output.value for output in outputs])
    self.model.fit(
        x=input_boards,
        y=[target_pis, target_vs],
        batch_size=self.params.batch_size,
        epochs=self.params.epochs,
        shuffle=True,
    )
```

The `isinstance` check provides backward compatibility with old pickled `A0NNInput` training data.

- [ ] **Step 3: Run all tests**

Run: `pipenv run pytest test -v`
Expected: PASS

- [ ] **Step 4: Run type checker**

Run: `pipenv run mypy src`
Expected: May have some type warnings from the generic refactor. Fix any errors.

- [ ] **Step 5: Commit**

```bash
git add src/games/tictactoe/run.py
git commit -m "feat: update TTT AlphaZero instantiation for new A0Parameters with dual MCTS params"
```

---

### Task 8: Redesign UTTT Neural Network and training config

**Files:**
- Modify: `src/games/ultimate_ttt/run.py`

- [ ] **Step 1: Rewrite `UltimateNeuralNetwork` with new architecture**

Replace the `UltimateNeuralNetwork` class in `src/games/ultimate_ttt/run.py`:

```python
class UltimateNeuralNetwork(NeuralNetwork):
    NUM_FILTERS = 64
    NUM_CONV_LAYERS = 4
    DROPOUT_RATE = 0.05
    LEARN_RATE = 0.001
    BATCH_SIZE = 128
    EPOCHS = 30

    def __init__(self, model_folder: str) -> None:
        super().__init__(model_folder)

        input = Input(shape=(9, 9, 2), name="UltimateBoardInput")
        prev = input

        for _ in range(self.NUM_CONV_LAYERS):
            prev = Activation("relu")(
                BatchNormalization(axis=3)(
                    Conv2D(
                        filters=self.NUM_FILTERS,
                        kernel_size=(3, 3),
                        padding="same",
                    )(prev)
                )
            )

        flat = Flatten()(prev)
        dense1 = Dropout(rate=self.DROPOUT_RATE)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(512)(flat)))
        )
        dense2 = Dropout(rate=self.DROPOUT_RATE)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(256)(dense1)))
        )

        pi = Dense(UltimateTicTacToe.num_actions(), activation="softmax", name="pi")(
            dense2
        )
        v = Dense(1, activation="tanh", name="v")(dense2)

        self.model = Model(inputs=input, outputs=[pi, v])
        self.model.compile(
            loss=["categorical_crossentropy", "mean_squared_error"],
            optimizer=Adam(learning_rate=self.LEARN_RATE),
            metrics={"pi": ["accuracy", "categorical_crossentropy"], "v": ["mse"]},
        )
        self.model.summary()

    def train(self, data):
        inputs, outputs = list(zip(*data))
        input_tensors = np.asarray(list(inputs))
        target_pis = np.asarray([output.policy for output in outputs])
        target_vs = np.asarray([output.value for output in outputs])
        self.model.fit(
            x=input_tensors,
            y=[target_pis, target_vs],
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            shuffle=True,
        )

    def predict(self, inputs):
        tensors = np.asarray(list(inputs))
        pis, vs = self.model.predict(tensors, verbose=0)
        return [A0NNOutput(policy=pi, value=v) for pi, v in zip(pis, vs)]

    # save, load, set_weights, get_weights stay the same
```

- [ ] **Step 2: Update `alpha_zero_trained_game` with new params**

```python
training_mcts_params = MCTSParameters(
    num_searches=200,
    cpuct=1,
    epsilon=1e-4,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
)

eval_mcts_params = MCTSParameters(
    num_searches=1000,
    cpuct=1,
    epsilon=1e-4,
    dirichlet_alpha=0,
    dirichlet_epsilon=0,
)


def alpha_zero_trained_game():
    cur_dir = pathlib.Path(__file__).parent.resolve()
    a0 = AlphaZero(
        UltimateTicTacToe,
        lambda: UltimateNeuralNetwork(model_folder=f"{cur_dir}/a0_nn_models/"),
        A0Parameters(
            temp_threshold=11,
            pit_games=20,
            pit_threshold=0.55,
            training_episodes=200,
            training_games_per_episode=25,
            training_queue_length=25000,
            training_hist_max_len=20,
            thread_max_workers=4,
            training_mcts_params=training_mcts_params,
            eval_mcts_params=eval_mcts_params,
        ),
        training_examples_folder=f"{cur_dir}/a0_training_examples/",
    )
    a0.train()
```

- [ ] **Step 3: Update `vs_alpha_zero_game` to use `eval_mcts_params`**

Replace the hardcoded `MCTSParameters` with the module-level `eval_mcts_params`.

- [ ] **Step 4: Remove commented-out debug code in `main()`**

Clean up the `main()` function. Remove the pickle inspection block (lines 413-425). Keep only the active function calls.

- [ ] **Step 5: Clean up imports**

Remove `Reshape` from Keras imports (no longer needed). Update `A0NNInput` import to `A0NNOutput` only (or keep `A0NNInput` if used in `predict` return). Remove any unused imports.

- [ ] **Step 6: Run all tests**

Run: `pipenv run pytest test -v`
Expected: PASS

- [ ] **Step 7: Run type checker and formatter**

Run: `pipenv run mypy src && pipenv run black src test && pipenv run isort src test --profile black`
Expected: Clean output (fix any issues).

- [ ] **Step 8: Commit**

```bash
git add src/games/ultimate_ttt/run.py
git commit -m "feat: redesign UTTT NN architecture (64 filters, 2-channel input, active_nonant mask)"
```

---

### Task 9: Archive stale UTTT training artifacts

**Files:**
- Move: `src/games/ultimate_ttt/a0_nn_models/` → `src/games/ultimate_ttt/archived_v1/a0_nn_models/`
- Move: `src/games/ultimate_ttt/a0_training_examples/` → `src/games/ultimate_ttt/archived_v1/a0_training_examples/`
- Create: `src/games/ultimate_ttt/archived_v1/README.md`

- [ ] **Step 1: Create archive directory and move artifacts**

```bash
mkdir -p src/games/ultimate_ttt/archived_v1
mv src/games/ultimate_ttt/a0_nn_models src/games/ultimate_ttt/archived_v1/
mv src/games/ultimate_ttt/a0_training_examples src/games/ultimate_ttt/archived_v1/
```

- [ ] **Step 2: Write archive README**

Create `src/games/ultimate_ttt/archived_v1/README.md` with the original training configuration that produced these artifacts:

```markdown
# Archived UTTT AlphaZero Training Artifacts (v1)

These artifacts were produced by the original UTTT AlphaZero training configuration, which had several issues that prevented effective learning. They are kept here for reference before deletion.

## Original Architecture

- Input: (3,3,3,3) reshaped to (9,9,1) — spatial layout was incorrect (row-major reshape scrambled board topology)
- 6 conv layers, 3 filters each, 3x3 kernel
- 3 dense layers: 2048, 1024, 512
- Dropout: 0.3, Learning rate: 0.01, Batch size: 64, Epochs: 10
- No active_nonant encoding — NN could not distinguish forced vs free move constraints
- No Dirichlet noise in MCTS
- MCTS searches: 1000 (both training and evaluation)

## Original Training Parameters

- training_episodes: 200 (only reached episode 12)
- training_games_per_episode: 50
- training_queue_length: 50000
- training_hist_max_len: 50
- thread_max_workers: 8
- temp_threshold: 11
- pit_games: 20, pit_threshold: 0.55

## Results

- 13 episodes completed over ~2 months (June-August 2024)
- Only 4 model improvements saved (episodes 6, 7, 8, 9)
- Agent barely better than random play
```

- [ ] **Step 3: Run all tests**

Run: `pipenv run pytest test -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/games/ultimate_ttt/archived_v1/
git commit -m "chore: archive stale UTTT training artifacts with documentation"
```

---

### Task 10: Final integration verification

**Files:** None new — verification only

- [ ] **Step 1: Run full test suite**

Run: `pipenv run pytest test -v`
Expected: All tests pass.

- [ ] **Step 2: Run full lint pipeline**

Run: `make lint`
Expected: mypy, black, isort, flake8 all pass clean.

- [ ] **Step 3: Verify TTT AlphaZero can still load saved models**

Run: `pipenv run python -c "
from games.tictactoe.run import TTTNeuralNetwork, opt_nn_params
import pathlib
cur_dir = pathlib.Path('src/games/tictactoe').resolve()
nn = TTTNeuralNetwork(params=opt_nn_params, model_folder=f'{cur_dir}/a0_nn_models/')
nn.load('best_model.weights.h5')
print('TTT model loaded successfully')
"`
Expected: Prints "TTT model loaded successfully".

- [ ] **Step 4: Verify UTTT NN can be instantiated with new architecture**

Run: `pipenv run python -c "
from games.ultimate_ttt.run import UltimateNeuralNetwork
nn = UltimateNeuralNetwork(model_folder='/tmp/test_uttt_nn/')
print(f'Model parameters: {nn.model.count_params()}')
print('UTTT NN instantiated successfully')
"`
Expected: Prints parameter count and success message. Verify model size is reasonable (should be much less than the old 38MB).

- [ ] **Step 5: Commit any final fixes**

If any issues were found and fixed, commit them.

```bash
git add -A
git commit -m "fix: final integration fixes for UTTT AlphaZero refactor"
```
