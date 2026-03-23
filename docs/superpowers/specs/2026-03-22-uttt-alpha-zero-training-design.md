# Ultimate Tic-Tac-Toe AlphaZero Training Improvements

## Problem

The UTTT AlphaZero training produces an agent barely better than random play after 13 episodes over ~2 months of local training on an M1 Pro MacBook (32GB RAM). Root causes identified:

1. The NN never sees `active_nonant`, so it can't distinguish board states with different legal move constraints
2. Conv filters (3) are too few to learn spatial patterns on a 9x9 board
3. The `(3,3,3,3)` → `(9,9)` reshape scrambles board topology, breaking conv spatial reasoning
4. No Dirichlet noise in MCTS means self-play games are too deterministic
5. 1000 MCTS searches during self-play makes each episode take days
6. High dropout (0.3) and learning rate (0.01) impede learning
7. `A0NNInput` is a concrete type with only a `board` field, preventing game-specific NN inputs

## Approach

Architecture-first: fix the NN and input representation, then tune parameters, then add algorithmic improvements.

## Design

### 1. Generic A0NNInput refactor

Make `A0NNInput` a generic type parameter so each game defines its own NN input shape.

**Changes to base classes:**

- `types.py`: Replace concrete `A0NNInput(board: Board)` with a `TypeVar` (e.g., `NNInput = TypeVar("NNInput")`). Keep `A0NNOutput` as-is. Keep the old `A0NNInput` class as a deprecated alias for backward pickle compatibility (see Saved Artifacts below).
- `game.py`: Add `NNInput` TypeVar. Add new abstract method:
  ```python
  @staticmethod
  @abstractmethod
  def to_nn_input(state: State) -> NNInput:
      pass
  ```
- `monte_carlo_tree_search.py`: Add `NNInput` generic parameter. Replace `A0NNInput(board=state.board)` with `self.game.to_nn_input(state)`.
- `alpha_zero.py`: Add `NNInput` generic parameter. Accept separate training and evaluation MCTS params (see Section 4). Use `game.to_nn_input()` for training data generation.
- `nn/neural_network.py`: Already generic on `Input`/`Output`, no changes needed beyond type param alignment.

**Game implementations:**

- TTT: `to_nn_input` returns `state.board` (a `(3,3)` NDArray). Behaviorally identical to today.
- UTTT: `to_nn_input` returns a `(9,9,2)` NDArray (board + active_nonant mask). See Section 3.
- Digit Party: Trivial implementation returning `state.board`.

Note: Random Walk does not extend `Game` (it is a standalone class), so it requires no changes.

**Saved artifacts:** TTT model weights (`.weights.h5`) remain fully compatible — they store numpy arrays with no knowledge of Python types. TTT training examples (`.pkl`) contain pickled `A0NNInput` NamedTuple objects. To maintain backward compatibility, keep the old `A0NNInput` class in `types.py` as a deprecated stub so existing `.pkl` files still deserialize. New training code will use the game-specific `NNInput` type via `to_nn_input()`. UTTT artifacts are incompatible with the new NN architecture regardless and will be archived (see Section 6).

### 2. Replace `symmetries_of` with `training_symmetries`

Replace the existing `symmetries_of(a: NDArray) -> List[NDArray]` abstract method on `Game` with:

```python
@staticmethod
@abstractmethod
def training_symmetries(nn_input: NNInput, policy: NDArray) -> List[Tuple[NNInput, NDArray]]:
    pass
```

This returns paired (nn_input, policy) symmetries, guaranteeing the same transformation is applied to both. **The returned list MUST include the identity (un-transformed) pair as one of its elements**, since `alpha_zero.py` will only append from this list — there is no separate append for the original data point.

**Rationale:** `symmetries_of` is only called in `alpha_zero.py` for training data augmentation. It's a no-op in Digit Party. The new method is more explicit about its purpose and handles the case where NN input shape differs from board/policy shape (e.g., UTTT's `(9,9,2)` input vs `(81,)` policy).

**Implementations:**

- TTT: Rotate/flip `(3,3)` board and `(9,)` policy (reshaped to `(3,3)` for rotation). Same logic as today. Note: current `symmetries_of` iterates `range(1, 5)` where rotation by 4 is the identity, so the identity pair is already included.
- UTTT: Rotate/flip `(9,9,2)` input using `np.rot90(nn_input, i, axes=(0, 1))` on spatial axes. Because the input is a `(9,9,2)` tensor, both channels (board and active_nonant mask) rotate together automatically — no separate handling needed for the mask channel. This works because `to_nn_input` has already converted the hierarchical `(3,3,3,3)` representation into a flat `(9,9)` spatial grid via transpose, so a single 2D rotation on the `(9,9)` grid is equivalent to the nested inner+outer rotation that `symmetries_of` currently performs on `(3,3,3,3)`. Rotate/flip `(81,)` policy reshaped to `(3,3,3,3)` using existing inner+outer rotation logic, then flatten back to `(81,)`. Note: the policy must use `(3,3,3,3)` rotation (not `(9,9)`) because action indices are defined by the hierarchical `[R][C][r][c]` mapping in `to_action`/`from_action`.
- Digit Party: Return list containing only the identity pair `[(nn_input, policy)]` (no augmentation).

`alpha_zero.py` training loop changes from:
```python
bs = game.symmetries_of(oriented_state.board)
pis = game.symmetries_of(np.asarray(pi))
for b, p in zip(bs, pis):
    training_data.append((b, player, p, None))
```
To:
```python
nn_input = game.to_nn_input(oriented_state)
syms = game.training_symmetries(nn_input, np.asarray(pi))
for inp, p in syms:
    training_data.append((inp, player, p, None))
```

### 3. UTTT Neural Network redesign

**Input encoding:**

`to_nn_input` produces a `(9,9,2)` tensor:
- Channel 0: Board state. `state.board.transpose(0, 2, 1, 3).reshape(9, 9)` — the transpose ensures correct spatial layout where `board[R][C][r][c]` maps to `grid[R*3+r][C*3+c]`. Without the transpose, Keras `Reshape` uses row-major ordering, which maps each section to a full row of the 9x9 grid — cells that are physically adjacent across section boundaries (e.g., bottom of section (0,0) and top of section (1,0)) end up far apart, and a 3x3 conv kernel cannot see them together. Values are -1 (opponent), 0 (empty), 1 (current player) after `orient_state`.
- Channel 1: Active nonant mask. A 9x9 binary grid. If `active_nonant = (R, C)`, the 3x3 region at rows `R*3:(R+1)*3`, cols `C*3:(C+1)*3` is 1, rest is 0. If `active_nonant is None` (free choice), all 1s.

The NN input shape changes from `Input(shape=(3,3,3,3))` + `Reshape((9,9,1))` to `Input(shape=(9,9,2))` directly.

**Architecture:**

| Component | Current | New |
|-----------|---------|-----|
| Input shape | (3,3,3,3) reshaped to (9,9,1) | (9,9,2) |
| Conv layers | 6 layers, 3 filters each | 4 layers, 64 filters each |
| Conv kernel | 3x3, same padding | 3x3, same padding (unchanged) |
| Conv pattern | BatchNorm(axis=3) + ReLU | Same (unchanged) |
| Dense layers | 3 layers: 2048 → 1024 → 512 | 2 layers: 512 → 256 |
| Dense pattern | BatchNorm(axis=1) + ReLU + Dropout | Same (unchanged) |
| Policy head | Dense(81, softmax) | Same (unchanged) |
| Value head | Dense(1, tanh) | Same (unchanged) |
| Dropout | 0.3 | 0.05 |
| Learning rate | 0.01 | 0.001 |
| Batch size | 64 | 128 |
| Epochs | 10 | 30 |

Estimated model size: approximately 2-5MB (down from 38MB). The savings come primarily from the dense layer reduction (2048/1024/512 → 512/256), which dominates parameter count. The conv layers have more filters (64 vs 3) but 3x3 kernels are small. Exact size should be verified after implementation.

### 4. MCTS improvements

**Separate training vs evaluation params:**

`AlphaZero.__init__` currently takes a single `m_params: MCTSParameters`. Replace this with two fields in `A0Parameters`: `training_mcts_params` for self-play and `eval_mcts_params` for pit evaluation and post-training play. This keeps all training config in one NamedTuple.

Self-play in `train_once()` uses `training_mcts_params`. Pit evaluation in `pit()` uses `eval_mcts_params`.

```python
training_mcts_params = MCTSParameters(
    num_searches=200,    # fast self-play (was 1000)
    cpuct=1,
    epsilon=1e-4,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
)

eval_mcts_params = MCTSParameters(
    num_searches=1000,   # thorough evaluation/play
    cpuct=1,
    epsilon=1e-4,
    dirichlet_alpha=0,   # no noise during evaluation
    dirichlet_epsilon=0,
)
```

**Dirichlet noise:**

Add `dirichlet_alpha` and `dirichlet_epsilon` fields to `MCTSParameters` (defaulting to 0 for backward compatibility).

Inject noise inside `search()` when a new root node is first expanded — specifically in the `if ir not in self.ps` block at `monte_carlo_tree_search.py:83`. After the policy is normalized and stored in `self.ps[ir]`, check if this is the root node (passed as a flag or detected by depth). If so and `dirichlet_alpha > 0`, mix in noise:

```python
if is_root and self.dirichlet_alpha > 0:
    valid_indices = [a for a in range(len(self.ps[ir])) if self.vas[ir][a]]
    noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_indices))
    valid_noise = np.zeros_like(self.ps[ir])
    for i, idx in enumerate(valid_indices):
        valid_noise[idx] = noise[i]
    self.ps[ir] = (1 - self.dirichlet_epsilon) * self.ps[ir] + self.dirichlet_epsilon * valid_noise
```

Injecting inside `search` at the expansion point is cleaner than injecting between iterations in `action_probabilities`, because the root node is only expanded once (the `if ir not in self.ps` guard), so the noise is applied exactly once. The `search()` method signature adds an `is_root: bool = False` parameter; `action_probabilities` passes `is_root=True` on each call to `self.search(state)`. The recursive call `self.search(next_s)` within `search()` must NOT pass `is_root=True` — the default `False` ensures noise is only applied at the root.

### 5. AlphaZero training parameters

| Parameter | Current | New | Rationale |
|-----------|---------|-----|-----------|
| `training_games_per_episode` | 50 | 25 | Cheaper per-game (200 vs 1000 searches). 25 games * 8 symmetries * ~50 moves = ~10k examples per episode |
| `training_queue_length` | 50000 | 25000 | Scaled with fewer games |
| `training_hist_max_len` | 50 | 20 | Keep training data fresher as model improves |
| `thread_max_workers` | 8 | 4 | Bigger NN per thread; 4 is safer for 32GB RAM |
| `temp_threshold` | 11 | 11 | Unchanged, reasonable for ~50 move games |
| `pit_games` | 20 | 20 | Unchanged |
| `pit_threshold` | 0.55 | 0.55 | Unchanged |
| `training_episodes` | 200 | 200 | Unchanged |

### 6. Cleanup

**Remove commented-out debug code** in `ultimate_ttt/run.py:413-425` (the pickle inspection block and `monte_carlo_trained_game` call).

**Draw reward:** Change UTTT `calculate_reward` from `return 0.1` to `return 0` for draws. Leave TTT unchanged.

**MCTS terminal detection fix (required for draw reward = 0):** The current `search()` in `monte_carlo_tree_search.py:74-81` uses `self.evs[ir] != 0` to detect terminal states:
```python
if ir not in self.evs:
    self.evs[ir] = (
        self.game.calculate_reward(state)
        if self.game.check_finished(state)
        else 0
    )
if self.evs[ir] != 0:
    return -self.evs[ir]
```
If `calculate_reward` returns `0` for a draw, the terminal state gets `evs[ir] = 0`, the `!= 0` check fails, and the search continues past the terminal state — attempting to expand children of a finished game. Fix: add a separate `self.terminals: Set[Immutable]` set that tracks which states are terminal, independent of the reward value. Replace the terminal detection with:
```python
if ir not in self.terminals:
    if self.game.check_finished(state):
        self.terminals.add(ir)
        self.evs[ir] = self.game.calculate_reward(state)
    else:
        self.evs[ir] = 0
if ir in self.terminals:
    return -self.evs[ir]
```

Note: the existing `pit()` method computes win ratio as `p1wins / (p1wins + p2wins)`, excluding draws. This means if both models are strong enough to mostly draw, the pit comparison has a small decisive sample. This is a pre-existing issue and not caused by this change, but worth noting for future improvement.

**Archive stale training artifacts:** Move `src/games/ultimate_ttt/a0_nn_models/` and `src/games/ultimate_ttt/a0_training_examples/` to `src/games/ultimate_ttt/archived_v1/`. Add a README in `archived_v1/` summarizing the original architecture and parameters that produced these artifacts.

## Pre-existing issues noted (out of scope)

- `tictactoe/run.py:422`: The conv layer loop iterates `range(self.params.conv_filters)` but should likely be `range(self.params.conv_layers)`. Pre-existing bug, not caused by this change.
- Digit Party's state includes `next` (upcoming digits) which could be NN-relevant. Not addressed here since Digit Party doesn't use AlphaZero, but `to_nn_input` could incorporate it in the future.

## Files changed

| File | Type of change |
|------|---------------|
| `src/learners/alpha_zero/types.py` | `A0NNInput` becomes TypeVar `NNInput`; keep old `A0NNInput` as deprecated stub for pickle compat |
| `src/games/game.py` | Add `NNInput` TypeVar, `to_nn_input()` abstract method, replace `symmetries_of` with `training_symmetries` |
| `src/learners/alpha_zero/monte_carlo_tree_search.py` | Add `NNInput` generic param, use `game.to_nn_input()`, add Dirichlet noise in `search()`, add new fields to `MCTSParameters`, fix terminal detection to use `self.terminals` set instead of `evs[ir] != 0` |
| `src/learners/alpha_zero/alpha_zero.py` | Add `NNInput` generic param, accept separate training/eval MCTS params, use `to_nn_input()` and `training_symmetries()` |
| `src/games/tictactoe/tictactoe.py` | Implement `to_nn_input`, `training_symmetries` |
| `src/games/tictactoe/run.py` | Update type params and imports on AlphaZero/NN instantiation |
| `src/games/ultimate_ttt/ultimate.py` | Implement `to_nn_input` (with correct transpose + nonant mask), `training_symmetries`, change draw reward to 0 |
| `src/games/ultimate_ttt/run.py` | Redesign NN architecture, update A0/MCTS params, separate training vs eval MCTS, remove commented code, update imports |
| `src/games/digit_party/game.py` | Replace `symmetries_of` with identity-only `training_symmetries`, add trivial `to_nn_input` |
| `src/games/ultimate_ttt/archived_v1/` | New directory with moved artifacts and README |

## What is NOT changing

- TTT model weights and training data remain compatible and untouched
- `A0NNOutput` stays the same (policy + value)
- Game logic (rules, `apply`, `actions`, `check_finished`) unchanged for all games
- AlphaZero training loop structure (self-play → train → pit) unchanged
- Monte Carlo and Q-learning code paths unaffected
- Random Walk (does not extend `Game`, unaffected)
