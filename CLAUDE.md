# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning techniques applied to simple board games. Python 3.11 with pipenv for dependency management. Uses TensorFlow/Keras for neural networks.

## Commands

```bash
make install              # Install dependencies via pipenv
make run ARGS="ttt"       # Run a game (see CLI aliases below)
make test                 # Run pytest + lint
make lint                 # mypy + black + isort + flake8
make format               # black + isort only
make type-check           # mypy only
pipenv run pytest test/games/test_ultimate.py  # Run a single test file
```

All commands go through `pipenv run`. The CLI entry point is `src/__main__.py` → `src/cli.py`.

### CLI Aliases

| Command | Aliases |
|---------|---------|
| `random_walk` | `rw` |
| `tictactoe` | `ttt` |
| `ultimate_ttt` | `u`, `ult` |
| `digit_party_simple_q` | `dpq`, `dp_simple_q` |
| `digit_party_deep` | `dpd`, `dp_deep` |
| `digit_party_deep_q` | `dpdq`, `dp_deep_q` |

## Architecture

### Game Abstraction (`src/games/game.py`)

All games implement the abstract `Game[State, Immutable]` class with static methods for state manipulation. Key design decisions:
- **State is always passed as a parameter** to static methods (`apply`, `actions`, `check_finished`, `calculate_reward`) rather than stored as mutable instance state
- `orient_state()` normalizes state to Player 1's perspective for consistent NN input
- `symmetries_of()` generates board/policy symmetries (rotations, reflections) to augment training data
- `Action` is always `int` (index into action space); `Board` is `NDArray`
- Two-player games use `Player = Literal[1, -1]` convention

### Learning Algorithms (`src/learners/`)

Four learners, progressing in sophistication:
1. **SimpleQLearner** (`q.py`) — tabular Q-learning, native Python only
2. **MonteCarloLearner** (`monte_carlo.py`) — episodic rewards, native Python only
3. **DeepQLearner** (`deep_q.py`) — DQN with replay memory, epsilon decay, target network
4. **AlphaZero** (`alpha_zero/`) — NN + MCTS self-play with threading, pit evaluation against previous model

AlphaZero training loop: self-play (threaded) → train NN → pit new vs old model → keep if win rate > threshold. Models save as `ep_NNNNNNN_model.weights.h5` and `best_model.weights.h5`.

### Neural Network Abstraction (`src/nn/neural_network.py`)

Generic `NeuralNetwork[Input, Output]` ABC. Each game provides its own concrete NN implementation (e.g., `src/games/tictactoe/neural_network.py`). Models are saved to per-game `a0_nn_models/` or `deepq_*_models/` directories.

### Game Implementations

Each game lives in `src/games/<name>/` with its own `run.py` (entry point), game class, and optionally NN and learner specializations. Pre-trained model weights and training examples (`.pkl`) are stored alongside game code.

## Code Style

- Strict mypy type checking with numpy plugin enabled
- Black formatting (line length 88) + isort with black profile
- Flake8 with E203, E501, W503 ignored (black compatibility)
- Extensive use of generics (`TypeVar`) for type-safe game/learner abstractions
- Simple Q-learning and Monte Carlo intentionally avoid frameworks to understand algorithmic details
