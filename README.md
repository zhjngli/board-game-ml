# board-game-ml
An exploration of machine learning techniques (mostly reinforcement learning) applied to simple games.

## How to run

There's a very simple cli used to run the games. `make run` is a thin alias around `pipenv run python src`, which runs the cli defined in [`src/__main__.py`](src/__main__.py). The cli then calls the `main()` function in the corresponding game's run file. For example, `make run ARGS="ttt"` runs the `main()` function in [`src/games/tictactoe/run.py`](src/games/tictactoe/run.py).

Some policy files are saved along with the repo, check each specific game for its corresponding policy files.

## What's been done

### ML methods

The following two methods currently use native python, without any fancy libraries or frameworks. It's inefficient, but forces me to consider the details.

- Simple [Q-learning](src/learners/q.py), which attempts to track the whole q-table. As opposed to deep q-learning which uses a neural network to guess the best possible action given a state. Works well when rewards can be calculated per action.
- [Monte carlo](src/learners/monte_carlo.py) learning. Works for episodic rewards, where the value of each state/action is primarily determined after the game is complete. Stores all the states that it's explored, and the associated value of each state.

The next two methods use a form of deep reinforcement learning with the introduction of neural networks.

- [Alpha Zero](src/learners/alpha_zero/). Combines a neural network with monte carlo tree search so that the training is more optimized. Only needs to store the neural network weights (and self-play training examples, if preferred), so it's much more memory efficient than storing all the explored states.
- [Deep Q Learning](src/learners/deep_q.py). Similar to q-learning, but it replaces the q-table with a neural network, drastically reducing the memory usage.

### Bayesian Optimization

I use bayesian optimization to tune my network parameters. It's not the most efficient since:
1. The parameter space can be pretty large, and I have limited resources/time to search the whole space. And all this time is spent not even to train the network.
1. Generally I need an objective set of training data to ensure that the objective function is accurate (minimizing the loss on a suboptimal set of training data is pointless). This is particularly tough for alpha zero cases where the network generates its own training data.

However, bayesian optimization does seem promising for training Digit Party, as I have an objective set of training data from running naive q-learning.

### Games
- A [random walk game](src/games/random_walk/README.md) which scores points at the right bound and loses points at the left bound. This is trained with q-learning and a monte carlo method.
- [Tic-tac-toe](src/games/tictactoe/README.md). This game can be successfully trained using the monte carlo learning method, and with alpha zero.
- [Digit party](src/games/digit_party/README.md). Though naive q-learning should be able to learn this game, it is intractable as the number of states is just too high. Deep q-learning should work better.
- [Ultimate tic tac toe](src/games/ultimate_ttt/ultimate.py). Works ok with simple Monte Carlo method but even better with Alpha Zero.

### TODO

#### Games

Consider implementing the following games and train them using an appropriate method:
- [nim](https://en.wikipedia.org/wiki/Nim)
- [photosynthesis](https://boardgamegeek.com/boardgame/218603/photosynthesis)

#### Tech features/infra
- refactor simple q and monte-carlo learners to use composition rather than inheritance. been lazy to do this
- clean up functions in Ultimate/TTT game classes. function names could be better, and ordering/structuring could be better
- logging with verbose option
- expanding and optimizing the CLI
- organize types

## References
- [TicTacToe](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter01/tic_tac_toe.py) with a simple Monte Carlo learning implementation
- [RL with TicTacToe](https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542)
- [Ultimate TicTacToe](https://github.com/Arnav235/ultimate_tic-tac-toe_alphazero) with the AlphaZero algorithm
- [Monte Carlo Tree Search](https://blog.theofekfoundation.org/artificial-intelligence/2016/06/27/what-is-the-monte-carlo-tree-search/)
- [AlphaZero tutorial](https://web.stanford.edu/~surag/posts/alphazero.html)
- [AlphaZero general](https://github.com/kevaday/alphazero-general/blob/main/README.md)
- More [Monte Carlo Tree Search](https://web.archive.org/web/20180629082128/http://mcts.ai/index.html)
- Original [AlphaZero paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
- [Deep Q Learning](https://towardsdatascience.com/how-to-teach-an-ai-to-play-games-deep-reinforcement-learning-28f9b920440a)

## Technical Considerations

### Typing

For bigger projects like these, I'm not a fan of how Python is dynamically typed. Tracking down errors in runtime is frustrating, and sometimes they don't even appear because of how flexible Python syntax can be, so it becomes an obscure bug. Plus, defining types lets me think clearly about the objects I'm handling, as well as the function pre/post-conditions.

However, typing in Python can only go so far. For example, it's tough to enforce the shape and type of values in an NDArray. So when we try to modify a nested value in the array, mypy will think it's an Any type and be unable to check subsequent usages of the nested value. This is especially apparent when implementing game logic that uses numpy arrays extensively. There are some libraries that can mitigate this to some extent, but none are perfect. I can dream about a beautifully statically typed landscape like Haskell but that likely won't happen in Python.
