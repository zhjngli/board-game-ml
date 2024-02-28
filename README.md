# board-game-ml
An exploration of machine learning techniques (mostly reinforcement learning) applied to simple games.

## How to run
Update [`src/__main__.py`](src/__main__.py) with whatever game you'd like to train and run `make run`. See the existing [file](src/__main__.py) for an example.

All policy files are saved locally, so cloning the repo fresh will require training each agent.

## What's been done

### ML methods

- Simple [Q-learning](src/learners/q.py), which attempts to track the whole q-table. As opposed to deep q-learning which uses a neural network to guess the best possible action given a state. Works well when rewards can be calculated per action.
- [Monte carlo](src/learners/monte_carlo.py) learning. Works for episodic rewards, where the value of each state/action is primarily determined after the game is complete.

Both of these methods currently use native python, without any fancy libraries or frameworks. It's inefficient, but forces me to consider the details.

### Games
- A [random walk game](src/random_walk/random_walk.py) which scores points at the right bound and loses points at the left bound. This is trained with q-learning and a monte carlo method.
- [Tic-tac-toe](src/tictactoe/tictactoe.py). This game can be successfully trained using the monte carlo learning method.
- [Digit party](src/digit_party/digit_party.py). Though naive q-learning should be able to learn this game, it is intractable as the number of states is just too high. Deep q-learning should work better.

## TODO
- consider implementing the following games and train them
  - [nim](https://en.wikipedia.org/wiki/Nim)
  - [ultimate tic tac toe](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe)
  - [photosynthesis](https://boardgamegeek.com/boardgame/218603/photosynthesis)
- deep q-learning

## References
- [TicTacToe](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter01/tic_tac_toe.py) with a simple Monte Carlo learning implementation
- [Ultimate TicTacToe](https://github.com/Arnav235/ultimate_tic-tac-toe_alphazero) with the AlphaZero algorithm
- [Monte Carlo Tree Search](https://blog.theofekfoundation.org/artificial-intelligence/2016/06/27/what-is-the-monte-carlo-tree-search/)
- [AlphaZero tutorial](https://web.stanford.edu/~surag/posts/alphazero.html)
- More [Monte Carlo Tree Search](https://web.archive.org/web/20180629082128/http://mcts.ai/index.html)
- Original [AlphaZero paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
- [Deep Q Learning](https://towardsdatascience.com/how-to-teach-an-ai-to-play-games-deep-reinforcement-learning-28f9b920440a)
