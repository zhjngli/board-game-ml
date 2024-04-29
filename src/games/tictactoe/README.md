# Tic Tac Toe training writeup

Tic Tac Toe is an exceedingly simple game, using the simple [monte carlo](/src/learners/monte_carlo.py) learning method is sufficient to train a pretty good agent. I went on with the [alpha zero](/src/learners/alpha_zero/alpha_zero.py) training method to train a neural network to play tic tac toe as a proof of concept that my implementation of alpha zero is correct. For sufficiently well-trained agents, I should expect that pitting these agents against each other results in 100% ties, no matter which agent goes first.

## Monte Carlo training

The monte carlo agents were trained on ~10,000 episodes, and it trains pretty quickly. Subsequent training episodes have close to 0 marginal value. I did notice that the training would probably require less episodes if I pass in all symmetric game states to the monte carlo training process, since the process will learn all the game symmetries at once instead of relying on randomness to happen upon a symmetry that it's already learned. That would be a good TODO.

## Alpha Zero training

Initially I guessed at some neural network parameters for Alpha Zero to train the network. I noticed that the loss from the training wasn't great, so I used the training examples from the Alpha Zero output to search for more optimal hyperparameters. (These trials from running the bayesian optimization are also saved in [`trials.pkl`](/src/games/tictactoe/trials.pkl).) The hyperparameters I used for the Alpha Zero algorithm can be found in [`run.py`](/src/games/tictactoe/run.py).

## Q-learning

While I _think_ it is theoretically possible to use q-learning to train a tic tac toe agent, it seems infeasible in practice because it doesn't backpropagate the rewards that it receives. Because rewards are only given at end states, the training process relies on randomness to land on a state which potentially leads to an end state with a previously calculated q-value. I left the code for q-learning in [`run.py`](/src/games/tictactoe/run.py) simply as a reference.

## Results

After training, I pitted the alpha zero agent against the monte carlo agent, and found the expected results, where all games end in ties. One thing to note however, is that the alpha zero agent heavily depends on how many searches we perform when running the monte carlo tree search. When playing, I ran 1000 searches, however, while training I only ran 100 searches to speed up the training process.
