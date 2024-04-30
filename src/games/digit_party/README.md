# Digit Party training writeup

Digit party as a game scales with complexity very very quickly, simply because of the number of available actions, as well as the number of possible digits. Because of this, I started with training a 3x3 game instead of the full 5x5 game that I first played.

## Simple Q-learning

### 3x3

For the 3x3 game, this works reasonably well. For a totally untrained agent (placing digits randomly), here's the distribution of scores over about ~100,000 games:

![an untrained agent playing 100,000 3x3 games follows a normal distribution around 50% score](./q-3x3-untrained.png "untrained 3x3 agent: games played per percent score")

After training the simple q-agent on about 20,000,000 games, here is the resulting distribution:

![a trained agent playing 100,000 3x3 games averages 90% and gets 100% more than 35% of the time](./q-3x3.png "trained 3x3 agent: games played per percent score")

While this is an impressive result, the resulting policy file is nearly 5 gigabytes, and for that reason I haven't uploaded it or stored it anywhere.

### 5x5

For the 5x5 game, using simple q-learning is pretty much intractable. There are just far too many state-action pairs to keep track of in a single policy file. It's theoretically doable, but I would need a lot of compute and memory to handle the training. I also think that it would take a few orders of magnitude more than 20,000,000 episodes to achieve a similar result as in the 3x3 case. Here's the distribution of scores of an untrained agent playing about 10,000 5x5 games:

![an untrained agent playing 10,000 5x5 games follows a normal distribution around 27% score](./q-5x5-untrained.png)

## Deep Q-learning

Here's where deep q-learning comes into play. The idea is that after training, the neural network can output very similar policies that result from simple q-learning, but without all that memory overhead. This is still a work in progress to find the proper neural network architecture and hyperparameters.

### 3x3

I started with guessing some architecture and hyperparameters for the neural network, but found that while training, it was totally overfitting, and not learning properly at all. In many cases, the agent would have to fall back to a random action because the action it chose was actually invalid. So essentially, even after training, the agent wasn't much better than a random agent.

Luckily, after training a simple q agent to play the 3x3 game, I have nearly 5 gigabytes of objective training data that can be used search for optimal hyperparameters. I've currently "chunked" the 5 gigabytes of training data, and using 1% to run a bayesian optimization algorithm to find optimal hyperparameters for the network.