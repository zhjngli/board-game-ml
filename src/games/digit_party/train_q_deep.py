import pathlib

import numpy as np

from games.digit_party.game import DigitParty, DigitPartyPlacement, DigitPartyState
from games.digit_party.run_helpers import computer_game
from games.digit_party.train_deep import DigitParty3x3NeuralNetwork, opt_nn_params
from games.game import VALID
from learners.deep_q import DeepQLearner, DeepQParameters

"""
Attempts to train a 3x3 digit party neural network using the deep q learning algorithm,
and with hyperparameters found from bayesian optimization.
"""

def deep_q_3x3_trained_game():
    cur_dir = pathlib.Path(__file__).parent.resolve()
    nn = DigitParty3x3NeuralNetwork(
        params=opt_nn_params, model_folder=f"{cur_dir}/deepq_3x3_models/"
    )
    deepq = DeepQLearner(
        DigitParty(n=3),
        nn,
        DeepQParameters(
            alpha=0.001,
            gamma=0.9,
            min_epsilon=0.01,
            max_epsilon=1,
            epsilon_decay=0.01,
            valid_action_reward=0.01,
            memory_size=1800,
            min_replay_size=900,
            minibatch_size=900,
            steps_to_train_longterm=9,
            steps_to_train_shortterm=1,
            steps_per_target_update=900,
            training_episodes=500,
            episodes_per_model_save=100,
            episodes_per_memory_save=100,
        ),
        memory_folder=f"{cur_dir}/deepq_3x3_memory/",
    )
    deepq.train()

    g = DigitParty(n=3)
    random = 0
    prediction = 0

    def deepq_play(state: DigitPartyState) -> DigitPartyPlacement:
        nonlocal random, prediction
        pi = nn.predict([state])[0]
        action_statuses = np.asarray(g.actions(state))
        valid_actions = np.where(action_statuses == VALID)[0]
        # a = valid_actions[np.argmax(pi[valid_actions])]
        a = np.argmax(pi)
        if not np.isin(valid_actions, a).any():
            a = np.random.choice(valid_actions)
            print("######### CHOSE RANDOM ACTION", a)
            random += 1
        else:
            prediction += 1

        n = state.board.shape[0]
        r = int(a / n)
        c = int(a % n)
        return r, c

    computer_game(g, 100, deepq_play)
    print(f"random: {random}")
    print(f"prediction: {prediction}")
