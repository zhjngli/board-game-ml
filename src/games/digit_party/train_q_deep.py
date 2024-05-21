import pathlib

from games.digit_party.game import DigitParty
from games.digit_party.train_deep import (
    DigitParty3x3NeuralNetwork,
    deep_play_digit_party,
    opt_nn_params,
)
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
    target_nn = DigitParty3x3NeuralNetwork(
        params=opt_nn_params, model_folder=f"{cur_dir}/deepq_3x3_models/"
    )
    deepq = DeepQLearner(
        DigitParty(n=3),
        nn,
        target_nn,
        DeepQParameters(
            alpha=0.1,
            gamma=0.9,
            min_epsilon=0.01,
            max_epsilon=1,
            epsilon_decay=0.01,
            valid_action_reward=0.01,
            memory_size=15041 * 10,
            min_replay_size=900,
            minibatch_size=15041,
            steps_to_train_longterm=15041,
            steps_to_train_shortterm=0,
            steps_per_target_update=15041 * 2,
            training_episodes=5000,
            episodes_per_model_save=200,
            episodes_per_memory_save=200,
        ),
        memory_folder=f"{cur_dir}/deepq_3x3_memory/",
    )
    deepq.train()

    deep_play_digit_party(games=1000, n=3, nn=nn)


def main() -> None:
    deep_q_3x3_trained_game()
