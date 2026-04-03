import pathlib
from typing import NamedTuple

import numpy as np

from games.digit_party.game import DigitParty, DigitPartyPlacement, DigitPartyState
from games.digit_party.train_deep import DigitParty3x3NeuralNetwork, DP3NNParams
from games.game import VALID
from learners.deep_q import DeepQLearner, DeepQParameters

"""
Attempts to train a 3x3 digit party neural network using the deep q learning algorithm,
and with hyperparameters found from bayesian optimization.
"""

# Temporary DQN baseline.
DQN_3X3_NN_PARAMS = DP3NNParams(
    conv_layers=8,
    conv_filters=14,
    dense_layers=1,
    dense_units=337,
    learning_rate=0.0009647204266707786,
    batch_size=64,
    epochs=1,
    dropout_rate=0.0,
    output_activation="linear",
)

TRAINING_EVALUATION_GAMES = 100
TRAINING_EVALUATION_INTERVAL = 500
FINAL_EVALUATION_GAMES = 1000


class DigitPartyEvaluation(NamedTuple):
    games: int
    average_pct: float
    aggregate_pct: float
    average_score: float
    average_theoretical_max: float


def greedy_digit_party_move(
    nn: DigitParty3x3NeuralNetwork, state: DigitPartyState
) -> DigitPartyPlacement:
    out = nn.predict([DigitParty.to_immutable(state)])[0]
    valid_actions = np.flatnonzero(np.asarray(DigitParty.actions(state)) == VALID)
    if len(valid_actions) == 0:
        raise ValueError("No valid actions are available for Digit Party evaluation")

    action = int(valid_actions[int(np.argmax(out.policy[valid_actions]))])
    n = state.board.shape[0]
    return int(action / n), int(action % n)


def evaluate_digit_party(
    nn: DigitParty3x3NeuralNetwork, games: int, n: int
) -> DigitPartyEvaluation:
    game = DigitParty(n=n)
    total_score = 0.0
    total_theoretical_max = 0.0
    total_pct = 0.0

    for _ in range(games):
        game.reset()
        while not game.is_finished():
            r, c = greedy_digit_party_move(nn, game.state())
            game.place(r, c)

        score = float(game.score)
        theoretical_max = float(game.theoretical_max_score())
        total_score += score
        total_theoretical_max += theoretical_max
        total_pct += score / theoretical_max if theoretical_max > 0 else 0.0

    return DigitPartyEvaluation(
        games=games,
        average_pct=100 * total_pct / games,
        aggregate_pct=100 * total_score / total_theoretical_max,
        average_score=total_score / games,
        average_theoretical_max=total_theoretical_max / games,
    )


def digit_party_evaluation_summary(
    nn: DigitParty3x3NeuralNetwork, games: int, n: int
) -> str:
    evaluation = evaluate_digit_party(nn=nn, games=games, n=n)
    return (
        f"{evaluation.games} games, "
        f"avg_pct={evaluation.average_pct:.2f}%, "
        f"aggregate_pct={evaluation.aggregate_pct:.2f}%, "
        f"avg_score={evaluation.average_score:.2f}/"
        f"{evaluation.average_theoretical_max:.2f}"
    )


def deep_q_3x3_trained_game():
    cur_dir = pathlib.Path(__file__).parent.resolve()
    nn = DigitParty3x3NeuralNetwork(
        params=DQN_3X3_NN_PARAMS, model_folder=f"{cur_dir}/deepq_3x3_models/"
    )
    target_nn = DigitParty3x3NeuralNetwork(
        params=DQN_3X3_NN_PARAMS, model_folder=f"{cur_dir}/deepq_3x3_models/"
    )
    nn.summary()
    deepq = DeepQLearner(
        DigitParty(n=3),
        nn,
        target_nn,
        DeepQParameters(
            alpha=0.1,
            gamma=0.9,
            min_epsilon=0.01,
            max_epsilon=1,
            epsilon_decay=0.0005,
            valid_action_reward=0.01,
            memory_size=20_000,
            min_replay_size=512,
            minibatch_size=64,
            steps_to_train_longterm=4,
            steps_to_train_shortterm=0,
            steps_per_target_update=250,
            training_episodes=10_000,
            episodes_per_model_save=1_000,
            episodes_per_memory_save=1_000,
            episodes_per_stats_print=100,
            episodes_per_evaluation=TRAINING_EVALUATION_INTERVAL,
        ),
        memory_folder=f"{cur_dir}/deepq_3x3_memory/",
        evaluator=lambda: digit_party_evaluation_summary(
            nn=nn, games=TRAINING_EVALUATION_GAMES, n=3
        ),
    )
    deepq.train()

    print(
        "Final evaluation: "
        + digit_party_evaluation_summary(nn=nn, games=FINAL_EVALUATION_GAMES, n=3)
    )


def main() -> None:
    deep_q_3x3_trained_game()
