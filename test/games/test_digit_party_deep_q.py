from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from games.digit_party.game import DigitPartyState
from games.game import P1
from learners.deep_q import DQNOutput


def test_digit_party_deep_q_defaults_to_3x3_best_eval() -> None:
    from games.digit_party import train_q_deep

    with (
        patch.object(train_q_deep, "eval_3x3_best") as eval_3x3_best,
        patch.object(train_q_deep, "train_3x3_dqn") as train_3x3_dqn,
        patch.object(train_q_deep, "eval_5x5_best") as eval_5x5_best,
        patch.object(train_q_deep, "train_5x5_dqn") as train_5x5_dqn,
    ):
        train_q_deep.main(SimpleNamespace(size="3", mode="eval-best", games=1000))

    eval_3x3_best.assert_called_once_with(games=1000)
    train_3x3_dqn.assert_not_called()
    eval_5x5_best.assert_not_called()
    train_5x5_dqn.assert_not_called()


def test_digit_party_deep_q_dispatches_5x5_train_override() -> None:
    from games.digit_party import train_q_deep

    with (
        patch.object(train_q_deep, "eval_3x3_best") as eval_3x3_best,
        patch.object(train_q_deep, "train_3x3_dqn") as train_3x3_dqn,
        patch.object(train_q_deep, "eval_5x5_best") as eval_5x5_best,
        patch.object(train_q_deep, "train_5x5_dqn") as train_5x5_dqn,
    ):
        train_q_deep.main(
            SimpleNamespace(size="5", mode="train", games=250, episodes=123)
        )

    eval_3x3_best.assert_not_called()
    train_3x3_dqn.assert_not_called()
    eval_5x5_best.assert_not_called()
    train_5x5_dqn.assert_called_once_with(episodes=123)


def test_digit_party_5x5_dqn_normalizes_inputs() -> None:
    from games.digit_party.train_q_deep_5x5 import (
        DQN_5X5_NN_PARAMS,
        DigitParty5x5DeepQNN,
    )

    nn = DigitParty5x5DeepQNN(params=DQN_5X5_NN_PARAMS, model_folder="")
    state = DigitPartyState(
        board=np.array(
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [9, 8, 7, 6, 5],
                [4, 3, 2, 1, 0],
                [1, 2, 3, 4, 5],
            ],
            dtype=float,
        ),
        player=P1,
        next=(9, 3),
        score=0,
        theoretical_max=0,
        digits=[],
    )
    output = DQNOutput(policy=np.zeros(25), value=0.0)
    fit_inputs: list[np.ndarray] = []
    predict_inputs: list[np.ndarray] = []

    def fake_fit(x, y, batch_size, epochs, shuffle, verbose) -> None:
        del y, batch_size, epochs, shuffle, verbose
        fit_inputs.extend(x)

    def fake_predict(x, verbose):
        del verbose
        predict_inputs.extend(x)
        return np.zeros((1, 25))

    nn.model.fit = fake_fit  # type: ignore[method-assign]
    nn.model.predict = fake_predict  # type: ignore[method-assign]

    nn.train([(state, output)])
    nn.predict([state])

    np.testing.assert_allclose(fit_inputs[0], np.asarray([state.board / 9.0]))
    np.testing.assert_allclose(fit_inputs[1], np.asarray([1.0]))
    np.testing.assert_allclose(fit_inputs[2], np.asarray([3.0 / 9.0]))
    np.testing.assert_allclose(predict_inputs[0], np.asarray([state.board / 9.0]))
    np.testing.assert_allclose(predict_inputs[1], np.asarray([1.0]))
    np.testing.assert_allclose(predict_inputs[2], np.asarray([3.0 / 9.0]))
