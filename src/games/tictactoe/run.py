from typing import List, Tuple

from typing_extensions import override

from games.game import P1, P2
from games.tictactoe.tictactoe import Empty, TicTacToe, TicTacToeIR
from learners.monte_carlo import MonteCarloLearner
from learners.q import SimpleQLearner
from learners.trainer import Trainer


def human_game() -> None:
    g = TicTacToe()
    p = 0
    play = [g.play1, g.play2]
    while not g.is_finished():
        print()
        print(g.show())
        print()

        coord = input(f"player {p + 1} choose a spot: ").strip()
        try:
            rc = coord.split(",")[:2]
            r = int(rc[0])
            c = int(rc[1])
        except (ValueError, IndexError):
            print("can't read your coordinate input")
            continue

        try:
            play[p](r, c)
        except ValueError as e:
            print(str(e))
            continue

        p += 1
        p %= 2

    print(g.show())
    print("game over!")


class TicTacToeMonteCarloTrainer(TicTacToe, Trainer):
    def __init__(self, p1: MonteCarloLearner, p2: MonteCarloLearner) -> None:
        super().__init__()
        self.p1 = p1
        self.p2 = p2

    @override
    def train(self, episodes=1000) -> None:
        super().train(episodes)

        self.p1.save_policy()
        self.p2.save_policy()

    def train_once(self) -> None:
        while not self.is_finished():
            r, c = self.p1.choose_action(self.to_immutable(self.state()))
            self.play1(r, c)
            self.p1.add_state(self.to_immutable(self.state()))

            if self.is_finished():
                break

            r, c = self.p2.choose_action(self.to_immutable(self.state()))
            self.play2(r, c)
            self.p2.add_state(self.to_immutable(self.state()))

            if self.is_finished():
                break

        self.give_rewards()
        self.reset()
        self.p1.reset_states()
        self.p2.reset_states()

    def give_rewards(self) -> None:
        # TODO: how might changing these rewards affect behavior?
        if self.win(P1):
            self.p1.propagate_reward(1)
            self.p2.propagate_reward(0)
        elif self.win(P2):
            self.p1.propagate_reward(0)
            self.p2.propagate_reward(1)
        elif self.board_filled():
            self.p1.propagate_reward(0.1)
            self.p2.propagate_reward(0.5)
        else:
            raise RuntimeError(
                "giving rewards when game's not over. something's wrong!"
            )


class TicTacToeMonteCarloLearner(MonteCarloLearner[TicTacToeIR, Tuple[int, int]]):
    def get_actions_from_state(self, state: TicTacToeIR) -> List[Tuple[int, int]]:
        return [
            (i, j) for i in range(3) for j in range(3) if state.board[i][j] == Empty
        ]

    def apply(self, state: TicTacToeIR, action: Tuple[int, int]) -> TicTacToeIR:
        r, c = action
        a = r * 3 + c
        return TicTacToe.applyIR(state, a)


class TicTacToeQTrainer(TicTacToe, Trainer):
    def __init__(self, p1: SimpleQLearner, p2: SimpleQLearner) -> None:
        super().__init__()
        self.p1 = p1
        self.p2 = p2

    @override
    def train(self, episodes=10000) -> None:
        super().train(episodes)

        self.p1.save_policy()
        self.p2.save_policy()

    # TODO: this training is pretty dumb, doesn't work well
    def train_once(self) -> None:
        while not self.is_finished():
            ir = self.to_immutable(self.state())
            action = self.p1.choose_action(ir)
            r, c = action
            self.play1(r, c)

            if self.is_finished():
                break

            ir = self.to_immutable(self.state())
            action = self.p2.choose_action(ir)
            r, c = action
            self.play2(r, c)

        if self.win(P1):
            self.p1.update_q_value(ir, action, 1, self.to_immutable(self.state()))
            self.p2.update_q_value(ir, action, -1, self.to_immutable(self.state()))
        elif self.win(P2):
            self.p1.update_q_value(ir, action, -1, self.to_immutable(self.state()))
            self.p2.update_q_value(ir, action, 1, self.to_immutable(self.state()))
        elif self.board_filled():
            self.p1.update_q_value(ir, action, -0.1, self.to_immutable(self.state()))
            self.p2.update_q_value(ir, action, 0, self.to_immutable(self.state()))
        else:
            raise Exception("giving rewards when game's not over. something's wrong!")
        self.reset()


class TicTacToeQLearner(SimpleQLearner[TicTacToeIR, Tuple[int, int]]):
    def default_action_q_values(self) -> dict[Tuple[int, int], float]:
        actions = {}
        for r in range(3):
            for c in range(3):
                actions[(r, c)] = 0.0
        return actions

    def get_actions_from_state(self, state: TicTacToeIR) -> List[Tuple[int, int]]:
        return [
            (i, j) for i in range(3) for j in range(3) if state.board[i][j] == Empty
        ]


def _computer_play(g: TicTacToe, p: TicTacToeMonteCarloLearner | TicTacToeQLearner):
    print(f"\n{g.show()}\n")
    r, c = p.choose_action(g.to_immutable(g.state()), exploit=True)
    g.play(r, c)
    print(f"\ncomputer plays at ({r}, {c})!")


def _human_play(g: TicTacToe):
    print(f"\n{g.show()}\n")

    coord = input("please choose a spot to play: ").strip()
    try:
        rc = coord.split(",")[:2]
        r = int(rc[0])
        c = int(rc[1])
    except (ValueError, IndexError):
        raise ValueError("can't read your coordinate input, try again")

    try:
        g.play(r, c)
    except ValueError as e:
        raise e


def _trained_game(  # noqa: C901
    g: TicTacToe,
    computer1: TicTacToeMonteCarloLearner | TicTacToeQLearner,
    computer2: TicTacToeMonteCarloLearner | TicTacToeQLearner,
):
    player = input(
        "play as player 1 or 2? or choose 0 to spectate computers play. "
    ).strip()
    try:
        p = int(player)
    except ValueError:
        print("unrecognized input, defaulting to player 1!")
        p = 1

    if p == 1:
        pass
    elif p == 2 or p == 0:
        _computer_play(g, computer1)
    else:
        print("unrecognized input, defaulting to player 1!")

    while not g.is_finished():
        if p == 0:
            _computer_play(g, computer2)
        else:
            try:
                _human_play(g)
            except ValueError as e:
                print(str(e))
                continue

        if g.is_finished():
            break

        computer = computer1 if p == 0 or p == 2 else computer2
        _computer_play(g, computer)

    print(f"\n{g.show()}\n")
    print("game over!")
    if g.win(P1):
        print("X won!")
    elif g.win(P2):
        print("O won!")
    else:
        print("tie!")


def _many_games(
    g: TicTacToe,
    computer1: TicTacToeMonteCarloLearner | TicTacToeQLearner,
    computer2: TicTacToeMonteCarloLearner | TicTacToeQLearner,
    games: int,
):
    x_wins = 0
    o_wins = 0
    ties = 0
    for _ in range(games):
        while not g.is_finished():
            r, c = computer1.choose_action(g.to_immutable(g.state()), exploit=True)
            g.play(r, c)
            if g.is_finished():
                break
            r, c = computer2.choose_action(g.to_immutable(g.state()), exploit=True)
            g.play(r, c)

        if g.win(P1):
            x_wins += 1
        elif g.win(P2):
            o_wins += 1
        else:
            ties += 1
        g.reset()

    print(f"played {games} games")
    print(f"x won {x_wins} times")
    print(f"o won {o_wins} times")
    print(f"{ties} ties")


MCP1_POLICY = "src/games/tictactoe/mcp1.pkl"
MCP2_POLICY = "src/games/tictactoe/mcp2.pkl"


def monte_carlo_trained_game(training_episodes=0):
    computer1 = TicTacToeMonteCarloLearner(policy_file=MCP1_POLICY)
    computer2 = TicTacToeMonteCarloLearner(policy_file=MCP2_POLICY)
    g = TicTacToeMonteCarloTrainer(p1=computer1, p2=computer2)
    g.train(episodes=training_episodes)

    _trained_game(g, computer1, computer2)


def monte_carlo_many_games(games=10000):
    computer1 = TicTacToeMonteCarloLearner(policy_file=MCP1_POLICY)
    computer2 = TicTacToeMonteCarloLearner(policy_file=MCP2_POLICY)
    g = TicTacToeMonteCarloTrainer(p1=computer1, p2=computer2)

    _many_games(g, computer1, computer2, games)


QP1_POLICY = "src/games/tictactoe/qp1.pkl"
QP2_POLICY = "src/games/tictactoe/qp2.pkl"


def q_trained_game(training_episodes=0):
    computer1 = TicTacToeQLearner(q_pickle=QP1_POLICY)
    computer2 = TicTacToeQLearner(q_pickle=QP2_POLICY)
    g = TicTacToeQTrainer(p1=computer1, p2=computer2)
    g.train(episodes=training_episodes)

    _trained_game(g, computer1, computer2)


def q_many_games(games=10000):
    computer1 = TicTacToeQLearner(q_pickle=QP1_POLICY)
    computer2 = TicTacToeQLearner(q_pickle=QP2_POLICY)
    g = TicTacToeQTrainer(p1=computer1, p2=computer2)

    _many_games(g, computer1, computer2, games)
