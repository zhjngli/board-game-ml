from enum import Enum, auto
from typing import List, NamedTuple, Tuple, override

from learners.monte_carlo import MonteCarloLearner
from learners.q import SimpleQLearner
from learners.trainer import Trainer


class Tile(Enum):
    N = auto()
    X = auto()
    O = auto()  # noqa: E741

    def __str__(self) -> str:
        if self.name == "N":
            return " "
        return f"{self.name}"


TTTBoardState = Tuple[
    Tuple[Tile, Tile, Tile], Tuple[Tile, Tile, Tile], Tuple[Tile, Tile, Tile]
]


class TicTacToeState(NamedTuple):
    board: TTTBoardState
    player: Tile


class TicTacToe:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.board: List[List[Tile]] = [[Tile.N for _ in range(3)] for _ in range(3)]
        self.player = Tile.X

    def _in_range(self, r: int, c: int) -> bool:
        return r < 3 and r >= 0 and c < 3 and c >= 0

    def _play(self, t: Tile, r: int, c: int) -> None:
        if not self._in_range(r, c):
            raise ValueError(f"Row {r} or column {c} outside of the TicTacToe board")

        if self.board[r][c] != Tile.N:
            raise ValueError(
                f"Board already contains tile {self.board[r][c]} at row {r} column {c}"
            )

        self.board[r][c] = t
        self.player = Tile.O if self.player == Tile.X else Tile.X

    @staticmethod
    def get_board_state(board: List[List[Tile]]) -> TTTBoardState:
        # really ugly way to get correct typing :x
        # assumes that the board is 3 rows and 3 columns
        return (
            (board[0][0], board[0][1], board[0][2]),
            (board[1][0], board[1][1], board[1][2]),
            (board[2][0], board[2][1], board[2][2]),
        )

    @staticmethod
    def apply(state: TicTacToeState, r: int, c: int) -> TicTacToeState:
        s = TicTacToeState(board=state.board, player=state.player)

        if s.board[r][c] != Tile.N:
            raise ValueError(
                f"Board already contains tile {s.board[r][c]} at row {r} column {c}"
            )

        board = list(list(row) for row in s.board)
        board[r][c] = s.player
        new_s = TicTacToeState(
            board=TicTacToe.get_board_state(board),
            player=Tile.O if s.player == Tile.X else Tile.X,
        )

        return new_s

    @staticmethod
    def _is_win(t: Tile, board: List[List[Tile]]) -> bool:
        # fmt: off
        return (
            # horizontal
            (board[0][0] == t and board[0][1] == t and board[0][2] == t) or  # noqa: W504
            (board[1][0] == t and board[1][1] == t and board[1][2] == t) or  # noqa: W504
            (board[2][0] == t and board[2][1] == t and board[2][2] == t) or  # noqa: W504
            # vertical
            (board[0][0] == t and board[1][0] == t and board[2][0] == t) or  # noqa: W504
            (board[0][1] == t and board[1][1] == t and board[2][1] == t) or  # noqa: W504
            (board[0][2] == t and board[1][2] == t and board[2][2] == t) or  # noqa: W504
            # diag
            (board[0][0] == t and board[1][1] == t and board[2][2] == t) or  # noqa: W504
            (board[2][0] == t and board[1][1] == t and board[0][2] == t)
        )
        # fmt: on

    def win(self, t: Tile) -> bool:
        return self._is_win(t, self.board)

    @staticmethod
    def _is_board_filled(board: List[List[Tile]]) -> bool:
        return all(all(t != Tile.N for t in row) for row in board)

    def board_filled(self) -> bool:
        return self._is_board_filled(self.board)

    @staticmethod
    def _is_finished(board: List[List[Tile]]) -> bool:
        xWin = TicTacToe._is_win(Tile.X, board)
        oWin = TicTacToe._is_win(Tile.O, board)

        return xWin or oWin or TicTacToe._is_board_filled(board)

    def finished(self) -> bool:
        return self._is_finished(self.board)

    def get_state(self) -> TicTacToeState:
        return TicTacToeState(TicTacToe.get_board_state(self.board), self.player)

    def play(self, r: int, c: int) -> None:
        self._play(self.player, r, c)

    def play1(self, r: int, c: int) -> None:
        self._play(Tile.X, r, c)

    def play2(self, r: int, c: int) -> None:
        self._play(Tile.O, r, c)

    def _board_array(self) -> List[str]:
        return [
            "  0 1 2",
            f"0 {self.board[0][0]}|{self.board[0][1]}|{self.board[0][2]}",
            "  -+-+-",
            f"1 {self.board[1][0]}|{self.board[1][1]}|{self.board[1][2]}",
            "  -+-+-",
            f"2 {self.board[2][0]}|{self.board[2][1]}|{self.board[2][2]}",
        ]

    def show(self) -> str:
        return "\n".join(self._board_array())


def human_game() -> None:
    g = TicTacToe()
    p = 0
    play = [g.play1, g.play2]
    while not g.finished():
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
        while not self.finished():
            r, c = self.p1.choose_action(self.get_state())
            self.play1(r, c)
            self.p1.add_state(self.get_state())

            if self.finished():
                break

            r, c = self.p2.choose_action(self.get_state())
            self.play2(r, c)
            self.p2.add_state(self.get_state())

            if self.finished():
                break

        self.give_rewards()
        self.reset()
        self.p1.reset_states()
        self.p2.reset_states()

    def give_rewards(self) -> None:
        # TODO: how might changing these rewards affect behavior?
        if self.win(Tile.X):
            self.p1.propagate_reward(1)
            self.p2.propagate_reward(0)
        elif self.win(Tile.O):
            self.p1.propagate_reward(0)
            self.p2.propagate_reward(1)
        elif self.board_filled():
            self.p1.propagate_reward(0.1)
            self.p2.propagate_reward(0.5)
        else:
            raise Exception("giving rewards when game's not over. something's wrong!")


class TicTacToeMonteCarloLearner(MonteCarloLearner[TicTacToeState, Tuple[int, int]]):
    def get_actions_from_state(self, state: TicTacToeState) -> List[Tuple[int, int]]:
        return [
            (i, j) for i in range(3) for j in range(3) if state.board[i][j] == Tile.N
        ]

    def apply(self, state: TicTacToeState, action: Tuple[int, int]) -> TicTacToeState:
        r, c = action
        return TicTacToe.apply(state, r, c)


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
        while not self.finished():
            state = self.get_state()
            action = self.p1.choose_action(state)
            r, c = action
            self.play1(r, c)

            if self.finished():
                break

            state = self.get_state()
            action = self.p2.choose_action(state)
            r, c = action
            self.play2(r, c)

        if self.win(Tile.X):
            self.p1.update_q_value(state, action, 1, self.get_state())
            self.p2.update_q_value(state, action, -1, self.get_state())
        elif self.win(Tile.O):
            self.p1.update_q_value(state, action, -1, self.get_state())
            self.p2.update_q_value(state, action, 1, self.get_state())
        elif self.board_filled():
            self.p1.update_q_value(state, action, -0.1, self.get_state())
            self.p2.update_q_value(state, action, 0, self.get_state())
        else:
            raise Exception("giving rewards when game's not over. something's wrong!")
        self.reset()


class TicTacToeQLearner(SimpleQLearner[TicTacToeState, Tuple[int, int]]):
    def default_action_q_values(self) -> dict[Tuple[int, int], float]:
        actions = {}
        for r in range(3):
            for c in range(3):
                actions[(r, c)] = 0.0
        return actions

    def get_actions_from_state(self, state: TicTacToeState) -> List[Tuple[int, int]]:
        return [
            (i, j) for i in range(3) for j in range(3) if state.board[i][j] == Tile.N
        ]


def _computer_play(g: TicTacToe, p: TicTacToeMonteCarloLearner | TicTacToeQLearner):
    print(f"\n{g.show()}\n")
    r, c = p.choose_action(g.get_state(), exploit=True)
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

    while not g.finished():
        if p == 0:
            _computer_play(g, computer2)
        else:
            try:
                _human_play(g)
            except ValueError as e:
                print(str(e))
                continue

        if g.finished():
            break

        computer = computer1 if p == 0 or p == 2 else computer2
        _computer_play(g, computer)

    print(f"\n{g.show()}\n")
    print("game over!")
    if g.win(Tile.X):
        print("X won!")
    elif g.win(Tile.O):
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
        while not g.finished():
            r, c = computer1.choose_action(g.get_state(), exploit=True)
            g.play(r, c)
            if g.finished():
                break
            r, c = computer2.choose_action(g.get_state(), exploit=True)
            g.play(r, c)

        if g.win(Tile.X):
            x_wins += 1
        elif g.win(Tile.O):
            o_wins += 1
        else:
            ties += 1
        g.reset()

    print(f"played {games} games")
    print(f"x won {x_wins} times")
    print(f"o won {o_wins} times")
    print(f"{ties} ties")


MCP1_POLICY = "src/tictactoe/mcp1.pkl"
MCP2_POLICY = "src/tictactoe/mcp2.pkl"


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


QP1_POLICY = "src/tictactoe/qp1.pkl"
QP2_POLICY = "src/tictactoe/qp2.pkl"


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
