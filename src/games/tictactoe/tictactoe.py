from typing import Annotated, List, Literal, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from games.game import (
    P1,
    P1WIN,
    P2,
    P2WIN,
    Action,
    ActionStatus,
    BasicState,
    Game,
    Player,
    switch_player,
)
from learners.monte_carlo import MonteCarloLearner
from learners.q import SimpleQLearner
from learners.trainer import Trainer

Tile = Literal[1, 0, -1]
XTile: Tile = 1
OTile: Tile = -1
Empty: Tile = 0


def tile_char(t: Tile) -> str:
    if t == Empty:
        return " "
    elif t == XTile:
        return "X"
    elif t == OTile:
        return "O"
    else:
        raise Exception("Typing Tile literals should have caught this case")


TTTBoard = Annotated[NDArray, Literal[3, 3]]  # TODO: force NDArray to hold Tile types?
TTTBoardIR = Tuple[
    Tuple[Tile, Tile, Tile], Tuple[Tile, Tile, Tile], Tuple[Tile, Tile, Tile]
]


class TicTacToeIR(NamedTuple):
    board: TTTBoardIR
    player: Player


class TicTacToeState(BasicState):
    def __init__(self, board: TTTBoard, player: Player) -> None:
        super().__init__(board, player)


# TODO: mixing a lot of state/IR nomenclature due to refactoring. clean that up
# TODO: would be cleaner to have a way to get IR from the game itself instead of passing the state to the static method
# TODO: CLEANUP, play and apply are kinda redundant?
class TicTacToe(Game[TicTacToeState, TicTacToeIR]):
    def __init__(self, state: Optional[TicTacToeIR] = None) -> None:
        if state is None:
            self.reset()
        else:
            self.board: TTTBoard = np.asarray(list(list(row) for row in state.board))
            self.player: Player = state.player

    def reset(self) -> None:
        self.board = np.zeros((3, 3))
        self.player = P1

    def _in_range(self, r: int, c: int) -> bool:
        return r < 3 and r >= 0 and c < 3 and c >= 0

    def _play(self, t: Tile, r: int, c: int) -> None:
        if not self._in_range(r, c):
            raise ValueError(f"Row {r} or column {c} outside of the TicTacToe board")

        if self.board[r][c] != Empty:
            raise ValueError(
                f"Board already contains tile {self.board[r][c]} at row {r} column {c}"
            )

        if self.player != t:
            raise ValueError(f"Trying to play tile {t} as player {self.player}")

        self.board[r][c] = t
        self.player = switch_player(self.player)

    @staticmethod
    def get_board_rep(board: TTTBoard) -> TTTBoardIR:
        # really ugly way to get correct typing :x
        # assumes that the board is 3 rows and 3 columns
        return (
            (board[0][0], board[0][1], board[0][2]),
            (board[1][0], board[1][1], board[1][2]),
            (board[2][0], board[2][1], board[2][2]),
        )

    @staticmethod
    def apply(state: TicTacToeState, a: Action) -> TicTacToeState:
        r = int(a / 3)
        c = int(a % 3)
        if state.board[r][c] != Empty:
            raise ValueError(
                f"Board already contains tile {state.board[r][c]} at row {r} column {c}"
            )

        # TODO: does this need a copy?
        board = state.board
        board[r][c] = state.player
        return TicTacToeState(
            board=board,
            player=switch_player(state.player),
        )

    @staticmethod
    def applyIR(ir: TicTacToeIR, a: Action) -> TicTacToeIR:
        s = TicTacToeState(board=np.asarray(ir.board), player=ir.player)
        return TicTacToe.immutable_of(TicTacToe.apply(s, a))

    @staticmethod
    def _is_win(p: Player, board: TTTBoard) -> bool:
        # fmt: off
        return (
            # horizontal
            (board[0][0] == p and board[0][1] == p and board[0][2] == p) or  # noqa: W504
            (board[1][0] == p and board[1][1] == p and board[1][2] == p) or  # noqa: W504
            (board[2][0] == p and board[2][1] == p and board[2][2] == p) or  # noqa: W504
            # vertical
            (board[0][0] == p and board[1][0] == p and board[2][0] == p) or  # noqa: W504
            (board[0][1] == p and board[1][1] == p and board[2][1] == p) or  # noqa: W504
            (board[0][2] == p and board[1][2] == p and board[2][2] == p) or  # noqa: W504
            # diag
            (board[0][0] == p and board[1][1] == p and board[2][2] == p) or  # noqa: W504
            (board[2][0] == p and board[1][1] == p and board[0][2] == p)
        )
        # fmt: on

    def win(self, p: Player) -> bool:
        return TicTacToe._is_win(p, self.board)

    @staticmethod
    def _is_board_filled(board: TTTBoard) -> bool:
        return bool(np.all(board != Empty))

    def board_filled(self) -> bool:
        return self._is_board_filled(self.board)

    @staticmethod
    def check_finished(state: TicTacToeState) -> bool:
        board = state.board
        xWin = TicTacToe._is_win(P1, board)
        oWin = TicTacToe._is_win(P2, board)

        return xWin or oWin or TicTacToe._is_board_filled(board)

    def is_finished(self) -> bool:
        return TicTacToe.check_finished(self.state())

    def play(self, r: int, c: int) -> None:
        self._play(self.player, r, c)

    def play1(self, r: int, c: int) -> None:
        self._play(XTile, r, c)

    def play2(self, r: int, c: int) -> None:
        self._play(OTile, r, c)

    @staticmethod
    def board_array(t: TTTBoard) -> List[str]:
        return [
            "  0 1 2",
            f"0 {tile_char(t[0][0])}|{tile_char(t[0][1])}|{tile_char(t[0][2])}",
            "  -+-+-",
            f"1 {tile_char(t[1][0])}|{tile_char(t[1][1])}|{tile_char(t[1][2])}",
            "  -+-+-",
            f"2 {tile_char(t[2][0])}|{tile_char(t[2][1])}|{tile_char(t[2][2])}",
        ]

    def show(self) -> str:
        return "\n".join(TicTacToe.board_array(self.board))

    @staticmethod
    def num_actions() -> int:
        return 9  # one for each area of the board

    @staticmethod
    def actions(state: TicTacToeState) -> List[ActionStatus]:
        b = np.copy(state.board)
        b[b == Empty] = 1
        b[b != Empty] = 0
        return list(b.reshape(TicTacToe.num_actions()))

    @staticmethod
    def symmetries_of(a: NDArray) -> List[NDArray]:
        syms: List[NDArray] = []
        b = np.copy(a)
        for i in range(1, 5):
            for mirror in [True, False]:
                s = np.rot90(b, i)
                if mirror:
                    s = np.fliplr(s)
                syms += s
        return syms

    @staticmethod
    def immutable_of(state: TicTacToeState) -> TicTacToeIR:
        return TicTacToeIR(
            board=TicTacToe.get_board_rep(state.board), player=state.player
        )

    @staticmethod
    def orient_state(state: TicTacToeState) -> TicTacToeState:
        return TicTacToeState(board=state.board * state.player, player=P1)

    @staticmethod
    def calculate_reward(state: TicTacToeState) -> float:
        if TicTacToe._is_win(P1, state.board):
            return P1WIN
        elif TicTacToe._is_win(P2, state.board):
            return P2WIN
        elif TicTacToe._is_board_filled(state.board):
            return 0  # TODO: different value for draws?
        else:
            raise RuntimeError(f"Calling reward function when game not ended: {state}")

    def state(self) -> TicTacToeState:
        return TicTacToeState(board=self.board, player=self.player)


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
            r, c = self.p1.choose_action(self.immutable_of(self.state()))
            self.play1(r, c)
            self.p1.add_state(self.immutable_of(self.state()))

            if self.is_finished():
                break

            r, c = self.p2.choose_action(self.immutable_of(self.state()))
            self.play2(r, c)
            self.p2.add_state(self.immutable_of(self.state()))

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


# class TicTacToeQTrainer(TicTacToe, Trainer):
#     def __init__(self, p1: SimpleQLearner, p2: SimpleQLearner) -> None:
#         super().__init__()
#         self.p1 = p1
#         self.p2 = p2

#     @override
#     def train(self, episodes=10000) -> None:
#         super().train(episodes)

#         self.p1.save_policy()
#         self.p2.save_policy()

#     # TODO: this training is pretty dumb, doesn't work well
#     def train_once(self) -> None:
#         while not self.is_finished():
#             ir = self.immutable_of(self.state())
#             action = self.p1.choose_action(ir)
#             r, c = action
#             self.play1(r, c)

#             if self.is_finished():
#                 break

#             ir = self.immutable_of(self.state())
#             action = self.p2.choose_action(ir)
#             r, c = action
#             self.play2(r, c)

#         if self.win(P1):
#             self.p1.update_q_value(ir, action, 1, self.immutable_of(self.state()))
#             self.p2.update_q_value(ir, action, -1, self.immutable_of(self.state()))
#         elif self.win(P2):
#             self.p1.update_q_value(ir, action, -1, self.immutable_of(self.state()))
#             self.p2.update_q_value(ir, action, 1, self.immutable_of(self.state()))
#         elif self.board_filled():
#             self.p1.update_q_value(ir, action, -0.1, self.immutable_of(self.state()))
#             self.p2.update_q_value(ir, action, 0, self.immutable_of(self.state()))
#         else:
#             raise Exception("giving rewards when game's not over. something's wrong!")
#         self.reset()


# class TicTacToeQLearner(SimpleQLearner[TicTacToeIR, Tuple[int, int]]):
#     def default_action_q_values(self) -> dict[Tuple[int, int], float]:
#         actions = {}
#         for r in range(3):
#             for c in range(3):
#                 actions[(r, c)] = 0.0
#         return actions

#     def get_actions_from_state(self, state: TicTacToeIR) -> List[Tuple[int, int]]:
#         return [
#             (i, j) for i in range(3) for j in range(3) if state.board[i][j] == Empty
#         ]


def _computer_play(g: TicTacToe, p: TicTacToeMonteCarloLearner):
    print(f"\n{g.show()}\n")
    r, c = p.choose_action(g.immutable_of(g.state()), exploit=True)
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
    computer1: TicTacToeMonteCarloLearner,
    computer2: TicTacToeMonteCarloLearner,
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
    computer1: TicTacToeMonteCarloLearner,
    computer2: TicTacToeMonteCarloLearner,
    games: int,
):
    x_wins = 0
    o_wins = 0
    ties = 0
    for _ in range(games):
        while not g.is_finished():
            r, c = computer1.choose_action(g.immutable_of(g.state()), exploit=True)
            g.play(r, c)
            if g.is_finished():
                break
            r, c = computer2.choose_action(g.immutable_of(g.state()), exploit=True)
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


# QP1_POLICY = "src/games/tictactoe/qp1.pkl"
# QP2_POLICY = "src/games/tictactoe/qp2.pkl"


# def q_trained_game(training_episodes=0):
#     computer1 = TicTacToeQLearner(q_pickle=QP1_POLICY)
#     computer2 = TicTacToeQLearner(q_pickle=QP2_POLICY)
#     g = TicTacToeQTrainer(p1=computer1, p2=computer2)
#     g.train(episodes=training_episodes)

#     _trained_game(g, computer1, computer2)


# def q_many_games(games=10000):
#     computer1 = TicTacToeQLearner(q_pickle=QP1_POLICY)
#     computer2 = TicTacToeQLearner(q_pickle=QP2_POLICY)
#     g = TicTacToeQTrainer(p1=computer1, p2=computer2)

#     _many_games(g, computer1, computer2, games)
