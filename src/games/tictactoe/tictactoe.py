from typing import Annotated, List, Literal, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from games.game import (
    INVAL,
    P1,
    P1WIN,
    P2,
    P2WIN,
    VALID,
    Action,
    ActionStatus,
    BasicState,
    Game,
    Player,
    switch_player,
)

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
    def from_action(a: Action) -> Tuple[int, int]:
        r = int(a / 3)
        c = int(a % 3)
        return (r, c)

    @staticmethod
    def apply(state: TicTacToeState, a: Action) -> TicTacToeState:
        r, c = TicTacToe.from_action(a)
        if state.board[r][c] != Empty:
            raise ValueError(
                f"Board already contains tile {state.board[r][c]} at row {r} column {c}"
            )

        board = np.copy(state.board)
        board[r][c] = state.player
        return TicTacToeState(
            board=board,
            player=switch_player(state.player),
        )

    @staticmethod
    def applyIR(ir: TicTacToeIR, a: Action) -> TicTacToeIR:
        s = TicTacToeState(board=np.asarray(ir.board), player=ir.player)
        return TicTacToe.to_immutable(TicTacToe.apply(s, a))

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
        mask = state.board == Empty
        b = np.where(mask, VALID, INVAL)
        return list(b.reshape(TicTacToe.num_actions()))

    @staticmethod
    def symmetries_of(a: NDArray) -> List[NDArray]:
        syms: List[NDArray] = []
        input_shape = a.shape
        b = np.copy(a)
        b = b.reshape((3, 3))
        for i in range(1, 5):
            for mirror in [True, False]:
                s = np.rot90(b, i)
                if mirror:
                    s = np.fliplr(s)
                syms.append(s.reshape(input_shape))
        return syms

    @staticmethod
    def to_immutable(state: TicTacToeState) -> TicTacToeIR:
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
            return 0.1  # TODO: different value for draws?
        else:
            raise RuntimeError(f"Calling reward function when game not ended: {state}")

    def state(self) -> TicTacToeState:
        return TicTacToeState(board=self.board, player=self.player)
