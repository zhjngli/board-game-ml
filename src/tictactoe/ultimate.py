from enum import Enum, auto
from typing import List, NamedTuple, Tuple

from tictactoe.tictactoe import TicTacToe, Tile, TTTBoardState


class FinishedTTTState(Enum):
    Tie = auto()
    XWin = auto()
    OWin = auto()


SimplifiedTTTState = FinishedTTTState | TTTBoardState
UltimateBoardState = Tuple[
    Tuple[SimplifiedTTTState, SimplifiedTTTState, SimplifiedTTTState],
    Tuple[SimplifiedTTTState, SimplifiedTTTState, SimplifiedTTTState],
    Tuple[SimplifiedTTTState, SimplifiedTTTState, SimplifiedTTTState],
]

Section = Tuple[int, int]
Location = Tuple[int, int]


class UltimateState(NamedTuple):
    board: UltimateBoardState
    player: Tile
    active_nonant: Section | None


class UltimateTicTacToe:
    def __init__(self) -> None:
        # TODO: perhaps consider not reusing TicTacToe class and optimizing it a little
        self.board: List[List[TicTacToe]] = [
            [TicTacToe() for _ in range(3)] for _ in range(3)
        ]
        self.player: Tile = Tile.X
        self.active_nonant: Section | None = None

    @staticmethod
    def _is_win(t: Tile, board: List[List[TicTacToe]]) -> bool:
        # fmt: off
        return (
            # horizontal
            (board[0][0].win(t) and board[0][1].win(t) and board[0][2].win(t)) or  # noqa: W504
            (board[1][0].win(t) and board[1][1].win(t) and board[1][2].win(t)) or  # noqa: W504
            (board[2][0].win(t) and board[2][1].win(t) and board[2][2].win(t)) or  # noqa: W504
            # vertical
            (board[0][0].win(t) and board[1][0].win(t) and board[2][0].win(t)) or  # noqa: W504
            (board[0][1].win(t) and board[1][1].win(t) and board[2][1].win(t)) or  # noqa: W504
            (board[0][2].win(t) and board[1][2].win(t) and board[2][2].win(t)) or  # noqa: W504
            # diag
            (board[0][0].win(t) and board[1][1].win(t) and board[2][2].win(t)) or  # noqa: W504
            (board[2][0].win(t) and board[1][1].win(t) and board[0][2].win(t))
        )
        # fmt: on

    def win(self, t: Tile) -> bool:
        return self._is_win(t, self.board)

    @staticmethod
    def _is_board_filled(board: List[List[TicTacToe]]) -> bool:
        return all(all(t.finished() for t in row) for row in board)

    def board_filled(self) -> bool:
        return self._is_board_filled(self.board)

    @staticmethod
    def _is_finished(board: List[List[TicTacToe]]) -> bool:
        xWin = UltimateTicTacToe._is_win(Tile.X, board)
        oWin = UltimateTicTacToe._is_win(Tile.O, board)

        return xWin or oWin or UltimateTicTacToe._is_board_filled(board)

    def finished(self) -> bool:
        return self._is_finished(self.board)

    @staticmethod
    def get_board_state(board: List[List[TicTacToe]]) -> UltimateBoardState:
        def get_simplified_state(t: TicTacToe) -> SimplifiedTTTState:
            if t.win(Tile.X):
                return FinishedTTTState.XWin
            elif t.win(Tile.O):
                return FinishedTTTState.OWin
            elif t.board_filled():
                return FinishedTTTState.Tie
            else:
                return t.get_state().board

        # really ugly way to get correct typing :x
        # assumes that the board is 3 rows and 3 columns
        return (
            (
                get_simplified_state(board[0][0]),
                get_simplified_state(board[0][1]),
                get_simplified_state(board[0][2]),
            ),
            (
                get_simplified_state(board[1][0]),
                get_simplified_state(board[1][1]),
                get_simplified_state(board[1][2]),
            ),
            (
                get_simplified_state(board[2][0]),
                get_simplified_state(board[2][1]),
                get_simplified_state(board[2][2]),
            ),
        )

    def get_state(self) -> UltimateState:
        return UltimateState(
            board=UltimateTicTacToe.get_board_state(self.board),
            player=self.player,
            active_nonant=self.active_nonant,
        )
