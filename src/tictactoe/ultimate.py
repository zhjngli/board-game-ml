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
    def _in_range(sl: Section | Location) -> bool:
        (r, c) = sl
        return r < 3 and r >= 0 and c < 3 and c >= 0

    @staticmethod
    def switch_player(player: Tile) -> Tile:
        return Tile.O if player == Tile.X else Tile.X

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

    def play(self, sec: Section, loc: Location) -> None:
        (R, C) = sec
        if not UltimateTicTacToe._in_range(sec):
            raise ValueError(f"Row {R} or column {C} outside of the Ultimate board")

        if self.active_nonant and self.active_nonant != sec:
            (activeR, activeC) = self.active_nonant
            raise ValueError(
                f"Last play was in section {activeR, activeC}, you must play there as well"
            )

        if self.board[R][C].finished():
            raise ValueError(f"Section {R, C} is finished, you cannot play there.")

        (r, c) = loc
        try:
            self.board[R][C]._play(self.player, r, c)
        except ValueError as e:
            raise ValueError(f"Error at trying to play {r, c} at {R, C}: {e}")

        self.player = UltimateTicTacToe.switch_player(self.player)

        if self.board[r][c].finished():
            self.active_nonant = None
        else:
            self.active_nonant = loc

    @staticmethod
    def simplified_ttt_board(t: TicTacToe) -> List[str]:
        # should be the same structure (length of array, and length of strings in array) as TicTacToe.show()
        if t.win(Tile.X):
            return [
                "x     X",
                " X   X ",
                "  X X  ",
                "  X X  ",
                " X   X ",
                "X     X",
            ]
        elif t.win(Tile.O):
            return [
                "  OOO  ",
                " O   O ",
                "O     O",
                "O     O",
                " O   O ",
                "  OOO  ",
            ]
        elif t.finished():
            return [
                "- - - -",
                " - - - ",
                "- - - -",
                " - - - ",
                "- - - -",
                " - - - ",
            ]
        else:
            return t._board_array()

    def show(self) -> str:
        spacing_constant = "            |         |"
        final = [
            #    ---     x   ---   x   ---   x   |
            "        0         1         2",
            spacing_constant,
        ]

        for r in range(3):
            board_arrays = []
            for c in range(3):
                board_arrays.append(
                    UltimateTicTacToe.simplified_ttt_board(self.board[r][c])
                )

            for i in range(len(board_arrays[0])):
                prefix = "   "
                if i == 3:  # middle row of the mini board
                    prefix = f" {r} "
                final.append(
                    f"{prefix} {board_arrays[0][i]} | {board_arrays[1][i]} | {board_arrays[2][i]}"
                )

            if r < 2:  # separate board in between the rows
                final.append(spacing_constant)
                final.append("   ---------+---------+---------")
                final.append(spacing_constant)

        final.append(spacing_constant)
        final.append("\n")
        return "\n".join(final)


def human_game() -> None:
    g = UltimateTicTacToe()
    p = 0
    t = [Tile.X, Tile.O]

    while not g.finished():
        print(f"\n{g.show()}\n")

        if g.active_nonant is None:
            print(
                f"Last active section {g.active_nonant} is finished, or it's a new game and you have free choice."
            )
            sec = input(f"player {p + 1} ({t[p]}) choose a section: ").strip()
            try:
                RC = sec.split(",")[:2]
                R = int(RC[0])
                C = int(RC[1])
            except (ValueError, IndexError):
                print(
                    "can't read your section input. input as comma separated, e.g. 1,1"
                )
                continue
        else:
            # gets past mypy error. if this happens, error will happen downstream when trying to play at (-1, -1)
            (R, C) = g.active_nonant if g.active_nonant is not None else (-1, -1)
            print(
                f"The last active section is {R, C}, you are forced to play in that section."
            )

        loc = input(
            f"player {p + 1} ({t[p]}) choose a location in section {R, C}: "
        ).strip()
        try:
            rc = loc.split(",")[:2]
            r = int(rc[0])
            c = int(rc[1])
        except (ValueError, IndexError):
            print("can't read your location input. input as comma separated, e.g. 1,1")
            continue

        try:
            g.play((R, C), (r, c))
        except ValueError as e:
            print(f"\n\n{str(e)}")
            continue

        p += 1
        p %= 2

    print(f"{g.show()}\n")
    print("game over!")
