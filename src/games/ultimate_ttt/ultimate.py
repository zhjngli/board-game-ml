from enum import Enum, auto
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
from games.tictactoe.tictactoe import (
    Empty,
    OTile,
    TicTacToe,
    TicTacToeState,
    TTTBoard,
    TTTBoardIR,
    XTile,
    tile_char,
)


class FinishedTTTState(Enum):
    Tie = auto()
    XWin = auto()
    OWin = auto()


UltimateBoard = Annotated[
    NDArray, Literal[3, 3, 3, 3]
]  # TODO: force NDArray to hold Tile types?
SimplifiedTTTIR = FinishedTTTState | TTTBoardIR
UltimateBoardIR = Tuple[
    Tuple[SimplifiedTTTIR, SimplifiedTTTIR, SimplifiedTTTIR],
    Tuple[SimplifiedTTTIR, SimplifiedTTTIR, SimplifiedTTTIR],
    Tuple[SimplifiedTTTIR, SimplifiedTTTIR, SimplifiedTTTIR],
]

Section = Tuple[int, int]
Location = Tuple[int, int]


class UltimateState(BasicState):
    def __init__(
        self, board: UltimateBoard, player: Player, active_nonant: Optional[Section]
    ) -> None:
        super().__init__(board, player)
        self.active_nonant = active_nonant


class UltimateIR(NamedTuple):
    board: UltimateBoardIR
    player: Player
    active_nonant: Optional[Section]


def ir_to_state(ir: UltimateIR) -> UltimateState:
    board = np.zeros((3, 3, 3, 3))
    for R in range(3):
        for C in range(3):
            if ir.board[R][C] == FinishedTTTState.XWin:
                board[R][C] = P1
            elif ir.board[R][C] == FinishedTTTState.OWin:
                board[R][C] = P2
            elif ir.board[R][C] == FinishedTTTState.Tie:
                board[R][C] = np.asarray(
                    [
                        [XTile, XTile, OTile],
                        [OTile, XTile, XTile],
                        [XTile, OTile, OTile],
                    ]
                )
            else:
                board[R][C] = np.asarray(ir.board[R][C])
    return UltimateState(board=board, player=ir.player, active_nonant=ir.active_nonant)


class UltimateTicTacToe(Game[UltimateState, UltimateIR]):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.board: UltimateBoard = np.zeros((3, 3, 3, 3))
        self.player: Player = P1
        self.active_nonant: Optional[Section] = None

    @staticmethod
    def _in_range(sl: Section | Location) -> bool:
        (r, c) = sl
        return r < 3 and r >= 0 and c < 3 and c >= 0

    @staticmethod
    def to_action(sec: Section, loc: Location) -> Action:
        (R, C) = sec
        (r, c) = loc
        if not UltimateTicTacToe._in_range(sec):
            raise ValueError(
                f"Row {R} or column {C} outside of the Ultimate board, cannot convert {sec, loc} to action."
            )
        if not UltimateTicTacToe._in_range(loc):
            raise ValueError(
                f"Row {r} or column {c} outside of the section {R, C}, cannot convert {sec, loc} to action."
            )
        sec_ix = R * 3 + C
        loc_ix = r * 3 + c
        a = sec_ix * 9 + loc_ix
        return a

    @staticmethod
    def from_action(a: Action) -> Tuple[Section, Location]:
        # should be the same sec/loc as doing np.arange(81).reshape((3,3,3,3))
        sec_ix = int(a / 9)
        loc_ix = int(a % 9)
        R = int(sec_ix / 3)
        C = int(sec_ix % 3)
        r = int(loc_ix / 3)
        c = int(loc_ix % 3)

        sec = (R, C)
        loc = (r, c)
        return (sec, loc)

    @staticmethod
    def apply(s: UltimateState, a: Action) -> UltimateState:
        state = UltimateState(
            board=np.copy(s.board), player=s.player, active_nonant=s.active_nonant
        )
        ((sec), loc) = UltimateTicTacToe.from_action(a)
        (R, C) = sec
        (r, c) = loc
        action = (sec, loc)

        if not UltimateTicTacToe._in_range(sec):
            raise ValueError(
                f"Row {R} or column {C} outside of the Ultimate board, cannot apply {action} (from {a})."
            )

        if state.active_nonant and state.active_nonant != sec:
            (activeR, activeC) = state.active_nonant
            raise ValueError(
                f"Last play was in section {activeR, activeC}, cannot apply {action} (from {a})."
            )

        section: NDArray = state.board[R][C]
        if TicTacToe.check_finished(TicTacToeState(board=section, player=state.player)):
            raise ValueError(
                f"Section {R, C} is finished, cannot apply {action} (from {a})."
            )

        if not UltimateTicTacToe._in_range(loc):
            raise ValueError(
                f"Row {r} or column {c} outside of the section {R, C}, cannot apply {action} (from {a})."
            )

        if section[r, c] != Empty:
            raise ValueError(
                f"Section {R, C} location {r, c} already contains tile {tile_char(section[r, c])}"
            )

        section[r, c] = state.player

        player = switch_player(state.player)

        if TicTacToe.check_finished(
            TicTacToeState(board=state.board[loc], player=player)
        ):
            active_nonant = None
        else:
            active_nonant = loc

        return UltimateState(
            board=state.board,
            player=player,
            active_nonant=active_nonant,
        )

    @staticmethod
    def _3_in_a_row(p: Player, board: UltimateBoard) -> bool:
        # fmt: off
        return (
            # horizontal
            (TicTacToe._is_win(p, board[0][0]) and TicTacToe._is_win(p, board[0][1]) and TicTacToe._is_win(p, board[0][2])) or  # noqa: W504
            (TicTacToe._is_win(p, board[1][0]) and TicTacToe._is_win(p, board[1][1]) and TicTacToe._is_win(p, board[1][2])) or  # noqa: W504
            (TicTacToe._is_win(p, board[2][0]) and TicTacToe._is_win(p, board[2][1]) and TicTacToe._is_win(p, board[2][2])) or  # noqa: W504
            # vertical
            (TicTacToe._is_win(p, board[0][0]) and TicTacToe._is_win(p, board[1][0]) and TicTacToe._is_win(p, board[2][0])) or  # noqa: W504
            (TicTacToe._is_win(p, board[0][1]) and TicTacToe._is_win(p, board[1][1]) and TicTacToe._is_win(p, board[2][1])) or  # noqa: W504
            (TicTacToe._is_win(p, board[0][2]) and TicTacToe._is_win(p, board[1][2]) and TicTacToe._is_win(p, board[2][2])) or  # noqa: W504
            # diag
            (TicTacToe._is_win(p, board[0][0]) and TicTacToe._is_win(p, board[1][1]) and TicTacToe._is_win(p, board[2][2])) or  # noqa: W504
            (TicTacToe._is_win(p, board[2][0]) and TicTacToe._is_win(p, board[1][1]) and TicTacToe._is_win(p, board[0][2]))
        )
        # fmt: on

    @staticmethod
    def _is_win(p: Player, board: UltimateBoard) -> bool:
        # TODO: might be able to simplify this, but would take some refactoring
        threeinarow = UltimateTicTacToe._3_in_a_row(p, board)
        if threeinarow:
            return threeinarow

        opp = switch_player(p)
        if UltimateTicTacToe._3_in_a_row(opp, board):
            return False

        if not UltimateTicTacToe._is_board_filled(board):
            return False

        # if no one has 3 in a row, and board is filled, check if p has more mini wins
        p_wins = 0
        opp_wins = 0
        for r in range(3):
            for c in range(3):
                if TicTacToe._is_win(p, board[r][c]):
                    p_wins += 1
                if TicTacToe._is_win(opp, board[r][c]):
                    opp_wins += 1

        return p_wins > opp_wins

    def win(self, p: Player) -> bool:
        return self._is_win(p, self.board)

    @staticmethod
    def _is_board_filled(board: UltimateBoard) -> bool:
        for row in board:
            for b in row:
                s = TicTacToeState(board=b, player=P1)
                if not TicTacToe.check_finished(s):
                    return False
        return True

    def board_filled(self) -> bool:
        return self._is_board_filled(self.board)

    @staticmethod
    def check_finished(state: UltimateState) -> bool:
        board = state.board
        xWin = UltimateTicTacToe._is_win(P1, board)
        oWin = UltimateTicTacToe._is_win(P2, board)

        return xWin or oWin or UltimateTicTacToe._is_board_filled(board)

    def is_finished(self) -> bool:
        return UltimateTicTacToe.check_finished(self.state())

    @staticmethod
    def simplified_ttt_ir(t: TTTBoard) -> SimplifiedTTTIR:
        if TicTacToe._is_win(P1, t):
            return FinishedTTTState.XWin
        elif TicTacToe._is_win(P2, t):
            return FinishedTTTState.OWin
        elif TicTacToe._is_board_filled(t):
            return FinishedTTTState.Tie
        else:
            return TicTacToe.to_immutable(TicTacToeState(board=t, player=P1)).board

    @staticmethod
    def get_board_rep(board: UltimateBoard) -> UltimateBoardIR:
        # really ugly way to get correct typing :x
        # assumes that the board is 3 rows and 3 columns
        return (
            (
                UltimateTicTacToe.simplified_ttt_ir(board[0][0]),
                UltimateTicTacToe.simplified_ttt_ir(board[0][1]),
                UltimateTicTacToe.simplified_ttt_ir(board[0][2]),
            ),
            (
                UltimateTicTacToe.simplified_ttt_ir(board[1][0]),
                UltimateTicTacToe.simplified_ttt_ir(board[1][1]),
                UltimateTicTacToe.simplified_ttt_ir(board[1][2]),
            ),
            (
                UltimateTicTacToe.simplified_ttt_ir(board[2][0]),
                UltimateTicTacToe.simplified_ttt_ir(board[2][1]),
                UltimateTicTacToe.simplified_ttt_ir(board[2][2]),
            ),
        )

    def state(self) -> UltimateState:
        return UltimateState(
            board=self.board,
            player=self.player,
            active_nonant=self.active_nonant,
        )

    def play(self, sec: Section, loc: Location) -> None:
        a = UltimateTicTacToe.to_action(sec, loc)
        s = UltimateState(
            board=self.board, player=self.player, active_nonant=self.active_nonant
        )
        new_s = UltimateTicTacToe.apply(s, a)
        self.board = new_s.board
        self.player = new_s.player
        self.active_nonant = new_s.active_nonant

    @staticmethod
    def simplified_ttt_board(t: TTTBoard) -> List[str]:
        # should be the same structure (length of array, and length of strings in array) as TicTacToe.show()
        if TicTacToe._is_win(P1, t):
            return [
                "x     X",
                " X   X ",
                "  X X  ",
                "  X X  ",
                " X   X ",
                "X     X",
            ]
        elif TicTacToe._is_win(P2, t):
            return [
                "  OOO  ",
                " O   O ",
                "O     O",
                "O     O",
                " O   O ",
                "  OOO  ",
            ]
        elif TicTacToe.check_finished(TicTacToeState(board=t, player=P1)):
            return [
                "- - - -",
                " - - - ",
                "- - - -",
                " - - - ",
                "- - - -",
                " - - - ",
            ]
        else:
            return TicTacToe.board_array(t)

    @staticmethod
    def show_board(state: UltimateState) -> str:
        spacing_constant = "            |         |"
        final = [
            "        0         1         2",
            spacing_constant,
        ]

        for r in range(3):
            board_arrays = []
            for c in range(3):
                board_arrays.append(
                    UltimateTicTacToe.simplified_ttt_board(state.board[r][c])
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

    def show(self) -> str:
        return UltimateTicTacToe.show_board(self.state())

    @staticmethod
    def num_actions() -> int:
        return 81  # one for each area of the board

    @staticmethod
    def actions(state: UltimateState) -> List[ActionStatus]:
        actions = np.full((3, 3, 3, 3), INVAL)
        for R in range(3):
            for C in range(3):
                if state.active_nonant is not None and state.active_nonant != (R, C):
                    # if active_nonant, only that subboard has potentially valid spots
                    continue
                b = state.board[R][C]
                if TicTacToe.check_finished(TicTacToeState(board=b, player=P1)):
                    # any subboard that is finished is invalid
                    continue
                for r in range(3):
                    for c in range(3):
                        if b[r][c] == Empty:
                            actions[R, C, r, c] = VALID

        return list(actions.reshape(UltimateTicTacToe.num_actions()))

    @staticmethod
    def symmetries_of(b: NDArray) -> List[NDArray]:
        # TODO: test this?
        # TODO: is reshaping multiple times slow? can optimize by having a separate symmetries function specifically for policy vectors
        syms: List[NDArray] = []
        input_shape = b.shape
        b = b.reshape((3, 3, 3, 3))
        for i in range(1, 5):
            for mirror in [True, False]:
                s = np.rot90(np.rot90(b, i), i, (2, 3))
                if mirror:
                    # only flip along axis 1 and 3, essentially flipping each individual ttt board and the whole board
                    # don't need to flip the other 2 axes since those are covered by rotation and flip
                    s = np.flip(np.flip(s, axis=1), axis=3)
                syms.append(s.reshape(input_shape))
        return syms

    @staticmethod
    def to_immutable(state: UltimateState) -> UltimateIR:
        return UltimateIR(
            board=UltimateTicTacToe.get_board_rep(state.board),
            player=state.player,
            active_nonant=state.active_nonant,
        )

    @staticmethod
    def orient_state(state: UltimateState) -> UltimateState:
        return UltimateState(
            board=state.board * state.player,
            player=P1,
            active_nonant=state.active_nonant,
        )

    @staticmethod
    def calculate_reward(state: UltimateState) -> float:
        if UltimateTicTacToe._is_win(P1, state.board):
            return P1WIN
        elif UltimateTicTacToe._is_win(P2, state.board):
            return P2WIN
        elif UltimateTicTacToe._is_board_filled(state.board):
            # TODO: if board is filled, check who won more squares?
            return 0.1  # TODO: different value for draws?
        else:
            raise RuntimeError(f"Calling reward function when game not ended: {state}")
