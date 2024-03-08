import copy
import random
from collections import Counter
from math import ceil
from typing import List, NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray

from games.digit_party.data import max_conns
from games.game import P1, Action, ActionStatus, BasicState, Game, Player

"""
For more information about this game, see the following links:
https://digit.party/
https://www.cambridgemathshub.co.uk/post/digit-party-on
https://fivethirtyeight.com/features/how-much-money-can-you-pull-out-of-a-hat/
"""


Digit = int  # TODO: enforce positive only?
Empty: Digit = 0


def digit_to_char(d: Digit) -> str:
    if d == Empty:
        return "."
    else:
        return str(int(d))


class DigitPartyState(BasicState):
    def __init__(
        self,
        board: NDArray,
        player: Player,
        next: Tuple[Digit | None, Digit | None],
        score: int,
        digits: List[Digit],
    ) -> None:
        super().__init__(board, player)
        self.next = next
        self.score = score
        # digits not intended to be used for training
        self.digits = digits


class DigitPartyIR(NamedTuple):
    board: Tuple  # TODO: better type
    next: Tuple[Digit | None, Digit | None]


DigitPartyPlacement = Tuple[int, int]


class DigitParty(Game[DigitPartyState, DigitPartyIR]):
    digit_ratio = 9 / 25

    def __init__(self, n: int = 5, digits: List[int] | None = None) -> None:
        self.n = n
        self.board: NDArray = np.zeros((n, n))
        self.max_num = ceil(self.digit_ratio * n * n)
        if digits is not None:
            if len(digits) < self.n * self.n:
                raise ValueError(
                    f"Only {len(digits)} given digits, but board has size"
                    f" {self.n * self.n}"
                )
            self.digits: List[Digit] = list(map(lambda d: Digit(d), digits))
        else:
            self.digits = [Digit(random.randint(1, self.max_num)) for _ in range(n * n)]
        self.placements: List[Tuple[Tuple[int, int], Digit]] = []
        self.score = 0

    def reset(self) -> None:
        self.digits = [
            Digit(random.randint(1, self.max_num)) for _ in range(self.n * self.n)
        ]
        self.board = np.zeros((self.n, self.n))
        self.placements = []
        self.score = 0

    def theoretical_max_score(self) -> int:
        score = 0
        counts = Counter(list(map(lambda p: p[1], self.placements)))
        for d in counts:
            score += max_conns[counts[d]] * d
        return score

    @staticmethod
    def _check_range(n: int, r: int, c: int) -> None:
        if r < 0 or r >= n or c < 0 or c >= n:
            raise ValueError(f"Row {r} or column {c} outside of board of size {n}")

    def place(self, r: int, c: int) -> None:
        """
        Places the next digit on the given r,c tile.
        """
        DigitParty._check_range(self.n, r, c)
        if self.board[r][c] != Empty:
            raise ValueError(
                f"Board already contains tile {self.board[r][c]} at row"
                f" {r} column {c}"
            )
        d = self.digits.pop()
        self.board[r][c] = d
        self.placements.append(
            (
                (
                    r,
                    c,
                ),
                d,
            )
        )

        for dr, dc in [
            (-1, -1),  # up left
            (-1, 0),  # up
            (-1, 1),  # up right
            (0, -1),  # left
            (0, 1),  # right
            (1, -1),  # down left
            (1, 0),  # down
            (1, 1),  # down right
        ]:
            try:
                DigitParty._check_range(self.n, r + dr, c + dc)
                if self.board[r + dr][c + dc] == d:
                    self.score += d
            except ValueError:
                continue

    @staticmethod
    def apply(s: DigitPartyState, a: Action) -> DigitPartyState:
        state = DigitPartyState(
            board=np.copy(s.board),
            player=s.player,
            next=s.next,
            score=s.score,
            digits=copy.deepcopy(s.digits),
        )
        shape = state.board.shape
        n = shape[0]  # TODO: needs to change if generalized to non-square boards
        r = int(a / n)
        c = a % n
        DigitParty._check_range(n, r, c)

        d = state.digits.pop()
        state.board[r][c] = d

        for dr, dc in [
            (-1, -1),  # up left
            (-1, 0),  # up
            (-1, 1),  # up right
            (0, -1),  # left
            (0, 1),  # right
            (1, -1),  # down left
            (1, 0),  # down
            (1, 1),  # down right
        ]:
            try:
                DigitParty._check_range(n, r + dr, c + dc)
                if state.board[r + dr][c + dc] == d:
                    state.score += d
            except ValueError:
                continue

        state.next = DigitParty.next_digits_from_digits(state.digits)
        return state

    def is_finished(self) -> bool:
        return not self.digits and len(self.placements) == self.n * self.n

    @staticmethod
    def check_finished(state: DigitPartyState) -> bool:
        return bool(np.all([state != 0]))

    def _intersperse_board(self) -> List[List[Digit | str]]:
        """
        Intersperses the board with space to depict connections.
        """
        newlen = self.n + self.n - 1
        matrix = []
        for r in range(newlen):
            row: List[Digit | str] = []
            for c in range(newlen):
                if r % 2 == 1 or c % 2 == 1:
                    row.append("")
                else:
                    row.append(self.board[int(r / 2)][int(c / 2)])
            matrix.append(row)
        return matrix

    def _add_connection(self, r: int, c: int, matrix: List[List[Digit | str]]) -> None:
        if r % 2 == 0 and c % 2 == 1:
            # in between tiles on a row of tiles
            lt = matrix[r][c - 1]
            rt = matrix[r][c + 1]
            if lt != Empty and rt != Empty and lt == rt:
                matrix[r][c] = "---"

        elif r % 2 == 1 and c % 2 == 0:
            # in between tiles on a col of tiles
            up = matrix[r - 1][c]
            dn = matrix[r + 1][c]
            if up != Empty and dn != Empty and up == dn:
                matrix[r][c] = "|"

        elif r % 2 == 1 and c % 2 == 1:
            # diagonally centered between 4 tiles
            ul = matrix[r - 1][c - 1]
            ur = matrix[r - 1][c + 1]
            dl = matrix[r + 1][c - 1]
            dr = matrix[r + 1][c + 1]
            if (
                ul != Empty
                and ur != Empty
                and dl != Empty
                and dr != Empty
                and ur == dl
                and ul == dr
                # only check cross equivalence since we can score that way too
            ):
                matrix[r][c] = "X"
            elif ul != Empty and dr != Empty and ul == dr:
                matrix[r][c] = "\\"
            elif ur != Empty and dl != Empty and ur == dl:
                matrix[r][c] = "/"

    def _add_connections(
        self, matrix: List[List[Digit | str]]
    ) -> List[List[Digit | str]]:
        """
        Adds connections in between the board tiles.
        """
        # TODO: do this per placement instead of per board render. though it doesn't matter much since the game isn't meant to be played by users
        for r in range(len(matrix)):
            for c in range(len(matrix[0])):
                if isinstance(matrix[r][c], Digit):
                    continue

                self._add_connection(r, c, matrix)

        return matrix

    def show_board(self) -> str:
        matrix = self._add_connections(self._intersperse_board())

        s = [
            [(e if isinstance(e, str) else digit_to_char(e)) for e in row]
            for row in matrix
        ]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = "\t".join("{{:{}}}".format(x) for x in lens)
        table = [fmt.format(*row) for row in s]

        return "\n".join(table)

    def next_digits(self) -> Tuple[Digit | None, Digit | None]:
        if len(self.digits) >= 2:
            return self.digits[-1], self.digits[-2]
        elif len(self.digits) == 1:
            return self.digits[0], None
        else:
            return None, None

    @staticmethod
    def next_digits_from_digits(
        digits: List[Digit],
    ) -> Tuple[Digit | None, Digit | None]:
        if len(digits) >= 2:
            return digits[-1], digits[-2]
        elif len(digits) == 1:
            return digits[0], None
        else:
            return None, None

    @staticmethod
    def to_immutable(state: DigitPartyState) -> DigitPartyIR:
        return DigitPartyIR(
            board=tuple(tuple(row) for row in state.board), next=state.next
        )

    def state(self) -> DigitPartyState:
        return DigitPartyState(
            board=self.board,
            player=P1,
            next=self.next_digits(),
            score=self.score,
            digits=self.digits,
        )

    @staticmethod
    def orient_state(state: DigitPartyState) -> DigitPartyState:
        # DigitParty is single player, no need to orient it
        return state

    @staticmethod
    def actions(state: DigitPartyState) -> List[ActionStatus]:
        b = np.copy(state.board)
        b[b == 0] = 1
        b[b != 0] = 0
        return list(b.reshape(state.board.size))

    def num_actions(self) -> int:
        return self.n * self.n

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
    def calculate_reward(state: DigitPartyState) -> float:
        # TODO: should this be difference between current score and previous score?
        return state.score
