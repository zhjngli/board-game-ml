import random
from abc import ABC, abstractmethod
from collections import Counter
from math import ceil
from typing import List, NamedTuple, Tuple

from digit_party.data import max_conns
from q_learners.default import DefaultQLearner


class Tile(ABC):
    @property
    @abstractmethod
    def val(self) -> int:
        pass

    def __str__(self) -> str:
        return "."


class Digit(Tile):
    def __init__(self, val: int) -> None:
        if val <= 0:
            raise ValueError("Digit must have positive integer value")
        self._val = val

    @property
    def val(self) -> int:
        return self._val

    def __eq__(self, d) -> bool:
        return self.val == d.val

    def __str__(self) -> str:
        return str(self._val)

    def __hash__(self) -> int:
        return hash(self._val)


class Empty(Tile):
    @property
    def val(self) -> int:
        raise ValueError("Empty tile has no value")

    def __eq__(self, e) -> bool:
        return isinstance(e, Empty)

    def __hash__(self) -> int:
        return hash(0)


class State(NamedTuple):
    board: Tuple  # TODO: better type
    next: Tuple[Digit | None, Digit | None]


Action = Tuple[int, int]


class DigitParty:
    digit_ratio = 9 / 25

    def __init__(self, n: int = 5, digits: List[int] | None = None) -> None:
        self.n = n
        self.board: List[List[Tile]] = [[Empty() for _ in range(n)] for _ in range(n)]
        self.max_num = ceil(self.digit_ratio * n * n)
        if digits:
            if len(digits) < self.n * self.n:
                raise ValueError(
                    f"Only {len(digits)} given digits, but board has size {self.n * self.n}"
                )
            self.digits = list(map(lambda d: Digit(d), digits))
        else:
            self.digits = [Digit(random.randint(1, self.max_num)) for _ in range(n * n)]
        self.placements: List[Tuple[Tuple[int, int], Digit]] = []
        self.score = 0

    def theoretical_max_score(self) -> int:
        score = 0
        counts = Counter(list(map(lambda p: p[1], self.placements)))
        for d in counts:
            score += max_conns[counts[d]] * d.val
        return score

    def _check_range(self, r: int, c: int) -> None:
        if r < 0 or r >= self.n or c < 0 or c >= self.n:
            raise ValueError(f"Row {r} or column {c} outside of board of size {self.n}")

    def place(self, r: int, c: int) -> None:
        """
        Places the next digit on the given r,c tile.
        """
        self._check_range(r, c)
        if isinstance(self.board[r][c], Digit):
            raise ValueError(
                f"Board already contains tile {self.board[r][c].val} at row {r} column {c}"
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
                self._check_range(r + dr, c + dc)
                if self.board[r + dr][c + dc] == d:
                    self.score += d.val
            except ValueError:
                continue

    def finished(self) -> bool:
        return not self.digits and len(self.placements) == self.n * self.n

    def _intersperse_board(self) -> List[List[Tile | str]]:
        """
        Intersperses the board with space to depict connections.
        """
        newlen = self.n + self.n - 1
        matrix = []
        for r in range(newlen):
            row: List[Tile | str] = []
            for c in range(newlen):
                if r % 2 == 1 or c % 2 == 1:
                    row.append("")
                else:
                    row.append(self.board[int(r / 2)][int(c / 2)])
            matrix.append(row)
        return matrix

    def _add_connection(self, r: int, c: int, matrix: List[List[Tile | str]]) -> None:
        if r % 2 == 0 and c % 2 == 1:
            # in between tiles on a row of tiles
            lt = matrix[r][c - 1]
            rt = matrix[r][c + 1]
            if isinstance(lt, Digit) and isinstance(rt, Digit) and lt.val == rt.val:
                matrix[r][c] = "---"

        elif r % 2 == 1 and c % 2 == 0:
            # in between tiles on a col of tiles
            up = matrix[r - 1][c]
            dn = matrix[r + 1][c]
            if isinstance(up, Digit) and isinstance(dn, Digit) and up.val == dn.val:
                matrix[r][c] = "|"

        elif r % 2 == 1 and c % 2 == 1:
            # diagonally centered between 4 tiles
            ul = matrix[r - 1][c - 1]
            ur = matrix[r - 1][c + 1]
            dl = matrix[r + 1][c - 1]
            dr = matrix[r + 1][c + 1]
            if (
                isinstance(ul, Digit)
                and isinstance(dr, Digit)
                and isinstance(ur, Digit)
                and isinstance(dl, Digit)
                and ur.val == dl.val
                and ul.val == dr.val
                # only check cross equivalence since we can score that way too
            ):
                matrix[r][c] = "X"
            elif isinstance(ul, Digit) and isinstance(dr, Digit) and ul.val == dr.val:
                matrix[r][c] = "\\"
            elif isinstance(ur, Digit) and isinstance(dl, Digit) and ur.val == dl.val:
                matrix[r][c] = "/"

    def _add_connections(
        self, matrix: List[List[Tile | str]]
    ) -> List[List[Tile | str]]:
        """
        Adds connections in between the board tiles.
        """
        # TODO: do this per placement instead of per board render. though it doesn't matter much since the game isn't meant to be played by users
        for r in range(len(matrix)):
            for c in range(len(matrix[0])):
                if isinstance(matrix[r][c], Tile):
                    continue

                self._add_connection(r, c, matrix)

        return matrix

    def show_board(self) -> str:
        matrix = self._add_connections(self._intersperse_board())

        s = [[str(e) for e in row] for row in matrix]
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

    def get_state(self) -> State:
        return State(tuple(tuple(row) for row in self.board), self.next_digits())


def human_game() -> None:
    x = input("hi this is digit party. what game size? (default 5): ").strip()
    if x == "":
        n = 5
    else:
        n = int(x)

    ds = input(
        "do you want to input a series of digits? (for testing, default random): "
    )
    if ds == "":
        game = DigitParty(n=n, digits=None)
    else:
        game = DigitParty(
            n=n, digits=list(map(lambda s: int(s.strip()), ds.split(",")))
        )

    while not game.finished():
        print(game.show_board())
        print(f"current score: {game.score}")
        curr_digit, next_digit = game.next_digits()
        print(f"current digit: {curr_digit}")
        print(f"next digit: {next_digit}")
        print()
        coord = input(
            "give me 0-indexed row col coords from the top left to place the current digit (delimit with ','): "
        ).strip()
        print()

        try:
            rc = coord.split(",")[:2]
            r = int(rc[0])
            c = int(rc[1])
        except (ValueError, IndexError):
            print("can't read your coordinate input")
            continue

        try:
            game.place(r, c)
        except ValueError as e:
            print(str(e))

    print(game.show_board())
    print("game finished!")
    print(f"your score: {game.score}")
    print(f"theoretical max score: {game.theoretical_max_score()}")
    print(f"% of total: {100 * game.score / game.theoretical_max_score()}")


class DigitPartyQLearner(DefaultQLearner[State, Action]):
    def get_actions_from_state(self, state: State) -> List[Action]:
        r = len(state.board)
        c = len(state.board[0])
        return [
            (i, j) for i in range(r) for j in range(c) if Empty() == state.board[i][j]
        ]

    def train_once(self) -> None:
        g = DigitParty()

        while not g.finished():
            curr_score = g.score
            state = g.get_state()
            action = self.choose_action(state)

            r, c = action
            g.place(r, c)
            new_score = g.score

            self.update_q_value(state, action, new_score - curr_score, g.get_state())


def trained_game() -> None:
    q = DigitPartyQLearner(q_pickle="src/digit_party/q.pkl")
    q.train(episodes=100000)
    g = DigitParty()

    while not g.finished():
        print(f"\n{g.show_board()}\n")
        curr_digit, next_digit = g.next_digits()
        print(f"current digit: {curr_digit}")
        print(f"next digit: {next_digit}")
        r, c = q.choose_action(g.get_state(), exploit=True)
        g.place(r, c)
        print(f"\ncomputer plays {curr_digit} at ({r}, {c})!")

    print(g.show_board())
    print("game finished!")
    print(f"computer score: {g.score}")
    print(f"theoretical max score: {g.theoretical_max_score()}")
    print(f"% of total: {100 * g.score / g.theoretical_max_score()}")
