from enum import Enum, auto


class Tile(Enum):
    N = auto()
    X = auto()
    O = auto()  # noqa: E741

    def __str__(self) -> str:
        if self.name == "N":
            return " "
        return f"{self.name}"


class TicTacToe:
    def __init__(self) -> None:
        self.board = [[Tile.N for _ in range(3)] for _ in range(3)]

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

    def win(self, t: Tile) -> bool:
        # fmt: off
        return (
            # horizontal
            (self.board[0][0] == t and self.board[0][1] == t and self.board[0][2] == t) or  # noqa: W504
            (self.board[1][0] == t and self.board[1][1] == t and self.board[1][2] == t) or  # noqa: W504
            (self.board[2][0] == t and self.board[2][1] == t and self.board[2][2] == t) or  # noqa: W504
            # vertical
            (self.board[0][0] == t and self.board[1][0] == t and self.board[2][0] == t) or  # noqa: W504
            (self.board[0][1] == t and self.board[1][1] == t and self.board[2][1] == t) or  # noqa: W504
            (self.board[0][2] == t and self.board[1][2] == t and self.board[2][2] == t) or  # noqa: W504
            # diag
            (self.board[0][0] == t and self.board[1][1] == t and self.board[2][2] == t) or  # noqa: W504
            (self.board[2][0] == t and self.board[1][1] == t and self.board[0][2] == t)
        )
        # fmt: on

    def finished(self) -> bool:
        xWin = self.win(Tile.X)
        oWin = self.win(Tile.O)
        tie = all(all(t != Tile.N for t in row) for row in self.board)

        return xWin or oWin or tie

    def play1(self, r: int, c: int) -> None:
        self._play(Tile.X, r, c)

    def play2(self, r: int, c: int) -> None:
        self._play(Tile.O, r, c)

    def show(self) -> str:
        s = [
            "  0 1 2",
            f"0 {self.board[0][0]}|{self.board[0][1]}|{self.board[0][2]}",
            "  -+-+-",
            f"1 {self.board[1][0]}|{self.board[1][1]}|{self.board[1][2]}",
            "  -+-+-",
            f"2 {self.board[2][0]}|{self.board[2][1]}|{self.board[2][2]}",
        ]
        return "\n".join(s)


def game():
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
