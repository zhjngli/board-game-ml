from enum import Enum, auto
from typing import List, NamedTuple, Tuple

from q_learners.default import DefaultQLearner


class Tile(Enum):
    N = auto()
    X = auto()
    O = auto()  # noqa: E741

    def __str__(self) -> str:
        if self.name == "N":
            return " "
        return f"{self.name}"


class State(NamedTuple):
    board: Tuple  # TODO: better board type
    player: Tile


class TicTacToe:
    def __init__(self) -> None:
        self.board = [[Tile.N for _ in range(3)] for _ in range(3)]
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

    def tie(self) -> bool:
        return all(all(t != Tile.N for t in row) for row in self.board)

    def finished(self) -> bool:
        xWin = self.win(Tile.X)
        oWin = self.win(Tile.O)

        return xWin or oWin or self.tie()

    def get_state(self) -> State:
        return State(tuple(tuple(row) for row in self.board), self.player)

    def play(self, r: int, c: int) -> None:
        self._play(self.player, r, c)
        self.player = Tile.O if self.player == Tile.X else Tile.X

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


# TODO: this learner is still an idiot
class TicTacToeQLearner(DefaultQLearner[State, Tuple[int, int]]):
    def get_actions_from_state(self, state: State) -> List[Tuple[int, int]]:
        return [
            (i, j) for i in range(3) for j in range(3) if state.board[i][j] == Tile.N
        ]

    # TODO: punish learner for leaving winning spots for its oppponent
    def train_once(self) -> None:
        g = TicTacToe()

        while True:
            state = g.get_state()
            action = self.choose_action(state)

            r, c = action
            g.play(r, c)

            if g.win(g.player):
                reward = 1.0
                self.update_q_value(state, action, reward, g.get_state())
                break
            elif g.tie():
                reward = 0.5
                self.update_q_value(state, action, reward, g.get_state())
                break


def trained_game() -> None:
    q = TicTacToeQLearner(q_pickle="src/tictactoe/q.pkl")
    q.train(episodes=10000)
    g = TicTacToe()

    player = input("play as player 1 or 2? ").strip()
    if int(player) == 1:
        pass
    elif int(player) == 2:
        print(f"\n{g.show()}\n")
        print("\ncomputer's turn!")
        r, c = q.choose_action(g.get_state())
        g.play(r, c)
    else:
        print("unrecognized input, defaulting to player 1!")

    while not g.finished():
        print(f"\n{g.show()}\n")

        coord = input("please choose a spot to play: ").strip()
        try:
            rc = coord.split(",")[:2]
            r = int(rc[0])
            c = int(rc[1])
        except (ValueError, IndexError):
            print("can't read your coordinate input")
            continue

        try:
            g.play(r, c)
        except ValueError as e:
            print(str(e))
            continue

        if g.finished():
            break

        print(f"\n\n{g.show()}\n")
        r, c = q.choose_action(g.get_state(), exploit=True)
        g.play(r, c)
        print(f"\ncomputer plays at ({r}, {c})!")

    print(f"\n{g.show()}\n")
    print("game over!")
