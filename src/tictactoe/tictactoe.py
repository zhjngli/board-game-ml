from enum import Enum, auto
from typing import List, NamedTuple, Tuple

from q_learners.monte_carlo import MonteCarloLearner


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
        self.reset()

    def reset(self) -> None:
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
        self.player = Tile.O if self.player == Tile.X else Tile.X

    @staticmethod
    def apply(state: State, r: int, c: int) -> State:
        s = State(board=state.board, player=state.player)

        if s.board[r][c] != Tile.N:
            raise ValueError(
                f"Board already contains tile {s.board[r][c]} at row {r} column {c}"
            )

        board = list(list(row) for row in s.board)
        board[r][c] = s.player
        new_s = State(
            board=tuple(tuple(row) for row in board),
            player=Tile.O if s.player == Tile.X else Tile.X,
        )

        return new_s

    @staticmethod
    def _is_win(t: Tile, board: List) -> bool:
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
    def _is_board_filled(board: List) -> bool:
        return all(all(t != Tile.N for t in row) for row in board)

    def board_filled(self) -> bool:
        return self._is_board_filled(self.board)

    @staticmethod
    def _is_finished(board: List) -> bool:
        xWin = TicTacToe._is_win(Tile.X, board)
        oWin = TicTacToe._is_win(Tile.O, board)

        return xWin or oWin or TicTacToe._is_board_filled(board)

    def finished(self) -> bool:
        return self._is_finished(self.board)

    def get_state(self) -> State:
        return State(tuple([tuple(row) for row in self.board]), self.player)

    def play(self, r: int, c: int) -> None:
        self._play(self.player, r, c)

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


class TicTacToeMonteCarloTrainer(TicTacToe):
    def __init__(self, p1: MonteCarloLearner, p2: MonteCarloLearner) -> None:
        super().__init__()
        self.p1 = p1
        self.p2 = p2

    def train(self, episodes=1000) -> None:
        for e in range(1, episodes + 1):
            self.train_once()

            if e % 100 == 0:
                print(f"Episode {e}/{episodes}")

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


class TicTacToeMonteCarloLearner(MonteCarloLearner[State, Tuple[int, int]]):
    def get_actions_from_state(self, state: State) -> List[Tuple[int, int]]:
        return [
            (i, j) for i in range(3) for j in range(3) if state.board[i][j] == Tile.N
        ]

    def apply(self, state: State, action: Tuple[int, int]) -> State:
        r, c = action
        return TicTacToe.apply(state, r, c)


def computer_play(g: TicTacToe, p: TicTacToeMonteCarloLearner):
    print(f"\n{g.show()}\n")
    r, c = p.choose_action(g.get_state(), exploit=True)
    g.play(r, c)
    print(f"\ncomputer plays at ({r}, {c})!")


def human_play(g: TicTacToe):
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


def monte_carlo_trained_game(training_episodes=0):  # noqa: C901
    computer1 = TicTacToeMonteCarloLearner(policy_file="src/tictactoe/p1.pkl")
    computer2 = TicTacToeMonteCarloLearner(policy_file="src/tictactoe/p2.pkl")
    g = TicTacToeMonteCarloTrainer(p1=computer1, p2=computer2)
    g.train(episodes=training_episodes)

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
        computer_play(g, computer1)
    else:
        print("unrecognized input, defaulting to player 1!")

    while not g.finished():
        if p == 0:
            computer_play(g, computer2)
        else:
            try:
                human_play(g)
            except ValueError as e:
                print(str(e))
                continue

        if g.finished():
            break

        computer = computer1 if p == 0 or p == 2 else computer2
        computer_play(g, computer)

    print(f"\n{g.show()}\n")
    print("game over!")
    if g.win(Tile.X):
        print("X won!")
    elif g.win(Tile.O):
        print("O won!")
    else:
        print("tie!")


def monte_carlo_many_games(games=10000):
    computer1 = TicTacToeMonteCarloLearner(policy_file="src/tictactoe/p1.pkl")
    computer2 = TicTacToeMonteCarloLearner(policy_file="src/tictactoe/p2.pkl")
    g = TicTacToeMonteCarloTrainer(p1=computer1, p2=computer2)

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
