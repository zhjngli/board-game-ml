from typing import List, Tuple

from typing_extensions import override

from games.game import P1, P2
from games.tictactoe.tictactoe import Empty
from games.ultimate_ttt.ultimate import (
    FinishedTTTState,
    Location,
    Section,
    UltimateIR,
    UltimateTicTacToe,
    ir_to_state,
)
from learners.monte_carlo import MonteCarloLearner
from learners.trainer import Trainer


def human_game() -> None:
    g = UltimateTicTacToe()
    p = 0
    t = "XO"

    while not g.is_finished():
        print(f"\n{g.show()}\n")

        if g.active_nonant is None:
            print(
                "Last active section is finished, or it's a new game. You can freely choose a section to play in."
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


class UltimateMonteCarloTrainer(UltimateTicTacToe, Trainer):
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
            r, c = self.p1.choose_action(
                UltimateTicTacToe.immutable_representation(self.state())
            )
            self.play(r, c)
            self.p1.add_state(UltimateTicTacToe.immutable_representation(self.state()))

            if self.is_finished():
                break

            r, c = self.p2.choose_action(
                UltimateTicTacToe.immutable_representation(self.state())
            )
            self.play(r, c)
            self.p2.add_state(UltimateTicTacToe.immutable_representation(self.state()))

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
            raise Exception("giving rewards when game's not over. something's wrong!")


class UltimateMonteCarloLearner(
    MonteCarloLearner[UltimateIR, Tuple[Section, Location]]
):
    def get_actions_from_state(
        self, state: UltimateIR
    ) -> List[Tuple[Section, Location]]:
        if state.active_nonant is not None:
            (R, C) = state.active_nonant
            ttt_state = state.board[R][C]
            if (
                ttt_state != FinishedTTTState.Tie
                and ttt_state != FinishedTTTState.XWin
                and ttt_state != FinishedTTTState.OWin
            ):
                return [
                    ((R, C), (r, c))
                    for r in range(3)
                    for c in range(3)
                    if ttt_state[r][c] == Empty
                ]

        actions = []
        for R in range(3):
            for C in range(3):
                ttt_state = state.board[R][C]
                for r in range(3):
                    for c in range(3):
                        if (
                            ttt_state != FinishedTTTState.Tie
                            and ttt_state != FinishedTTTState.XWin
                            and ttt_state != FinishedTTTState.OWin
                        ):
                            if ttt_state[r][c] == Empty:
                                actions.append(((R, C), (r, c)))

        return actions

    def apply(self, ir: UltimateIR, action: Tuple[Section, Location]) -> UltimateIR:
        (sec, loc) = action
        a = UltimateTicTacToe.to_action(sec, loc)
        s = ir_to_state(ir)
        return UltimateTicTacToe.immutable_representation(UltimateTicTacToe.apply(s, a))


MCP1_POLICY = "src/games/ultimate_ttt/mcp1.pkl"
MCP2_POLICY = "src/games/ultimate_ttt/mcp2.pkl"


def trained_game():
    computer1 = UltimateMonteCarloLearner(policy_file=MCP1_POLICY)
    computer2 = UltimateMonteCarloLearner(policy_file=MCP2_POLICY)
    g = UltimateMonteCarloTrainer(p1=computer1, p2=computer2)
    g.train(episodes=1)

    while not g.is_finished():
        print(f"\n{g.show()}\n")
        sec, loc = computer1.choose_action(
            UltimateTicTacToe.immutable_representation(g.state()), exploit=True
        )
        g.play(sec, loc)
        print(f"computer X plays at section {sec} location {loc}")
        if g.is_finished():
            break

        print(f"\n{g.show()}\n")
        sec, loc = computer2.choose_action(
            UltimateTicTacToe.immutable_representation(g.state()), exploit=True
        )
        g.play(sec, loc)
        print(f"computer O plays at section {sec} location {loc}")

    print(g.show())
    print("\ngame over!")
