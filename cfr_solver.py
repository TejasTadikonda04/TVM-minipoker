"""
CFR+ solver for MiniPoker (TeamTVM).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, Tuple

VALUES = ("T", "J", "Q", "K", "A")
SUITS = ("\u2660", "\u2665")  
DECK: Tuple[str, ...] = tuple(v + s for v in VALUES for s in SUITS)


def card_rank(c: str) -> int:
    """Numeric rank (0..4) of a card given as 'A♠' etc."""
    return VALUES.index(c[0])


def evaluate(private: str, table: str) -> Tuple[float, int]:
    """Replicates evaluate_hand() from the notebook."""
    pair = private[0] == table[0]
    suit_same = private[1:] == table[1:]
    straight = abs(card_rank(private) - card_rank(table)) == 1
    if pair:
        rank = 3.0
    elif straight and suit_same:
        rank = 2.5
    elif straight:
        rank = 2.0
    elif suit_same:
        rank = 1.0
    else:
        rank = 0.0
    return (rank, max(card_rank(private), card_rank(table)))


def resolve_util_for_p1(c1: str, c2: str, table: str, call_type: str) -> float:
    """Utility for P1 at a showdown using the given call type."""
    amount = {"C": 1, "C1": 2, "C2": 3}[call_type]
    h1 = evaluate(c1, table)
    h2 = evaluate(c2, table)
    if h1 > h2:
        return float(amount)
    if h2 > h1:
        return -float(amount)
    return 0.0


# Legal action sets
S1_P1_OPEN = ("C", "R1", "R2")
S1_P2_AFTER_CHECK = ("C", "R1", "R2")
S1_P2_AFTER_R1 = ("F", "C1")
S1_P2_AFTER_R2 = ("F", "C2")
S1_P1_AFTER_R1 = ("F", "C1")
S1_P1_AFTER_R2 = ("F", "C2")

S2_P1_OPEN = ("C", "R1", "R2")
S2_P2_AFTER_CHECK = ("C", "R1", "R2")
S2_P2_AFTER_R1 = ("F", "C1")
S2_P2_AFTER_R2 = ("F", "C2")
S2_P1_AFTER_R1 = ("F", "C1")
S2_P1_AFTER_R2 = ("F", "C2")


class CFRSolver:
    """
    Vanilla CFR+ with full chance-node enumeration.

    Data layout:
        regret[info_set][action]        cumulative positive regret (CFR+)
        strategy_sum[info_set][action]  cumulative (reach-weighted, linearly
                                        averaged) strategy probability

    Info-set keys:
        Stage 1: (player, my_card, history_tuple)
        Stage 2: (player, my_card, table_card, history_tuple)
    """

    def __init__(self) -> None:
        self.regret: Dict[tuple, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.strategy_sum: Dict[tuple, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.t: int = 0

    # Regret-matching+ strategy at an info set
    def _current_strategy(self, info: tuple, actions: Tuple[str, ...]) -> Dict[str, float]:
        regret_pos = {a: max(0.0, self.regret[info][a]) for a in actions}
        total = sum(regret_pos.values())
        if total > 0.0:
            return {a: regret_pos[a] / total for a in actions}
        # Uniform when no positive regret yet
        return {a: 1.0 / len(actions) for a in actions}

    # Regret & strategy-sum update for the acting player
    def _accumulate(
        self,
        info: tuple,
        actions: Tuple[str, ...],
        strat: Dict[str, float],
        action_utils: Dict[str, float],
        node_util: float,
        own_reach: float,
        opp_reach: float,
        acting_player: int,
    ) -> None:
        # From P1's perspective, node_util is returned. P1 maximizes, P2 minimizes.
        sign = 1.0 if acting_player == 1 else -1.0
        for a in actions:
            regret = sign * (action_utils[a] - node_util)
            # CFR+ floor at 0
            self.regret[info][a] = max(0.0, self.regret[info][a] + opp_reach * regret)
            # Linear averaging (weight by iteration t)
            self.strategy_sum[info][a] += self.t * own_reach * strat[a]

    # Traversal routines (utility always from P1's perspective)
    def _s1_p1_open(self, c1: str, c2: str, r1: float, r2: float) -> float:
        info = (1, c1, ())
        actions = S1_P1_OPEN
        strat = self._current_strategy(info, actions)
        utils: Dict[str, float] = {}
        node_util = 0.0
        for a in actions:
            hist = (f"P1-{a}",)
            u = self._s1_p2_respond(c1, c2, hist, r1 * strat[a], r2)
            utils[a] = u
            node_util += strat[a] * u
        self._accumulate(info, actions, strat, utils, node_util, r1, r2, acting_player=1)
        return node_util

    def _s1_p2_respond(self, c1: str, c2: str, hist: tuple, r1: float, r2: float) -> float:
        last_p1 = hist[-1].split("-")[1]
        info = (2, c2, hist)
        if last_p1 == "C":
            actions = S1_P2_AFTER_CHECK
        elif last_p1 == "R1":
            actions = S1_P2_AFTER_R1
        else:
            actions = S1_P2_AFTER_R2

        strat = self._current_strategy(info, actions)
        utils: Dict[str, float] = {}
        node_util = 0.0
        for a in actions:
            hist2 = hist + (f"P2-{a}",)
            if a == "F":
                # P2 folds -> P1 wins +1
                u = 1.0
            elif a == "C":
                # No raise yet, straight transition to stage 2
                u = self._chance_stage2(c1, c2, hist2, r1, r2 * strat[a])
            elif a in ("C1", "C2"):
                # P2 calls P1's raise, on to stage 2
                u = self._chance_stage2(c1, c2, hist2, r1, r2 * strat[a])
            elif a in ("R1", "R2"):
                # P2 raised P1's check; P1 gets to respond.
                u = self._s1_p1_respond(c1, c2, hist2, r1, r2 * strat[a])
            else:
                raise RuntimeError(f"Unknown action {a}")
            utils[a] = u
            node_util += strat[a] * u

        self._accumulate(info, actions, strat, utils, node_util, r2, r1, acting_player=2)
        return node_util

    def _s1_p1_respond(self, c1: str, c2: str, hist: tuple, r1: float, r2: float) -> float:
        last_p2 = hist[-1].split("-")[1]
        info = (1, c1, hist)
        actions = S1_P1_AFTER_R1 if last_p2 == "R1" else S1_P1_AFTER_R2
        strat = self._current_strategy(info, actions)
        utils: Dict[str, float] = {}
        node_util = 0.0
        for a in actions:
            hist2 = hist + (f"P1-{a}",)
            if a == "F":
                u = -1.0  # P1 folds -> loses 1
            else:
                u = self._chance_stage2(c1, c2, hist2, r1 * strat[a], r2)
            utils[a] = u
            node_util += strat[a] * u
        self._accumulate(info, actions, strat, utils, node_util, r1, r2, acting_player=1)
        return node_util

    # Chance node: reveal table card
    def _chance_stage2(self, c1: str, c2: str, hist: tuple, r1: float, r2: float) -> float:
        remaining = [c for c in DECK if c != c1 and c != c2]
        total = 0.0
        inv = 1.0 / len(remaining)
        for tc in remaining:
            total += inv * self._s2_p1_open(c1, c2, tc, hist, r1, r2)
        return total

    # Stage 2 decision nodes
    def _s2_p1_open(
        self, c1: str, c2: str, tc: str, hist_s1: tuple, r1: float, r2: float
    ) -> float:
        info = (1, c1, tc, hist_s1)
        actions = S2_P1_OPEN
        strat = self._current_strategy(info, actions)
        utils: Dict[str, float] = {}
        node_util = 0.0
        for a in actions:
            hist2 = hist_s1 + (f"P1-{a}",)
            u = self._s2_p2_respond(c1, c2, tc, hist_s1, hist2, r1 * strat[a], r2)
            utils[a] = u
            node_util += strat[a] * u
        self._accumulate(info, actions, strat, utils, node_util, r1, r2, acting_player=1)
        return node_util

    def _s2_p2_respond(
        self,
        c1: str,
        c2: str,
        tc: str,
        hist_s1: tuple,
        hist: tuple,
        r1: float,
        r2: float,
    ) -> float:
        last_p1 = hist[-1].split("-")[1]
        info = (2, c2, tc, hist)
        if last_p1 == "C":
            actions = S2_P2_AFTER_CHECK
        elif last_p1 == "R1":
            actions = S2_P2_AFTER_R1
        else:
            actions = S2_P2_AFTER_R2

        strat = self._current_strategy(info, actions)
        utils: Dict[str, float] = {}
        node_util = 0.0
        for a in actions:
            hist2 = hist + (f"P2-{a}",)
            if a == "F":
                u = 1.0  # P2 folds -> P1 wins +1
            elif a == "C":
                # check-check resolve with 'C'
                u = resolve_util_for_p1(c1, c2, tc, "C")
            elif a in ("C1", "C2"):
                u = resolve_util_for_p1(c1, c2, tc, a)
            elif a in ("R1", "R2"):
                u = self._s2_p1_respond(c1, c2, tc, hist_s1, hist2, r1, r2 * strat[a])
            else:
                raise RuntimeError(f"Unknown action {a}")
            utils[a] = u
            node_util += strat[a] * u
        self._accumulate(info, actions, strat, utils, node_util, r2, r1, acting_player=2)
        return node_util

    def _s2_p1_respond(
        self,
        c1: str,
        c2: str,
        tc: str,
        hist_s1: tuple,
        hist: tuple,
        r1: float,
        r2: float,
    ) -> float:
        last_p2 = hist[-1].split("-")[1]
        info = (1, c1, tc, hist)
        actions = S2_P1_AFTER_R1 if last_p2 == "R1" else S2_P1_AFTER_R2
        strat = self._current_strategy(info, actions)
        utils: Dict[str, float] = {}
        node_util = 0.0
        for a in actions:
            if a == "F":
                u = -1.0
            else:
                u = resolve_util_for_p1(c1, c2, tc, a)
            utils[a] = u
            node_util += strat[a] * u
        self._accumulate(info, actions, strat, utils, node_util, r1, r2, acting_player=1)
        return node_util

    # Public entry point
    def iterate(self) -> float:
        """One CFR+ iteration with full chance enumeration at the root."""
        self.t += 1
        total = 0.0
        n = len(DECK)
        inv = 1.0 / (n * (n - 1))
        for c1 in DECK:
            for c2 in DECK:
                if c1 == c2:
                    continue
                u = self._s1_p1_open(c1, c2, 1.0, 1.0)
                total += inv * u
        return total

    # Extract the averaged strategy (this is the Nash approximant)
    def average_strategy(self) -> Dict[tuple, Dict[str, float]]:
        avg: Dict[tuple, Dict[str, float]] = {}
        for info, counts in self.strategy_sum.items():
            total = sum(counts.values())
            if total > 0.0:
                avg[info] = {a: counts[a] / total for a in counts}
            else:
                # Unreached info set - fall back to uniform over legal actions
                avg[info] = {a: 1.0 / len(counts) for a in counts}
        return avg


# Exploitability evaluation (best response computation)

def exploitability(strategy: Dict[tuple, Dict[str, float]]) -> float:
    """
    Compute the exploitability of `strategy` played by BOTH players, i.e.
    the value of a best-response versus it, averaged over both positions.
    Lower is better; 0 is an exact Nash equilibrium.

    We reuse the solver structure but with a best responder at every info set.
    """

    def best_util_for_p1(c1, c2, tc, hist_s1, hist, r1, r2):
        raise NotImplementedError

    def br_value(br_player: int) -> float:
        memo: Dict[tuple, float] = {}

        def traverse(node):
            if node in memo:
                return memo[node]
            raise NotImplementedError

        # Implemented below using explicit recursion similar to the solver.
        return _best_response_ev(strategy, br_player)

    ev_br1 = br_value(1)
    ev_br2 = br_value(2)
    return (ev_br1 + (-ev_br2)) / 2.0


def _best_response_ev(strategy: Dict[tuple, Dict[str, float]], br_player: int) -> float:
    """EV for P1 when br_player plays a best response and the other uses `strategy`."""

    def strat(info, actions):
        probs = strategy.get(info)
        if probs is None:
            return {a: 1.0 / len(actions) for a in actions}
        return {a: probs.get(a, 0.0) for a in actions}

    def choose_best(utils, player):
        if player == 1:
            return max(utils.values())
        return min(utils.values())

    def s1_p1_open(c1, c2):
        info = (1, c1, ())
        actions = S1_P1_OPEN
        utils = {a: s1_p2_respond(c1, c2, (f"P1-{a}",)) for a in actions}
        if br_player == 1:
            return max(utils.values())
        probs = strat(info, actions)
        return sum(probs[a] * utils[a] for a in actions)

    def s1_p2_respond(c1, c2, hist):
        last_p1 = hist[-1].split("-")[1]
        info = (2, c2, hist)
        if last_p1 == "C":
            actions = S1_P2_AFTER_CHECK
        elif last_p1 == "R1":
            actions = S1_P2_AFTER_R1
        else:
            actions = S1_P2_AFTER_R2

        utils = {}
        for a in actions:
            hist2 = hist + (f"P2-{a}",)
            if a == "F":
                utils[a] = 1.0
            elif a == "C":
                utils[a] = chance_stage2(c1, c2, hist2)
            elif a in ("C1", "C2"):
                utils[a] = chance_stage2(c1, c2, hist2)
            elif a in ("R1", "R2"):
                utils[a] = s1_p1_respond(c1, c2, hist2)
        if br_player == 2:
            # P2 minimizes P1's utility
            return min(utils.values())
        probs = strat(info, actions)
        return sum(probs[a] * utils[a] for a in actions)

    def s1_p1_respond(c1, c2, hist):
        last_p2 = hist[-1].split("-")[1]
        info = (1, c1, hist)
        actions = S1_P1_AFTER_R1 if last_p2 == "R1" else S1_P1_AFTER_R2
        utils = {}
        for a in actions:
            hist2 = hist + (f"P1-{a}",)
            if a == "F":
                utils[a] = -1.0
            else:
                utils[a] = chance_stage2(c1, c2, hist2)
        if br_player == 1:
            return max(utils.values())
        probs = strat(info, actions)
        return sum(probs[a] * utils[a] for a in actions)

    def chance_stage2(c1, c2, hist_s1):
        remaining = [c for c in DECK if c != c1 and c != c2]
        inv = 1.0 / len(remaining)
        total = 0.0
        for tc in remaining:
            total += inv * s2_p1_open(c1, c2, tc, hist_s1)
        return total

    def s2_p1_open(c1, c2, tc, hist_s1):
        info = (1, c1, tc, hist_s1)
        actions = S2_P1_OPEN
        utils = {a: s2_p2_respond(c1, c2, tc, hist_s1, hist_s1 + (f"P1-{a}",)) for a in actions}
        if br_player == 1:
            return max(utils.values())
        probs = strat(info, actions)
        return sum(probs[a] * utils[a] for a in actions)

    def s2_p2_respond(c1, c2, tc, hist_s1, hist):
        last_p1 = hist[-1].split("-")[1]
        info = (2, c2, tc, hist)
        if last_p1 == "C":
            actions = S2_P2_AFTER_CHECK
        elif last_p1 == "R1":
            actions = S2_P2_AFTER_R1
        else:
            actions = S2_P2_AFTER_R2

        utils = {}
        for a in actions:
            hist2 = hist + (f"P2-{a}",)
            if a == "F":
                utils[a] = 1.0
            elif a == "C":
                utils[a] = resolve_util_for_p1(c1, c2, tc, "C")
            elif a in ("C1", "C2"):
                utils[a] = resolve_util_for_p1(c1, c2, tc, a)
            elif a in ("R1", "R2"):
                utils[a] = s2_p1_respond(c1, c2, tc, hist_s1, hist2)
        if br_player == 2:
            return min(utils.values())
        probs = strat(info, actions)
        return sum(probs[a] * utils[a] for a in actions)

    def s2_p1_respond(c1, c2, tc, hist_s1, hist):
        last_p2 = hist[-1].split("-")[1]
        info = (1, c1, tc, hist)
        actions = S2_P1_AFTER_R1 if last_p2 == "R1" else S2_P1_AFTER_R2
        utils = {}
        for a in actions:
            if a == "F":
                utils[a] = -1.0
            else:
                utils[a] = resolve_util_for_p1(c1, c2, tc, a)
        if br_player == 1:
            return max(utils.values())
        probs = strat(info, actions)
        return sum(probs[a] * utils[a] for a in actions)

    total = 0.0
    n = len(DECK)
    inv = 1.0 / (n * (n - 1))
    for c1 in DECK:
        for c2 in DECK:
            if c1 == c2:
                continue
            total += inv * s1_p1_open(c1, c2)
    return total


# Serialization
def serialize_strategy(strategy: Dict[tuple, Dict[str, float]]) -> str:
    """Convert strategy dict to a pretty-printed Python literal string."""
    lines = ["STRATEGY = {"]
    # Sort for deterministic output
    for info in sorted(strategy.keys(), key=lambda k: (len(k), str(k))):
        probs = strategy[info]
        probs_str = "{" + ", ".join(
            f"{a!r}: {probs[a]:.6f}" for a in sorted(probs.keys())
        ) + "}"
        lines.append(f"    {info!r}: {probs_str},")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--out", default="strategy_data.py")
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    solver = CFRSolver()
    t0 = time.time()
    for it in range(1, args.iters + 1):
        ev = solver.iterate()
        if it % args.log_every == 0 or it == 1:
            elapsed = time.time() - t0
            print(f"  iter {it:5d}  ev_p1_selfplay={ev:+.5f}  elapsed={elapsed:.1f}s")

    avg = solver.average_strategy()
    expl = _best_response_ev(avg, 1) + (-_best_response_ev(avg, 2))
    expl /= 2.0
    print(f"\nFinal exploitability estimate: {expl:.5f}  (0 = exact Nash)")
    print(f"Number of info-sets covered: {len(avg)}")

    # Write strategy_data.py
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, args.out)
    header = (
        '"""Auto-generated by cfr_solver.py. DO NOT EDIT BY HAND.\n\n'
        f'Iterations: {args.iters}\n'
        f'Exploitability estimate: {expl:.5f}\n'
        '"""\n\n# fmt: off\n'
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(serialize_strategy(avg))
        f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
