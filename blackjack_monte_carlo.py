from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Optional

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
FULL_DECK = Counter({r: 4 for r in RANKS})

HILO_VALUES = {
    "2": 1, "3": 1, "4": 1, "5": 1, "6": 1,
    "7": 0, "8": 0, "9": 0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
}

BLACKJACK_PAYOUT = 1.5  # 3:2
MIN_CARDS_TO_START_NEW_ROUND = 6
DEFAULT_MAX_BET_MULTIPLIER = 20


class ShoeExhausted(RuntimeError):
    pass


def card_value(rank: str) -> int:
    if rank == "A":
        return 11
    if rank in {"J", "Q", "K"}:
        return 10
    return int(rank)


def hand_value(cards: list[str]) -> tuple[int, bool]:
    total = 0
    aces = 0

    for c in cards:
        if c == "A":
            total += 11
            aces += 1
        elif c in {"J", "Q", "K"}:
            total += 10
        else:
            total += int(c)

    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    soft = aces > 0
    return total, soft


def is_blackjack(cards: list[str]) -> bool:
    total, _ = hand_value(cards)
    return len(cards) == 2 and total == 21


def dealer_should_hit(cards: list[str], hit_soft_17: bool) -> bool:
    total, soft = hand_value(cards)
    if total < 17:
        return True
    if total == 17 and soft and hit_soft_17:
        return True
    return False


def recommend_action(player_cards: list[str], dealer_upcard: str) -> str:
    total, soft = hand_value(player_cards)
    dealer = card_value(dealer_upcard)

    if total >= 21:
        return "STAY"

    if soft:
        if total <= 17:
            return "HIT"
        if total == 18:
            return "STAY" if dealer in {2, 3, 4, 5, 6, 7, 8} else "HIT"
        return "STAY"

    if total <= 11:
        return "HIT"
    if total == 12:
        return "STAY" if dealer in {4, 5, 6} else "HIT"
    if 13 <= total <= 16:
        return "STAY" if dealer in {2, 3, 4, 5, 6} else "HIT"
    return "STAY"


def compare_hands(player_cards: list[str], dealer_cards: list[str]) -> str:
    p_total, _ = hand_value(player_cards)
    d_total, _ = hand_value(dealer_cards)

    p_bj = is_blackjack(player_cards)
    d_bj = is_blackjack(dealer_cards)

    if p_bj and d_bj:
        return "PUSH"
    if p_bj:
        return "PLAYER BLACKJACK"
    if d_bj:
        return "DEALER BLACKJACK"
    if p_total > 21:
        return "PLAYER BUST"
    if d_total > 21:
        return "DEALER BUST"
    if p_total > d_total:
        return "PLAYER WIN"
    if p_total < d_total:
        return "DEALER WIN"
    return "PUSH"


def build_shoe() -> list[str]:
    shoe: list[str] = []
    for rank, count in FULL_DECK.items():
        shoe.extend([rank] * count)
    random.shuffle(shoe)
    return shoe


def fmt_cards(cards: list[str]) -> str:
    return "[" + ", ".join(cards) + "]"


@dataclass
class ShoeTracker:
    bankroll: float
    starting_bankroll: float
    table_min: float
    table_max: float
    hit_soft_17: bool
    shoe: list[str] = field(default_factory=build_shoe)
    seen: list[str] = field(default_factory=list)
    round_no: int = 0
    peak_bankroll: float = 0.0
    lowest_bankroll: float = 0.0
    max_drawdown: float = 0.0
    peak_since_start: float = 0.0
    trace_rows: list[dict[str, object]] = field(default_factory=list)
    stop_reason: str = ""

    def __post_init__(self) -> None:
        self.peak_bankroll = self.bankroll
        self.lowest_bankroll = self.bankroll
        self.peak_since_start = self.bankroll

    def draw_card(self) -> str:
        if not self.shoe:
            raise ShoeExhausted("The shoe is empty.")
        return self.shoe.pop()

    def reveal_card(self, card: str) -> None:
        self.seen.append(card)

    def draw_and_reveal(self) -> str:
        card = self.draw_card()
        self.reveal_card(card)
        return card

    def running_count(self) -> int:
        return sum(HILO_VALUES[c] for c in self.seen)

    def cards_left(self) -> int:
        return len(self.shoe)

    def decks_left(self) -> float:
        return max(self.cards_left() / 52.0, 0.01)

    def true_count(self) -> float:
        return self.running_count() / self.decks_left()

    def recommended_bet(self) -> float:
        tc = self.true_count()

        if tc <= 0:
            multiplier = 1
        elif tc < 1:
            multiplier = 2
        elif tc < 2:
            multiplier = 4
        elif tc < 3:
            multiplier = 6
        elif tc < 4:
            multiplier = 8
        elif tc < 5:
            multiplier = 10
        elif tc < 6:
            multiplier = 12
        elif tc < 7:
            multiplier = 15
        else:
            multiplier = 20

        return min(self.table_max, self.table_min * multiplier)

    def estimated_edge(self) -> float:
        return -0.005 + 0.005 * self.true_count()

    def estimated_win_prob(self) -> float:
        tc = self.true_count()
        return min(max(0.42 + 0.015 * tc, 0.30), 0.60)

    def estimated_push_prob(self) -> float:
        tc = self.true_count()
        return min(max(0.08 - 0.002 * abs(tc), 0.04), 0.12)

    def estimated_loss_prob(self) -> float:
        return 1.0 - self.estimated_win_prob() - self.estimated_push_prob()

    def expected_value(self, bet: float) -> float:
        return bet * self.estimated_edge()

    def settle_bankroll(self, result: str, bet: float) -> None:
        if result == "PLAYER WIN":
            self.bankroll += bet
        elif result == "DEALER WIN":
            self.bankroll -= bet
        elif result == "PLAYER BLACKJACK":
            self.bankroll += bet * BLACKJACK_PAYOUT
        elif result == "DEALER BLACKJACK":
            self.bankroll -= bet
        elif result == "DEALER BUST":
            self.bankroll += bet
        elif result == "PLAYER BUST":
            self.bankroll -= bet

        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
        self.lowest_bankroll = min(self.lowest_bankroll, self.bankroll)
        self.peak_since_start = max(self.peak_since_start, self.bankroll)
        self.max_drawdown = max(self.max_drawdown, self.peak_since_start - self.bankroll)

    def record_trace(
        self,
        round_no: int,
        bet: float,
        player_cards: list[str],
        dealer_visible: str,
        dealer_hole_card: str,
        dealer_hole_revealed: bool,
        dealer_cards_for_total: list[str],
        result: str,
        actions: list[str],
        bankroll_before: float,
        notes: str = "",
    ) -> None:
        player_total, _ = hand_value(player_cards)
        dealer_total, _ = hand_value(dealer_cards_for_total)

        self.trace_rows.append(
            {
                "round": round_no,
                "bet": round(bet, 2),
                "bankroll_before": round(bankroll_before, 2),
                "bankroll_after": round(self.bankroll, 2),
                "profit_loss_after": round(self.bankroll - self.starting_bankroll, 2),
                "peak_bankroll": round(self.peak_bankroll, 2),
                "lowest_bankroll": round(self.lowest_bankroll, 2),
                "player_cards": fmt_cards(player_cards),
                "player_total": player_total,
                "dealer_visible": dealer_visible,
                "dealer_hole_card": dealer_hole_card,
                "dealer_hole_revealed": dealer_hole_revealed,
                "dealer_final": fmt_cards(dealer_cards_for_total) if dealer_hole_revealed else f"[{dealer_visible}, ?]",
                "dealer_total": dealer_total if dealer_hole_revealed else "Hidden",
                "running_count": self.running_count(),
                "true_count": round(self.true_count(), 2),
                "cards_left": self.cards_left(),
                "result": result,
                "actions": " | ".join(actions),
                "notes": notes,
            }
        )


def simulate_one_shoe(tracker: ShoeTracker, keep_trace: bool = False) -> None:
    while tracker.cards_left() >= MIN_CARDS_TO_START_NEW_ROUND:
        # Snapshot the current completed state so any mid-hand shoe exhaustion can be rolled back.
        state = {
            "shoe": tracker.shoe.copy(),
            "seen": tracker.seen.copy(),
            "bankroll": tracker.bankroll,
            "peak_bankroll": tracker.peak_bankroll,
            "lowest_bankroll": tracker.lowest_bankroll,
            "max_drawdown": tracker.max_drawdown,
            "peak_since_start": tracker.peak_since_start,
            "round_no": tracker.round_no,
            "trace_len": len(tracker.trace_rows),
        }

        tracker.round_no += 1
        round_no = tracker.round_no
        bankroll_before = tracker.bankroll
        actions: list[str] = []

        try:
            bet = tracker.recommended_bet()

            player: list[str] = []
            dealer: list[str] = []
            dealer_hole_revealed = False

            p1 = tracker.draw_and_reveal()
            player.append(p1)
            actions.append(f"Player first card: {p1}")

            d_up = tracker.draw_and_reveal()
            dealer.append(d_up)
            actions.append(f"Dealer upcard: {d_up}")

            p2 = tracker.draw_and_reveal()
            player.append(p2)
            actions.append(f"Player second card: {p2}")

            dealer_hole = tracker.draw_card()
            actions.append(f"Dealer hole card drawn hidden: {dealer_hole}")

            # Dealer peek on Ace or 10-value upcard.
            if d_up in {"A", "10", "J", "Q", "K"} and is_blackjack([d_up, dealer_hole]):
                tracker.reveal_card(dealer_hole)
                dealer.append(dealer_hole)
                dealer_hole_revealed = True
                actions.append(f"Dealer peek reveals blackjack hole card: {dealer_hole}")

                result = compare_hands(player, dealer)
                tracker.settle_bankroll(result, bet)
                if keep_trace:
                    tracker.record_trace(
                        round_no=round_no,
                        bet=bet,
                        player_cards=player,
                        dealer_visible=d_up,
                        dealer_hole_card=dealer_hole,
                        dealer_hole_revealed=True,
                        dealer_cards_for_total=dealer,
                        result=result,
                        actions=actions,
                        bankroll_before=bankroll_before,
                        notes="Dealer blackjack on peek",
                    )
                continue

            # Player blackjack.
            if is_blackjack(player):
                if not dealer_hole_revealed:
                    tracker.reveal_card(dealer_hole)
                    dealer.append(dealer_hole)
                    dealer_hole_revealed = True
                    actions.append(f"Dealer hole card revealed: {dealer_hole}")

                result = compare_hands(player, dealer)
                tracker.settle_bankroll(result, bet)
                if keep_trace:
                    tracker.record_trace(
                        round_no=round_no,
                        bet=bet,
                        player_cards=player,
                        dealer_visible=d_up,
                        dealer_hole_card=dealer_hole,
                        dealer_hole_revealed=True,
                        dealer_cards_for_total=dealer,
                        result=result,
                        actions=actions,
                        bankroll_before=bankroll_before,
                        notes="Player blackjack",
                    )
                continue

            # Player turn.
            player_bust = False
            while True:
                p_total, _ = hand_value(player)
                action = recommend_action(player, d_up)
                actions.append(f"Decision at {p_total}: {action}")

                if p_total > 21:
                    player_bust = True
                    result = "PLAYER BUST"
                    tracker.settle_bankroll(result, bet)
                    if keep_trace:
                        tracker.record_trace(
                            round_no=round_no,
                            bet=bet,
                            player_cards=player,
                            dealer_visible=d_up,
                            dealer_hole_card=dealer_hole,
                            dealer_hole_revealed=False,
                            dealer_cards_for_total=[d_up],
                            result=result,
                            actions=actions,
                            bankroll_before=bankroll_before,
                            notes="Dealer hole card not revealed",
                        )
                    break

                move = "H" if action == "HIT" else "S"
                actions.append(f"Auto move: {move}")

                if move == "H":
                    new_card = tracker.draw_and_reveal()
                    player.append(new_card)
                    actions.append(f"Player hits: {new_card}")
                    continue

                break

            if player_bust:
                continue

            # Reveal dealer hole card if needed.
            if not dealer_hole_revealed:
                tracker.reveal_card(dealer_hole)
                dealer.append(dealer_hole)
                dealer_hole_revealed = True
                actions.append(f"Dealer hole card revealed: {dealer_hole}")

            # Dealer turn.
            while dealer_should_hit(dealer, tracker.hit_soft_17):
                hit_card = tracker.draw_and_reveal()
                dealer.append(hit_card)
                actions.append(f"Dealer hits: {hit_card}")

            result = compare_hands(player, dealer)
            tracker.settle_bankroll(result, bet)

            if keep_trace:
                tracker.record_trace(
                    round_no=round_no,
                    bet=bet,
                    player_cards=player,
                    dealer_visible=d_up,
                    dealer_hole_card=dealer_hole,
                    dealer_hole_revealed=True,
                    dealer_cards_for_total=dealer,
                    result=result,
                    actions=actions,
                    bankroll_before=bankroll_before,
                    notes="Completed hand",
                )

        except ShoeExhausted:
            # Roll back the incomplete hand and end the session at the last completed hand.
            tracker.shoe = state["shoe"]
            tracker.seen = state["seen"]
            tracker.bankroll = state["bankroll"]
            tracker.peak_bankroll = state["peak_bankroll"]
            tracker.lowest_bankroll = state["lowest_bankroll"]
            tracker.max_drawdown = state["max_drawdown"]
            tracker.peak_since_start = state["peak_since_start"]
            tracker.round_no = state["round_no"]
            tracker.trace_rows = tracker.trace_rows[: state["trace_len"]]
            tracker.stop_reason = "Shoe exhausted mid-hand"
            break

    if not tracker.stop_reason:
        tracker.stop_reason = (
            #"Bankroll depleted" if tracker.bankroll <= 0 else
            "Not enough cards left" if tracker.cards_left() < MIN_CARDS_TO_START_NEW_ROUND else
            "Unknown"
        )


def export_csv(rows: list[dict[str, object]], filepath: Path) -> None:
    if not rows:
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class SessionResult:
    session_no: int
    final_bankroll: float
    final_profit_loss: float
    peak_bankroll: float
    peak_profit: float
    lowest_bankroll: float
    max_drawdown: float
    rounds_played: int
    stop_reason: str


def run_one_session(
    session_no: int,
    starting_bankroll: float,
    table_min: float,
    hit_soft_17: bool,
    keep_trace: bool = False,
) -> tuple[SessionResult, list[dict[str, object]]]:
    tracker = ShoeTracker(
        bankroll=starting_bankroll,
        starting_bankroll=starting_bankroll,
        table_min=table_min,
        table_max=table_min * DEFAULT_MAX_BET_MULTIPLIER,
        hit_soft_17=hit_soft_17,
        shoe=build_shoe(),
    )
    simulate_one_shoe(tracker, keep_trace=keep_trace)

    result = SessionResult(
        session_no=session_no,
        final_bankroll=tracker.bankroll,
        final_profit_loss=tracker.bankroll - starting_bankroll,
        peak_bankroll=tracker.peak_bankroll,
        peak_profit=tracker.peak_bankroll - starting_bankroll,
        lowest_bankroll=tracker.lowest_bankroll,
        max_drawdown=tracker.max_drawdown,
        rounds_played=tracker.round_no,
        stop_reason=tracker.stop_reason,
    )
    return result, tracker.trace_rows


def main() -> None:
    print("Blackjack Monte Carlo Simulator (Exact Logic)")

    hit_soft_17_choice = input("Dealer hits soft 17? (y/n): ").strip().lower()
    hit_soft_17 = hit_soft_17_choice == "y"

    starting_bankroll = float(input("Starting bankroll: $").strip())
    table_min = float(input("Table minimum bet: $").strip())
    num_sessions = int(input("How many simulations? ").strip())

    out_dir = Path("simulation_output_exact")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[SessionResult] = []
    all_traces: list[dict[str, object]] = []

    for i in range(1, num_sessions + 1):
        keep_trace = i == 1
        session_result, trace_rows = run_one_session(
            session_no=i,
            starting_bankroll=starting_bankroll,
            table_min=table_min,
            hit_soft_17=hit_soft_17,
            keep_trace=keep_trace,
        )
        results.append(session_result)
        if keep_trace:
            all_traces.extend(trace_rows)

        print(
            f"Session {i}: final P/L ${session_result.final_profit_loss:.2f}, "
            f"peak profit ${session_result.peak_profit:.2f}, "
            f"max drawdown ${session_result.max_drawdown:.2f}, "
            f"rounds {session_result.rounds_played}"
        )

    overview_rows = [
        {
            "sessions": num_sessions,
            "starting_bankroll": starting_bankroll,
            "table_min": table_min,
            "table_max": table_min * DEFAULT_MAX_BET_MULTIPLIER,
            "hit_soft_17": hit_soft_17,
            "avg_final_profit_loss": round(mean(r.final_profit_loss for r in results), 2),
            "median_final_profit_loss": round(median(r.final_profit_loss for r in results), 2),
            "avg_peak_profit": round(mean(r.peak_profit for r in results), 2),
            "median_peak_profit": round(median(r.peak_profit for r in results), 2),
            "avg_lowest_bankroll": round(mean(r.lowest_bankroll for r in results), 2),
            "avg_max_drawdown": round(mean(r.max_drawdown for r in results), 2),
            "max_peak_profit": round(max(r.peak_profit for r in results), 2),
            "min_final_profit_loss": round(min(r.final_profit_loss for r in results), 2),
            "percent_positive_final": round(100.0 * sum(r.final_profit_loss > 0 for r in results) / len(results), 2),
            "percent_positive_peak": round(100.0 * sum(r.peak_profit > 0 for r in results) / len(results), 2),
        }
    ]

    results_path = out_dir / "simulation_results.csv"
    overview_path = out_dir / "simulation_overview.csv"
    trace_path = out_dir / "first_session_trace.csv"

    export_csv([
        {
            "session_no": r.session_no,
            "final_bankroll": round(r.final_bankroll, 2),
            "final_profit_loss": round(r.final_profit_loss, 2),
            "peak_bankroll": round(r.peak_bankroll, 2),
            "peak_profit": round(r.peak_profit, 2),
            "lowest_bankroll": round(r.lowest_bankroll, 2),
            "max_drawdown": round(r.max_drawdown, 2),
            "rounds_played": r.rounds_played,
            "stop_reason": r.stop_reason,
        }
        for r in results
    ], results_path)

    export_csv(overview_rows, overview_path)
    export_csv(all_traces, trace_path)

    print("\n--- SUMMARY ---")
    print(f"Average final profit/loss: ${overview_rows[0]['avg_final_profit_loss']:.2f}")
    print(f"Median final profit/loss: ${overview_rows[0]['median_final_profit_loss']:.2f}")
    print(f"Average peak profit: ${overview_rows[0]['avg_peak_profit']:.2f}")
    print(f"Median peak profit: ${overview_rows[0]['median_peak_profit']:.2f}")
    print(f"Average max drawdown: ${overview_rows[0]['avg_max_drawdown']:.2f}")
    print(f"Positive final sessions: {overview_rows[0]['percent_positive_final']:.2f}%")
    print(f"Results CSV: {results_path}")
    print(f"Overview CSV: {overview_path}")
    print(f"First-session trace CSV: {trace_path}")


if __name__ == "__main__":
    main()
