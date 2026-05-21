from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Literal

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
FULL_DECK = Counter({r: 4 for r in RANKS})
HILO_VALUES = {
    "2": 1, "3": 1, "4": 1, "5": 1, "6": 1,
    "7": 0, "8": 0, "9": 0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
}

BLACKJACK_PAYOUT = 1.5  # 3:2
MIN_CARDS_TO_START_NEW_ROUND = 6
MAX_BET_MULTIPLIER = 20
StrategyName = Literal["flat", "betramp"]


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
    return total, aces > 0


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
class SessionResult:
    session_no: int
    strategy: str
    final_bankroll: float
    final_profit_loss: float
    peak_bankroll: float
    peak_profit: float
    lowest_bankroll: float
    peak_loss: float
    max_drawdown: float
    rounds_played: int
    stop_reason: str


@dataclass
class ShoeTracker:
    bankroll: float
    starting_bankroll: float
    table_min: float
    table_max: float
    hit_soft_17: bool
    strategy: StrategyName
    shoe: list[str]
    seen: list[str] = None
    round_no: int = 0
    peak_bankroll: float = 0.0
    lowest_bankroll: float = 0.0
    peak_since_start: float = 0.0
    max_drawdown: float = 0.0
    stop_reason: str = ""

    def __post_init__(self) -> None:
        self.seen = [] if self.seen is None else self.seen
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
        if self.strategy == "flat":
            return self.table_min

        tc = self.true_count()
        if tc <= 0:
            mult = 1
        elif tc < 1:
            mult = 2
        elif tc < 2:
            mult = 4
        elif tc < 3:
            mult = 6
        elif tc < 4:
            mult = 8
        elif tc < 5:
            mult = 10
        elif tc < 6:
            mult = 12
        elif tc < 7:
            mult = 15
        else:
            mult = 20
        return min(self.table_max, self.table_min * mult)

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


def simulate_one_session(
    session_no: int,
    starting_bankroll: float,
    table_min: float,
    hit_soft_17: bool,
    strategy: StrategyName,
    shoe: list[str],
) -> SessionResult:
    tracker = ShoeTracker(
        bankroll=starting_bankroll,
        starting_bankroll=starting_bankroll,
        table_min=table_min,
        table_max=table_min * MAX_BET_MULTIPLIER,
        hit_soft_17=hit_soft_17,
        strategy=strategy,
        shoe=shoe.copy(),
    )

    while tracker.cards_left() >= MIN_CARDS_TO_START_NEW_ROUND:
        try:
            tracker.round_no += 1
            bet = tracker.recommended_bet()

            player: list[str] = []
            dealer: list[str] = []
            dealer_hole_revealed = False

            p1 = tracker.draw_and_reveal()
            d_up = tracker.draw_and_reveal()
            p2 = tracker.draw_and_reveal()
            dealer_hole = tracker.draw_card()

            player.extend([p1, p2])
            dealer.append(d_up)

            # Dealer peek on Ace / 10-value upcard.
            if d_up in {"A", "10", "J", "Q", "K"} and is_blackjack([d_up, dealer_hole]):
                tracker.reveal_card(dealer_hole)
                dealer.append(dealer_hole)
                dealer_hole_revealed = True
                result = compare_hands(player, dealer)
                tracker.settle_bankroll(result, bet)
                continue

            # Player blackjack.
            if is_blackjack(player):
                tracker.reveal_card(dealer_hole)
                dealer.append(dealer_hole)
                dealer_hole_revealed = True
                result = compare_hands(player, dealer)
                tracker.settle_bankroll(result, bet)
                continue

            # Player turn.
            while True:
                p_total, _ = hand_value(player)
                if p_total > 21:
                    tracker.settle_bankroll("PLAYER BUST", bet)
                    break

                action = recommend_action(player, d_up)
                if action == "HIT":
                    player.append(tracker.draw_and_reveal())
                    continue
                break

            if hand_value(player)[0] > 21:
                continue

            # Reveal dealer hole card if needed.
            if not dealer_hole_revealed:
                tracker.reveal_card(dealer_hole)
                dealer.append(dealer_hole)
                dealer_hole_revealed = True

            while dealer_should_hit(dealer, hit_soft_17):
                dealer.append(tracker.draw_and_reveal())

            result = compare_hands(player, dealer)
            tracker.settle_bankroll(result, bet)

        except ShoeExhausted:
            break

    tracker.stop_reason = "Not enough cards left" if tracker.cards_left() < MIN_CARDS_TO_START_NEW_ROUND else "Shoe exhausted mid-hand"

    return SessionResult(
        session_no=session_no,
        strategy=strategy,
        final_bankroll=tracker.bankroll,
        final_profit_loss=tracker.bankroll - starting_bankroll,
        peak_bankroll=tracker.peak_bankroll,
        peak_profit=tracker.peak_bankroll - starting_bankroll,
        lowest_bankroll=tracker.lowest_bankroll,
        peak_loss=starting_bankroll - tracker.lowest_bankroll,
        max_drawdown=tracker.max_drawdown,
        rounds_played=tracker.round_no,
        stop_reason=tracker.stop_reason,
    )


def export_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("Blackjack Flat Bet vs Bet+ Monte Carlo")

    hit_soft_17 = input("Dealer hits soft 17? (y/n): ").strip().lower() == "y"
    starting_bankroll = float(input("Starting bankroll: $").strip())
    table_min = float(input("Table minimum bet: $").strip())
    num_sessions = int(input("How many simulations? ").strip())

    out_dir = Path("simulation_output_flat_vs_betramp")
    out_dir.mkdir(parents=True, exist_ok=True)

    session_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []

    flat_results: list[SessionResult] = []
    ramp_results: list[SessionResult] = []

    for session_no in range(1, num_sessions + 1):
        shoe = build_shoe()

        flat = simulate_one_session(
            session_no=session_no,
            starting_bankroll=starting_bankroll,
            table_min=table_min,
            hit_soft_17=hit_soft_17,
            strategy="flat",
            shoe=shoe,
        )
        ramp = simulate_one_session(
            session_no=session_no,
            starting_bankroll=starting_bankroll,
            table_min=table_min,
            hit_soft_17=hit_soft_17,
            strategy="betramp",
            shoe=shoe,
        )

        flat_results.append(flat)
        ramp_results.append(ramp)

        session_rows.append({
            "session_no": session_no,
            "flat_final_profit_loss": round(flat.final_profit_loss, 2),
            "ramp_final_profit_loss": round(ramp.final_profit_loss, 2),
            "difference_ramp_minus_flat": round(ramp.final_profit_loss - flat.final_profit_loss, 2),
            "flat_peak_profit": round(flat.peak_profit, 2),
            "ramp_peak_profit": round(ramp.peak_profit, 2),
            "flat_peak_loss": round(flat.peak_loss, 2),
            "ramp_peak_loss": round(ramp.peak_loss, 2),
            "flat_max_drawdown": round(flat.max_drawdown, 2),
            "ramp_max_drawdown": round(ramp.max_drawdown, 2),
            "flat_rounds": flat.rounds_played,
            "ramp_rounds": ramp.rounds_played,
        })

        paired_rows.append({
            "session_no": session_no,
            "flat_final_profit_loss": round(flat.final_profit_loss, 2),
            "ramp_final_profit_loss": round(ramp.final_profit_loss, 2),
            "flat_peak_profit": round(flat.peak_profit, 2),
            "ramp_peak_profit": round(ramp.peak_profit, 2),
            "flat_peak_loss": round(flat.peak_loss, 2),
            "ramp_peak_loss": round(ramp.peak_loss, 2),
            "flat_max_drawdown": round(flat.max_drawdown, 2),
            "ramp_max_drawdown": round(ramp.max_drawdown, 2),
            "winner": "ramp" if ramp.final_profit_loss > flat.final_profit_loss else ("flat" if flat.final_profit_loss > ramp.final_profit_loss else "tie"),
        })

        print(
            f"Session {session_no}: flat ${flat.final_profit_loss:.2f}, "
            f"ramp ${ramp.final_profit_loss:.2f}, "
            f"difference ${ramp.final_profit_loss - flat.final_profit_loss:.2f}"
        )

    # Summary statistics.
    flat_final = [r.final_profit_loss for r in flat_results]
    ramp_final = [r.final_profit_loss for r in ramp_results]
    flat_peak_loss = [r.peak_loss for r in flat_results]
    ramp_peak_loss = [r.peak_loss for r in ramp_results]

    overview = [{
        "sessions": num_sessions,
        "starting_bankroll": starting_bankroll,
        "table_min": table_min,
        "table_max": table_min * MAX_BET_MULTIPLIER,
        "hit_soft_17": hit_soft_17,
        "flat_avg_final_profit_loss": round(mean(flat_final), 2),
        "flat_median_final_profit_loss": round(median(flat_final), 2),
        "flat_avg_peak_profit": round(mean(r.peak_profit for r in flat_results), 2),
        "flat_median_peak_profit": round(median(r.peak_profit for r in flat_results), 2),
        "flat_avg_peak_loss": round(mean(flat_peak_loss), 2),
        "flat_median_peak_loss": round(median(flat_peak_loss), 2),
        "flat_avg_max_drawdown": round(mean(r.max_drawdown for r in flat_results), 2),
        "flat_median_max_drawdown": round(median(r.max_drawdown for r in flat_results), 2),
        "ramp_avg_final_profit_loss": round(mean(ramp_final), 2),
        "ramp_median_final_profit_loss": round(median(ramp_final), 2),
        "ramp_avg_peak_profit": round(mean(r.peak_profit for r in ramp_results), 2),
        "ramp_median_peak_profit": round(median(r.peak_profit for r in ramp_results), 2),
        "ramp_avg_peak_loss": round(mean(ramp_peak_loss), 2),
        "ramp_median_peak_loss": round(median(ramp_peak_loss), 2),
        "ramp_avg_max_drawdown": round(mean(r.max_drawdown for r in ramp_results), 2),
        "ramp_median_max_drawdown": round(median(r.max_drawdown for r in ramp_results), 2),
        "ramp_better_count": sum(ramp_final[i] > flat_final[i] for i in range(num_sessions)),
        "flat_better_count": sum(flat_final[i] > ramp_final[i] for i in range(num_sessions)),
        "ties": sum(flat_final[i] == ramp_final[i] for i in range(num_sessions)),
        "avg_ramp_minus_flat": round(mean(ramp_final[i] - flat_final[i] for i in range(num_sessions)), 2),
        "median_ramp_minus_flat": round(median(ramp_final[i] - flat_final[i] for i in range(num_sessions)), 2),
        "flat_profitable_sessions": sum(r.final_profit_loss > 0 for r in flat_results),
        "ramp_profitable_sessions": sum(r.final_profit_loss > 0 for r in ramp_results),
    }]

    export_csv(session_rows, out_dir / "session_comparison.csv")
    export_csv(paired_rows, out_dir / "paired_session_winner.csv")
    export_csv(overview, out_dir / "overview.csv")

    print("\n--- SUMMARY ---")
    print(f"Flat avg final P/L: ${overview[0]['flat_avg_final_profit_loss']:.2f}")
    print(f"Flat median final P/L: ${overview[0]['flat_median_final_profit_loss']:.2f}")
    print(f"Flat avg peak loss: ${overview[0]['flat_avg_peak_loss']:.2f}")
    print(f"Flat median peak loss: ${overview[0]['flat_median_peak_loss']:.2f}")
    print(f"Average peak profit: ${overview[0]['flat_avg_peak_profit']:.2f}")
    print(f"Median peak profit: ${overview[0]['flat_median_peak_profit']:.2f}")
    print(f"Average max drawdown: ${overview[0]['flat_avg_max_drawdown']:.2f}")
    print(f"Profitable sessions: {overview[0]['flat_profitable_sessions']} / {num_sessions}")
    print(f"Ramp avg final P/L: ${overview[0]['ramp_avg_final_profit_loss']:.2f}")
    print(f"Ramp median final P/L: ${overview[0]['ramp_median_final_profit_loss']:.2f}")
    print(f"Ramp avg peak loss: ${overview[0]['ramp_avg_peak_loss']:.2f}")
    print(f"Ramp median peak loss: ${overview[0]['ramp_median_peak_loss']:.2f}")
    print(f"Average peak profit: ${overview[0]['ramp_avg_peak_profit']:.2f}")
    print(f"Median peak profit: ${overview[0]['ramp_median_peak_profit']:.2f}")
    print(f"Average max drawdown: ${overview[0]['ramp_avg_max_drawdown']:.2f}")
    print(f"Profitable sessions: {overview[0]['ramp_profitable_sessions']} / {num_sessions}")
    print(f"Ramp better in {overview[0]['ramp_better_count']} / {num_sessions} sessions")
    print(f"Flat better in {overview[0]['flat_better_count']} / {num_sessions} sessions")
    print(f"Ties: {overview[0]['ties']}")
    print(f"Average ramp-minus-flat final P/L: ${overview[0]['avg_ramp_minus_flat']:.2f}")
    print(f"Results saved in: {out_dir}")

    # print("\n--- FLAT BET SUMMARY ---")
    # print(f"Average peak profit: ${overview[0]['flat_avg_peak_profit']:.2f}")
    # print(f"Median peak profit: ${overview[0]['flat_median_peak_profit']:.2f}")
    # print(f"Average max drawdown: ${overview[0]['flat_avg_max_drawdown']:.2f}")
    # print(f"Profitable sessions: {overview[0]['flat_profitable_sessions']} / {num_sessions}")

    # print("\n--- BET RAMP SUMMARY ---")
    # print(f"Average peak profit: ${overview[0]['ramp_avg_peak_profit']:.2f}")
    # print(f"Median peak profit: ${overview[0]['ramp_median_peak_profit']:.2f}")
    # print(f"Average max drawdown: ${overview[0]['ramp_avg_max_drawdown']:.2f}")
    # print(f"Profitable sessions: {overview[0]['ramp_profitable_sessions']} / {num_sessions}")


if __name__ == "__main__":
    main()
