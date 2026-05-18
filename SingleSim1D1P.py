from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass, field
from math import erf, sqrt
from pathlib import Path

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
FULL_DECK = Counter({r: 4 for r in RANKS})

HILO_VALUES = {
    "2": 1, "3": 1, "4": 1, "5": 1, "6": 1,
    "7": 0, "8": 0, "9": 0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
}

BLACKJACK_PAYOUT = 1.5  # 3:2
MIN_CARDS_TO_START_NEW_ROUND = 4


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


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def fmt_cards(cards: list[str]) -> str:
    return "[" + ", ".join(cards) + "]"


def build_shoe() -> list[str]:
    shoe: list[str] = []
    for rank, count in FULL_DECK.items():
        shoe.extend([rank] * count)
    random.shuffle(shoe)
    return shoe


@dataclass
class ShoeTracker:
    bankroll: float
    starting_bankroll: float
    table_min: float
    table_max: float
    shoe: list[str]
    seen: list[str] = field(default_factory=list)
    round_no: int = 0
    peak_bankroll: float = 0.0
    lowest_bankroll: float = 0.0
    trace_rows: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.peak_bankroll = self.bankroll
        self.lowest_bankroll = self.bankroll

    def draw_card(self) -> str:
        if not self.shoe:
            raise RuntimeError("The shoe is empty.")
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

    def probability_profit_next_hand(self, bet: float) -> float:
        ev = self.expected_value(bet)
        sd = bet * sqrt(1.30)
        if sd <= 0:
            return 0.5
        return normal_cdf(ev / sd)

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

    def status(self) -> None:
        print(f"Bankroll: ${self.bankroll:.2f}")
        print(f"Profit/Loss: ${self.bankroll - self.starting_bankroll:.2f}")
        print(f"Seen cards: {len(self.seen)}/52")
        print(f"Cards left in shoe: {self.cards_left()}")
        print(f"Running count: {self.running_count()}")
        print(f"Decks left: {self.decks_left():.2f}")
        print(f"True count: {self.true_count():.2f}")

    def trace_row(
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


def export_trace_csv(tracker: ShoeTracker, filename: str = "blackjack_single_shoe_trace.csv") -> None:
    if not tracker.trace_rows:
        print("No trace data to export.")
        return

    fieldnames = list(tracker.trace_rows[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tracker.trace_rows)

    print(f"\nCSV trace written to: {filename}")


def simulate_one_shoe_trace() -> None:
    print("One-Deck Blackjack Single-Shoe Trace")

    hit_soft_17_choice = input("Dealer hits soft 17? (y/n): ").strip().lower()
    hit_soft_17 = hit_soft_17_choice == "y"

    starting_bankroll = float(input("Starting bankroll: $").strip())
    table_min = float(input("Table minimum bet: $").strip())
    table_max_input = input("Table maximum bet (press Enter for 20x min): ").strip()
    table_max = float(table_max_input) if table_max_input else table_min * 20

    step_mode = input("Pause after each decision? (y/n): ").strip().lower() == "y"

    tracker = ShoeTracker(
        bankroll=starting_bankroll,
        starting_bankroll=starting_bankroll,
        table_min=table_min,
        table_max=table_max,
        shoe=build_shoe(),
    )

    print("\n--- SHOE STARTED ---")
    tracker.status()

    while tracker.cards_left() >= MIN_CARDS_TO_START_NEW_ROUND and tracker.bankroll > 0:
        tracker.round_no += 1
        round_no = tracker.round_no
        bankroll_before = tracker.bankroll
        actions: list[str] = []

        bet = tracker.recommended_bet()
        edge = tracker.estimated_edge()
        win_p = tracker.estimated_win_prob()
        push_p = tracker.estimated_push_prob()
        loss_p = tracker.estimated_loss_prob()
        ev = tracker.expected_value(bet)
        profit_prob = tracker.probability_profit_next_hand(bet)

        print(f"\n========== ROUND {round_no} ==========")
        print(f"Suggested bet: ${bet:.2f}")
        print(f"True count: {tracker.true_count():.2f}")
        print(f"Estimated player edge: {edge * 100:.2f}%")
        print(f"Estimated EV on bet: ${ev:.2f}")
        print(f"Estimated win/push/loss: {win_p * 100:.1f}% / {push_p * 100:.1f}% / {loss_p * 100:.1f}%")
        print(f"Estimated chance of profit on next hand: {profit_prob * 100:.1f}%")

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

        p_total, p_soft = hand_value(player)
        print(f"Player hand: {fmt_cards(player)} -> {p_total}{' soft' if p_soft else ''}")
        print(f"Dealer shows: {d_up}")
        tracker.status()

        if d_up in {"A", "10", "J", "Q", "K"}:
            print("\nDealer checks for blackjack...")
            if is_blackjack([d_up, dealer_hole]):
                tracker.reveal_card(dealer_hole)
                dealer.append(dealer_hole)
                dealer_hole_revealed = True
                actions.append(f"Dealer peek reveals blackjack hole card: {dealer_hole}")

                print(f"Dealer has blackjack: {fmt_cards(dealer)}")
                result = compare_hands(player, dealer)
                print(f"Round result: {result}")

                tracker.settle_bankroll(result, bet)
                tracker.trace_row(
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
                tracker.status()
                if step_mode:
                    input("Press Enter for next round...")
                continue
            else:
                print("Dealer does not have blackjack.")
                actions.append("Dealer peek: no blackjack")
        else:
            print("No dealer blackjack peek needed.")
            actions.append("No dealer blackjack peek needed")

        if is_blackjack(player):
            print("Player has blackjack and stands.")
            actions.append("Player blackjack")

            if not dealer_hole_revealed:
                tracker.reveal_card(dealer_hole)
                dealer.append(dealer_hole)
                dealer_hole_revealed = True
                actions.append(f"Dealer hole card revealed: {dealer_hole}")

            result = compare_hands(player, dealer)
            print(f"Dealer hand: {fmt_cards(dealer)}")
            print(f"Round result: {result}")

            tracker.settle_bankroll(result, bet)
            tracker.trace_row(
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
            tracker.status()
            if step_mode:
                input("Press Enter for next round...")
            continue

        player_bust = False

        while True:
            p_total, p_soft = hand_value(player)
            action = recommend_action(player, d_up)

            print(f"Player hand: {fmt_cards(player)} -> {p_total}{' soft' if p_soft else ''}")
            print(f"Recommended action: {action}")

            if p_total > 21:
                player_bust = True
                print("Player busts.")
                actions.append("Player busts")
                result = "PLAYER BUST"
                tracker.settle_bankroll(result, bet)

                tracker.trace_row(
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
                tracker.status()
                if step_mode:
                    input("Press Enter for next round...")
                break

            move = input("Type H to hit, S to stay, or Enter to follow suggestion: ").strip().upper()
            if move == "":
                move = "H" if action == "HIT" else "S"

            if move == "H":
                new_card = tracker.draw_and_reveal()
                player.append(new_card)
                actions.append(f"Player hits: {new_card}")
                if step_mode:
                    input("Press Enter to continue...")
                continue

            if move == "S":
                actions.append("Player stays")
                break

            print("Please enter H or S.")

        if player_bust:
            continue

        if not dealer_hole_revealed:
            tracker.reveal_card(dealer_hole)
            dealer.append(dealer_hole)
            dealer_hole_revealed = True
            actions.append(f"Dealer hole card revealed: {dealer_hole}")

        while dealer_should_hit(dealer, hit_soft_17):
            d_total, d_soft = hand_value(dealer)
            print(f"Dealer hand: {fmt_cards(dealer)} -> {d_total}{' soft' if d_soft else ''}")
            hit_card = tracker.draw_and_reveal()
            dealer.append(hit_card)
            actions.append(f"Dealer hits: {hit_card}")

        p_total, _ = hand_value(player)
        d_total, _ = hand_value(dealer)
        result = compare_hands(player, dealer)

        print(f"Final player hand: {fmt_cards(player)} -> {p_total}")
        print(f"Final dealer hand: {fmt_cards(dealer)} -> {d_total}")
        print(f"Round result: {result}")

        tracker.settle_bankroll(result, bet)
        tracker.trace_row(
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
        tracker.status()

        if step_mode:
            input("Press Enter for next round...")

    print("\n========== SHOE COMPLETE ==========")
    tracker.status()
    print(f"Final profit/loss: ${tracker.bankroll - tracker.starting_bankroll:.2f}")
    print(f"Highest bankroll reached: ${tracker.peak_bankroll:.2f}")
    print(f"Lowest bankroll reached: ${tracker.lowest_bankroll:.2f}")
    print(f"Peak profit: ${tracker.peak_bankroll - tracker.starting_bankroll:.2f}")
    print(f"Max drawdown from peak: ${tracker.peak_bankroll - tracker.lowest_bankroll:.2f}")

    export_trace_csv(tracker)


if __name__ == "__main__":
    simulate_one_shoe_trace()