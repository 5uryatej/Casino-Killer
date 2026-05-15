from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
FULL_DECK = Counter({r: 4 for r in RANKS})

HILO_VALUES = {
    "2": 1, "3": 1, "4": 1, "5": 1, "6": 1,
    "7": 0, "8": 0, "9": 0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
}

BLACKJACK_PAYOUT = 1.5  # 3:2
MIN_CARDS_TO_START_NEW_ROUND = 4  # player, dealer upcard, player, dealer hole


def normalize_card(text: str) -> str:
    t = text.strip().upper()
    if t in {"T", "TEN"}:
        t = "10"
    if t not in RANKS:
        raise ValueError(f"Invalid card: {text}")
    return t


def parse_cards(text: str) -> list[str]:
    parts = text.replace(",", " ").split()
    return [normalize_card(p) for p in parts]


def card_value(rank: str) -> int:
    if rank == "A":
        return 11
    if rank in {"J", "Q", "K"}:
        return 10
    return int(rank)


def hand_value(cards: list[str]) -> tuple[int, bool]:
    total = sum(card_value(c) for c in cards)
    aces = cards.count("A")

    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    soft = "A" in cards and total <= 21 and any(card_value(c) == 11 for c in cards)
    return total, soft


def is_blackjack(cards: list[str]) -> bool:
    total, _ = hand_value(cards)
    return len(cards) == 2 and total == 21


def dealer_should_hit(cards: list[str]) -> bool:
    total, soft = hand_value(cards)
    return total < 17 or (total == 17 and soft)  # hit soft 17


def recommend_action(player_cards: list[str], dealer_upcard: str) -> str:
    """Basic strategy for hit/stay only. No doubles, no splits."""
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


@dataclass
class RoundSnapshot:
    round_no: int
    bankroll: float
    profit_loss: float
    running_count: int
    true_count: float
    seen_cards: int
    physical_cards_left: int
    outcome: str
    notes: str


@dataclass
class BlackjackTracker:
    bankroll: float
    starting_bankroll: float
    seen: list[str] = field(default_factory=list)
    unseen_cards: int = 0
    remaining: Counter = field(default_factory=lambda: Counter(FULL_DECK))
    history: list[RoundSnapshot] = field(default_factory=list)
    round_no: int = 0

    def add_seen_card(self, card: str) -> None:
        if self.remaining[card] <= 0:
            raise ValueError(f"No {card} cards left in the deck.")
        self.remaining[card] -= 1
        self.seen.append(card)

    def add_unseen_card(self) -> None:
        """A physical card was dealt, but its rank was not revealed to the user."""
        if self.cards_left() <= 0:
            raise ValueError("No cards left in the deck.")
        self.unseen_cards += 1

    def running_count(self) -> int:
        return sum(HILO_VALUES[c] for c in self.seen)

    def cards_left(self) -> int:
        return sum(self.remaining.values()) - self.unseen_cards

    def total_dealt_cards(self) -> int:
        return len(self.seen) + self.unseen_cards

    def true_count(self) -> float:
        cards_left = max(self.cards_left(), 1)
        decks_left = max(cards_left / 52.0, 0.01)
        return self.running_count() / decks_left

    def record_snapshot(self, outcome: str, notes: str = "") -> None:
        profit_loss = self.bankroll - self.starting_bankroll
        self.history.append(
            RoundSnapshot(
                round_no=self.round_no,
                bankroll=self.bankroll,
                profit_loss=profit_loss,
                running_count=self.running_count(),
                true_count=self.true_count(),
                seen_cards=len(self.seen),
                physical_cards_left=self.cards_left(),
                outcome=outcome,
                notes=notes,
            )
        )

    def status(self) -> None:
        profit = self.bankroll - self.starting_bankroll
        print("\n--- STATUS ---")
        print(f"Bankroll: ${self.bankroll:.2f}")
        print(f"Profit/Loss: ${profit:.2f}")
        print(f"Seen cards: {len(self.seen)}/52")
        print(f"Physical cards left in shoe: {self.cards_left()}")
        print(f"Unseen dealt cards: {self.unseen_cards}")
        print(f"Running count: {self.running_count()}")
        print(f"True count: {self.true_count():.2f}")
        print("--------------\n")

    def last_playable_hand_message(self) -> str:
        if self.cards_left() < MIN_CARDS_TO_START_NEW_ROUND:
            return "Deck is below the minimum needed to start another round."
        if self.cards_left() == MIN_CARDS_TO_START_NEW_ROUND:
            return "This is likely the last playable round with a fresh deal." 
        return ""


def prompt_card(label: str, tracker: BlackjackTracker) -> str:
    while True:
        try:
            c = normalize_card(input(label).strip())
            tracker.add_seen_card(c)
            return c
        except ValueError as e:
            print(f"Error: {e}")


def settle_bankroll(tracker: BlackjackTracker, result: str, bet: float) -> None:
    if result == "PLAYER WIN":
        tracker.bankroll += bet
    elif result == "DEALER WIN":
        tracker.bankroll -= bet
    elif result == "PLAYER BLACKJACK":
        tracker.bankroll += bet * BLACKJACK_PAYOUT
    elif result == "DEALER BLACKJACK":
        tracker.bankroll -= bet
    elif result == "DEALER BUST":
        tracker.bankroll += bet
    elif result == "PLAYER BUST":
        tracker.bankroll -= bet
    # PUSH = no change


def play_round(tracker: BlackjackTracker, bet: float) -> None:
    tracker.round_no += 1
    print(f"\nNew round #{tracker.round_no}")
    print("Enter cards in the order they appear.")

    player: list[str] = []
    dealer: list[str] = []

    p1 = prompt_card("Player first card: ", tracker)
    d_up = prompt_card("Dealer upcard: ", tracker)
    p2 = prompt_card("Player second card: ", tracker)

    player.extend([p1, p2])
    dealer.append(d_up)

    # Immediate player blackjack.
    if is_blackjack(player):
        print(f"\nPlayer hand: {player} -> BLACKJACK")
        print(f"Dealer upcard: {d_up}")
        print("Suggested action: STAY")

        dealer_hole = prompt_card("Dealer hole card: ", tracker)
        dealer.append(dealer_hole)

        result = compare_hands(player, dealer)
        print(f"Dealer hand: {dealer} -> {hand_value(dealer)[0]}")
        print(f"Result: {result}")
        settle_bankroll(tracker, result, bet)
        tracker.record_snapshot(result)
        return

    while True:
        p_total, p_soft = hand_value(player)
        print(f"\nPlayer hand: {player} -> {p_total}{' soft' if p_soft else ''}")
        print(f"Dealer upcard: {d_up}")
        print(f"Count: running {tracker.running_count()}, true {tracker.true_count():.2f}")

        if p_total > 21:
            print("Player busts.")
            print("Dealer hole card was dealt but not revealed.")
            tracker.add_unseen_card()
            print(f"One card was removed from the shoe unseen. Cards left: {tracker.cards_left()}")
            print(tracker.last_playable_hand_message())

            result = "PLAYER BUST"
            settle_bankroll(tracker, result, bet)
            tracker.record_snapshot(result, notes="Dealer hole card not seen")
            return

        action = recommend_action(player, d_up)
        print(f"Suggested action: {action}")

        move = input("Type H to hit, S to stay, or Enter to follow suggestion: ").strip().upper()
        if move == "":
            move = "H" if action == "HIT" else "S"

        if move == "H":
            new_card = prompt_card("Player hit card: ", tracker)
            player.append(new_card)
            continue

        if move == "S":
            break

        print("Please enter H or S.")

    dealer_hole = prompt_card("Dealer hole card: ", tracker)
    dealer.append(dealer_hole)

    while dealer_should_hit(dealer):
        print(f"Dealer hand so far: {dealer} -> {hand_value(dealer)[0]}")
        hit_card = prompt_card("Dealer hit card: ", tracker)
        dealer.append(hit_card)

    p_total, _ = hand_value(player)
    d_total, _ = hand_value(dealer)
    result = compare_hands(player, dealer)

    print(f"\nFinal player hand: {player} -> {p_total}")
    print(f"Final dealer hand: {dealer} -> {d_total}")
    print(f"Result: {result}")

    settle_bankroll(tracker, result, bet)
    tracker.record_snapshot(result)


def print_progress_table(tracker: BlackjackTracker) -> None:
    if not tracker.history:
        print("\nNo completed rounds to summarize.")
        return

    headers = [
        "Rnd", "Bankroll", "P/L", "RunCnt", "TrueCnt", "Seen", "Left", "Outcome", "Notes"
    ]

    rows = []
    for r in tracker.history:
        rows.append([
            str(r.round_no),
            f"${r.bankroll:.2f}",
            f"${r.profit_loss:.2f}",
            str(r.running_count),
            f"{r.true_count:.2f}",
            str(r.seen_cards),
            str(r.physical_cards_left),
            r.outcome,
            r.notes,
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(values: list[str]) -> str:
        return " | ".join(values[i].ljust(widths[i]) for i in range(len(values)))

    print("\nProfit/Loss Progression")
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    print("One-Deck Blackjack Simulator")
    starting = float(input("Starting bankroll: $").strip())
    bet = float(input("Flat bet per hand: $").strip())

    tracker = BlackjackTracker(bankroll=starting, starting_bankroll=starting)

    while tracker.cards_left() >= MIN_CARDS_TO_START_NEW_ROUND and tracker.bankroll > 0:
        tracker.status()
        play_round(tracker, bet)

        if tracker.cards_left() < MIN_CARDS_TO_START_NEW_ROUND:
            print("\nNot enough cards left to start another round.")
            break

        again = input("\nPlay another hand? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\nFinal results")
    tracker.status()
    print_progress_table(tracker)


if __name__ == "__main__":
    main()
