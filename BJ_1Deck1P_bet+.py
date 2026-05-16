from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import erf, sqrt

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
FULL_DECK = Counter({r: 4 for r in RANKS})

HILO_VALUES = {
    "2": 1, "3": 1, "4": 1, "5": 1, "6": 1,
    "7": 0, "8": 0, "9": 0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
}

BLACKJACK_PAYOUT = 1.5  # 3:2
MIN_CARDS_TO_START_NEW_ROUND = 4  # player, dealer upcard, player, dealer hole
BASE_PLAYER_EDGE = -0.005   # approximate neutral edge
EDGE_PER_TRUE_COUNT = 0.005  # rough rule of thumb
HAND_VARIANCE_UNITS = 1.30   # rough blackjack variance in betting units
KELLY_FRACTION = 0.25       # quarter Kelly for safer betting advice


def normalize_card(text: str) -> str:
    t = text.strip().upper()
    if t in {"T", "TEN"}:
        t = "10"
    if t not in RANKS:
        raise ValueError(f"Invalid card: {text}")
    return t


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


@dataclass
class RoundSnapshot:
    round_no: int
    bankroll: float
    profit_loss: float
    bet: float
    running_count: int
    true_count: float
    decks_left: float
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
        if self.cards_left() <= 0:
            raise ValueError("No cards left in the deck.")
        self.unseen_cards += 1

    def running_count(self) -> int:
        return sum(HILO_VALUES[c] for c in self.seen)

    def cards_left(self) -> int:
        return sum(self.remaining.values()) - self.unseen_cards

    def decks_left(self) -> float:
        return max(self.cards_left() / 52.0, 0.01)

    def true_count(self) -> float:
        return self.running_count() / self.decks_left()

    def estimated_player_edge(self) -> float:
        return BASE_PLAYER_EDGE + EDGE_PER_TRUE_COUNT * self.true_count()
    
    def recommended_bet(self, table_min: float, table_max: float) -> float:
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

        suggested_bet = table_min * multiplier

        return min(table_max, suggested_bet)
    # def recommended_bet(self, table_min: float, table_max: float) -> float:
    #     tc = self.true_count()

    #     if tc <= 0:
    #         suggested_bet = table_min
    #     elif tc < 1:
    #         suggested_bet = table_min * 2
    #     elif tc < 2:
    #         suggested_bet = table_min * 4
    #     elif tc < 3:
    #         suggested_bet = table_min * 6
    #     else:
    #         suggested_bet = table_min * 8

    #     return min(table_max, suggested_bet)
        

    def estimated_next_hand_win_prob(self) -> float:
        tc = self.true_count()
        win = 0.42 + 0.015 * tc
        return min(max(win, 0.30), 0.60)

    def estimated_next_hand_push_prob(self) -> float:
        tc = self.true_count()
        push = 0.08 - 0.002 * abs(tc)
        return min(max(push, 0.04), 0.12)

    def estimated_next_hand_loss_prob(self) -> float:
        return 1.0 - self.estimated_next_hand_win_prob() - self.estimated_next_hand_push_prob()

    def expected_value(self, bet: float) -> float:
        return bet * self.estimated_player_edge()

    def probability_profit_next_hand(self, bet: float) -> float:
        ev = self.expected_value(bet)
        sd = bet * sqrt(HAND_VARIANCE_UNITS)
        if sd <= 0:
            return 0.5
        return normal_cdf(ev / sd)

    def record_snapshot(self, bet: float, outcome: str, notes: str = "") -> None:
        profit_loss = self.bankroll - self.starting_bankroll
        self.history.append(
            RoundSnapshot(
                round_no=self.round_no,
                bankroll=self.bankroll,
                profit_loss=profit_loss,
                bet=bet,
                running_count=self.running_count(),
                true_count=self.true_count(),
                decks_left=self.decks_left(),
                seen_cards=len(self.seen),
                physical_cards_left=self.cards_left(),
                outcome=outcome,
                notes=notes,
            )
        )

    def status(self) -> None:
        profit = self.bankroll - self.starting_bankroll
        print("--- STATUS ---")
        print(f"Bankroll: ${self.bankroll:.2f}")
        print(f"Profit/Loss: ${profit:.2f}")
        print(f"Seen cards: {len(self.seen)}/52")
        print(f"Physical cards left in shoe: {self.cards_left()}")
        print(f"Unseen dealt cards: {self.unseen_cards}")
        print(f"Running count: {self.running_count()}")
        print(f"Decks left: {self.decks_left():.2f}")
        print(f"True count: {self.true_count():.2f}")
        print("--------------")

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

def prompt_hidden_card(label: str, tracker: BlackjackTracker) -> str:
    while True:
        try:
            c = normalize_card(input(label).strip())

            if tracker.remaining[c] <= 0:
                raise ValueError(f"No {c} cards left in the deck.")

            # Remove physically from shoe
            tracker.remaining[c] -= 1

            # DO NOT add to seen count yet
            return c

        except ValueError as e:
            print(f"Error: {e}")


def reveal_hidden_card(tracker: BlackjackTracker, card: str) -> None:
    # Now the card becomes visible to the player/count
    tracker.seen.append(card)
    
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


def play_round(tracker: BlackjackTracker, table_min: float, table_max: float, hit_soft_17: bool) -> None:
    
    tracker.round_no += 1
    bet = tracker.recommended_bet(table_min, table_max)

    print(f"New round #{tracker.round_no}")
    print(f"Recommended bet before cards are entered: ${bet:.2f}")
    print("Enter cards in the order they appear.")

    player: list[str] = []
    dealer: list[str] = []
    dealer_hole = None

    p1 = prompt_card("Player first card: ", tracker)
    d_up = prompt_card("Dealer upcard: ", tracker)
    p2 = prompt_card("Player second card: ", tracker)

    player.extend([p1, p2])
    dealer.append(d_up)

    if d_up in {"A", "10", "J", "Q", "K"}:
        print("\nDealer checks for blackjack...")

        dealer_hole = prompt_hidden_card("Dealer hole card: ", tracker)
        dealer.append(dealer_hole)

    if is_blackjack(dealer):
        reveal_hidden_card(tracker, dealer_hole)
        print(f"Dealer has blackjack: {dealer}")

        result = compare_hands(player, dealer)

        print(f"Final player hand: {player} -> {hand_value(player)[0]}")
        print(f"Final dealer hand: {dealer} -> {hand_value(dealer)[0]}")
        print(f"Result: {result}")

        settle_bankroll(tracker, result, bet)
        tracker.record_snapshot(bet, result)
        return

    print("Dealer does not have blackjack.")

    print("--- BETTING ADVICE ---")
    edge = tracker.estimated_player_edge()
    win_p = tracker.estimated_next_hand_win_prob()
    push_p = tracker.estimated_next_hand_push_prob()
    loss_p = tracker.estimated_next_hand_loss_prob()
    ev = tracker.expected_value(bet)
    profit_prob = tracker.probability_profit_next_hand(bet)
    print(f"True count: {tracker.true_count():.2f}")
    print(f"Estimated player edge: {edge * 100:.2f}%")
    print(f"Recommended bet: ${bet:.2f}")
    print(f"Estimated EV on bet: ${ev:.2f}")
    print(f"Estimated win/push/loss: {win_p * 100:.1f}% / {push_p * 100:.1f}% / {loss_p * 100:.1f}%")
    print(f"Estimated chance of profit on next hand: {profit_prob * 100:.1f}%")
    if edge > 0:
        print("Advice: this is a spot where pressing the bet is mathematically reasonable.")
    else:
        print("Advice: keep the bet at the table minimum; the shoe is not favorable yet.")
    print("----------------------")

    if is_blackjack(player):
        print(f"Player hand: {player} -> BLACKJACK")
        print(f"Dealer upcard: {d_up}")
        print("Suggested action: STAY")

        result = compare_hands(player, dealer)
        print(f"Dealer hand: {dealer} -> {hand_value(dealer)[0]}")
        print(f"Result: {result}")
        settle_bankroll(tracker, result, bet)
        tracker.record_snapshot(bet, result)
        return

    while True:
        p_total, p_soft = hand_value(player)
        print(f"Player hand: {player} -> {p_total}{' soft' if p_soft else ''}")
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
            tracker.record_snapshot(bet, result, notes="Dealer hole card not seen")
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

    if dealer_hole is None:
        dealer_hole = prompt_card("Dealer hole card: ", tracker)
        dealer.append(dealer_hole)

    reveal_hidden_card(tracker, dealer_hole)

    while dealer_should_hit(dealer, hit_soft_17):
        print(f"Dealer hand so far: {dealer} -> {hand_value(dealer)[0]}")
        hit_card = prompt_card("Dealer hit card: ", tracker)
        dealer.append(hit_card)

    p_total, _ = hand_value(player)
    d_total, _ = hand_value(dealer)
    result = compare_hands(player, dealer)

    print(f"Final player hand: {player} -> {p_total}")
    print(f"Final dealer hand: {dealer} -> {d_total}")
    print(f"Result: {result}")

    settle_bankroll(tracker, result, bet)
    tracker.record_snapshot(bet, result)


def print_progress_table(tracker: BlackjackTracker) -> None:
    if not tracker.history:
        print("No completed rounds to summarize.")
        return

    headers = [
        "Rnd", "Bet", "Bankroll", "P/L", "RunCnt", "TrueCnt", "DecksLft", "Seen", "Left", "Outcome", "Notes"
    ]

    rows = []
    for r in tracker.history:
        rows.append([
            str(r.round_no),
            f"${r.bet:.2f}",
            f"${r.bankroll:.2f}",
            f"${r.profit_loss:.2f}",
            str(r.running_count),
            f"{r.true_count:.2f}",
            f"{r.decks_left:.2f}",
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

    print("Profit/Loss Progression")
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    print("One-Deck Blackjack Simulator with Bet Advice")
    soft17_choice = input("Should dealer HIT on soft 17? (y/n): ").strip().lower()
    hit_soft_17 = soft17_choice == "y"
    starting = float(input("Starting bankroll: $").strip())
    table_min = float(input("Table minimum bet: $").strip())
    table_max = float(input("Table maximum bet: $").strip())

    tracker = BlackjackTracker(bankroll=starting, starting_bankroll=starting)

    while tracker.cards_left() >= MIN_CARDS_TO_START_NEW_ROUND and tracker.bankroll > 0:
        tracker.status()
        play_round(tracker, table_min, table_max, hit_soft_17)

        if tracker.cards_left() < MIN_CARDS_TO_START_NEW_ROUND:
            print("Not enough cards left to start another round.")
            break

        again = input("Play another hand? (y/n): ").strip().lower()
        if again != "y":
            break

    print("Final results")
    tracker.status()
    print_progress_table(tracker)


if __name__ == "__main__":
    main()
