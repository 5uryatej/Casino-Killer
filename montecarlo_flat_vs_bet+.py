from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


INPUT_DIR = Path("simulation_output_flat_vs_betramp")
TRACE_FILE = INPUT_DIR / "session_paths.csv"
OUTPUT_FILE = INPUT_DIR / "monte_carlo_path_overlay.png"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_paths(rows: list[dict[str, str]]) -> dict[str, dict[int, list[float]]]:
    """
    Returns:
        {
            "flat": {session_no: [bankroll_1, bankroll_2, ...]},
            "betramp": {session_no: [bankroll_1, bankroll_2, ...]},
        }
    """
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    # Sort so each path is in step order.
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r.get("strategy", ""),
            int(r.get("session_no", "0")),
            int(r.get("step_no", "0")),
        ),
    )

    for row in rows_sorted:
        strategy = row.get("strategy", "").strip().lower()
        session_no = int(row["session_no"])
        bankroll_after = float(row["bankroll_after"])
        grouped[strategy][session_no].append(bankroll_after)

    return grouped


def average_path(paths: list[list[float]]) -> list[float]:
    if not paths:
        return []

    max_len = max(len(p) for p in paths)
    avg: list[float] = []

    for i in range(max_len):
        vals = [p[i] for p in paths if i < len(p)]
        if vals:
            avg.append(mean(vals))
        else:
            avg.append(float("nan"))
    return avg


def plot_strategy_paths(
    ax,
    paths: dict[int, list[float]],
    title: str,
    line_style: str,
    color: str,
    alpha: float = 0.15,
) -> None:
    if not paths:
        ax.set_title(f"{title} (no data)")
        return

    # Individual paths.
    for _, bankroll_path in paths.items():
        x = list(range(1, len(bankroll_path) + 1))
        ax.plot(x, bankroll_path, linestyle=line_style, color=color, alpha=alpha, linewidth=1)

    # Average path.
    avg = average_path(list(paths.values()))
    ax.plot(
        list(range(1, len(avg) + 1)),
        avg,
        linestyle=line_style,
        color=color,
        linewidth=2.5,
        alpha=1.0,
        label=f"{title} average",
    )

    # Starting bankroll line.
    all_starts = [p[0] for p in paths.values() if p]
    if all_starts:
        start_bankroll = mean(all_starts)
        ax.axhline(start_bankroll, linestyle=":", color="black", linewidth=1.2, label="Starting bankroll")

    ax.set_title(title)
    ax.set_xlabel("Round / step")
    ax.set_ylabel("Bankroll")
    ax.grid(True, alpha=0.25)
    ax.legend()


def main() -> None:
    rows = read_rows(TRACE_FILE)
    grouped = group_paths(rows)

    flat_paths = grouped.get("flat", {})
    ramp_paths = grouped.get("betramp", {})

    if not flat_paths and not ramp_paths:
        raise RuntimeError("No bankroll paths found in the trace CSV.")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Light individual paths for flat and bet ramp.
    for _, bankroll_path in flat_paths.items():
        x = list(range(1, len(bankroll_path) + 1))
        ax.plot(x, bankroll_path, color="tab:blue", alpha=0.12, linewidth=1, linestyle="-")

    for _, bankroll_path in ramp_paths.items():
        x = list(range(1, len(bankroll_path) + 1))
        ax.plot(x, bankroll_path, color="tab:orange", alpha=0.12, linewidth=1, linestyle="--")

    # Average paths.
    flat_avg = average_path(list(flat_paths.values()))
    ramp_avg = average_path(list(ramp_paths.values()))

    if flat_avg:
        ax.plot(
            list(range(1, len(flat_avg) + 1)),
            flat_avg,
            color="tab:blue",
            linewidth=3,
            linestyle="-",
            label="Flat bet average",
        )

    if ramp_avg:
        ax.plot(
            list(range(1, len(ramp_avg) + 1)),
            ramp_avg,
            color="tab:orange",
            linewidth=3,
            linestyle="--",
            label="Bet+ average",
        )

    # Starting bankroll reference line.
    all_starts = []
    if flat_paths:
        all_starts.append(next(iter(flat_paths.values()))[0])
    if ramp_paths:
        all_starts.append(next(iter(ramp_paths.values()))[0])
    if all_starts:
        ax.axhline(mean(all_starts), color="black", linestyle=":", linewidth=1.5, label="Starting bankroll")

    ax.set_title("Monte Carlo Bankroll Paths: Flat Bet vs Bet+")
    ax.set_xlabel("Round / step")
    ax.set_ylabel("Bankroll")
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=180)
    plt.close()

    print(f"Saved graph to: {OUTPUT_FILE}")
    print(f"Flat sessions: {len(flat_paths)}")
    print(f"Bet+ sessions: {len(ramp_paths)}")
    if flat_avg:
        print(f"Flat average final bankroll: ${flat_avg[-1]:.2f}")
    if ramp_avg:
        print(f"Bet+ average final bankroll: ${ramp_avg[-1]:.2f}")


if __name__ == "__main__":
    main()
