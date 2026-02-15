from dataclasses import dataclass, asdict
from connect4_gym.env import Connect4Env
from agents.minimax_agent import MiniMaxAgent
from agents.mcts_agent import MCTS
import pandas as pd
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

@dataclass
class EpisodeRecord:
    episode: int
    winner: int            # 0, 1, or -1 for draw
    first_player: int      # who moved first in env (0 or 1)
    agent_name: str        # which agent we’re tracking (here: "MCTS" vs "MiniMax")
    chosen_q: float | None = None   # MCTS root Q for chosen action (optional)

class MetricsLogger:
    def __init__(self):
        self.rows = []

    def log(self, rec: EpisodeRecord):
        self.rows.append(asdict(rec))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, json_path: str, csv_path: str):
        with open(json_path, "w") as f:
            json.dump(self.rows, f, indent=2)
        self.to_frame().to_csv(csv_path, index=False)


def extract_mcts_chosen_q(mcts: MCTS, chosen_action: int) -> float | None:
    """
    If your MCTS class has a helper like root_q_stats() -> [(action, Q, N)], use it.
    Otherwise, compute Q from the child’s wins/visits at the root.
    """
    for a, q, n in mcts.root_q_stats():
        if a == chosen_action and n > 0:
            return float(q)

def run_match(player0, player1):
    env = Connect4Env()
    env.reset()
    while env.winner is None:
        if env.current_player == player0.player_id:
            move = player0.get_action(env)
        else:
            move = player1.get_action(env)
            mcts_first_q = extract_mcts_chosen_q(player1, move)
        env.step(move)
        env.render()

    return env.winner, mcts_first_q

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or window > len(x):
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[window:] - c[:-window]) / float(window)


MCTS_COLOR = "#2563eb"
MINIMAX_COLOR = "#dc2626"
DRAW_COLOR = "#6b7280"
BG_COLOR = "#fafafa"

def _apply_style(ax: plt.Axes):
    ax.set_facecolor(BG_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)

def plot_chosen_q_over_episodes(df: pd.DataFrame, window: int, outdir: Path, title_prefix: str):
    sub = df[df["chosen_q"].notnull()].sort_values("episode")
    if sub.empty:
        return
    q = sub["chosen_q"].to_numpy()
    episodes = sub["episode"].to_numpy()
    y = moving_average(q, window) if len(q) >= window else q
    x = episodes[window-1:] if len(q) >= window else episodes
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG_COLOR)
    _apply_style(ax)
    ax.plot(x, y, color=MCTS_COLOR, linewidth=2)
    ax.fill_between(x, y, alpha=0.12, color=MCTS_COLOR)
    ax.set_title(f"{title_prefix} Root Q-Value (rolling avg, window={window})", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Q (win probability)", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"{title_prefix.lower()}_chosenq.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_outcome_distribution(df: pd.DataFrame, outdir: Path):
    mcts_wins = (df["winner"] == 0).sum()
    minimax_wins = (df["winner"] == 1).sum()
    draws = (df["winner"] == -1).sum()
    total = len(df)

    labels = ["MCTS", "MiniMax", "Draw"]
    counts = [mcts_wins, minimax_wins, draws]
    colors = [MCTS_COLOR, MINIMAX_COLOR, DRAW_COLOR]

    # Filter out zero categories
    labels, counts, colors = zip(*[(l, c, col) for l, c, col in zip(labels, counts, colors) if c > 0])

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=BG_COLOR)
    _apply_style(ax)
    bars = ax.barh(labels, counts, color=colors, height=0.5, edgecolor="white", linewidth=1.5)

    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{count}  ({pct:.0f}%)", va="center", fontsize=12, fontweight="bold")

    ax.set_xlim(0, max(counts) * 1.25)
    ax.set_title(f"MCTS vs MiniMax  —  {total} Games", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Wins", fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "outcome_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_win_rate(df: pd.DataFrame, outdir: Path, window: int = 10):
    df_sorted = df.sort_values("episode")
    mcts_win = (df_sorted["winner"] == 0).astype(float).to_numpy()

    cum_rate = np.cumsum(mcts_win) / np.arange(1, len(mcts_win) + 1)

    rolling_rate = moving_average(mcts_win, window)
    rolling_x = np.arange(window, len(mcts_win) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG_COLOR)
    _apply_style(ax)

    ax.plot(np.arange(1, len(cum_rate) + 1), cum_rate,
            color=MCTS_COLOR, linewidth=2, label="Cumulative")
    ax.plot(rolling_x, rolling_rate,
            color=MINIMAX_COLOR, linewidth=1.5, alpha=0.7, linestyle="--",
            label=f"Rolling (window={window})")
    ax.axhline(0.5, color=DRAW_COLOR, linestyle=":", linewidth=1, alpha=0.6, label="50%")

    ax.set_title("MCTS Win Rate Over Episodes", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("MCTS Win Rate", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "cumulative_win_rate.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_q_by_outcome(df: pd.DataFrame, outdir: Path):
    sub = df[df["chosen_q"].notnull()].copy()
    if sub.empty:
        return
    sub["outcome"] = sub["winner"].map({0: "MCTS Win", 1: "MiniMax Win", -1: "Draw"})

    groups = sub.groupby("outcome")["chosen_q"]
    labels, data, colors = [], [], []
    for outcome, color in [("MCTS Win", MCTS_COLOR), ("MiniMax Win", MINIMAX_COLOR), ("Draw", DRAW_COLOR)]:
        if outcome in groups.groups:
            labels.append(outcome)
            data.append(groups.get_group(outcome).to_numpy())
            colors.append(color)

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=BG_COLOR)
    _apply_style(ax)

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.4, s=18, color="black", zorder=3)

    ax.set_title("MCTS Root Q-Value by Game Outcome", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Q (win probability)", fontsize=12)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "q_by_outcome.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def generate_all_plots(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    plot_chosen_q_over_episodes(df, window=10, outdir=outdir, title_prefix="MCTS")
    plot_outcome_distribution(df, outdir=outdir)
    plot_cumulative_win_rate(df, outdir=outdir, window=10)
    plot_q_by_outcome(df, outdir=outdir)
    print(f"Plots saved to {outdir}/")


def main():
    import sys
    outdir = Path("plots")

    if "--plot-only" in sys.argv:
        with open("episode_metrics.json") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        generate_all_plots(df, outdir)
        return

    wins = draws = losses = 0
    logger = MetricsLogger()
    for i in range(100):
        print(f"Game {i + 1}")
        player1 = MiniMaxAgent(player_id=1, max_depth=6)
        player2 = MCTS(player_id=0)
        winner, q0 = run_match(player1, player2)
        if winner == 1:
            print("MiniMaxAgent wins!")
            wins += 1
        elif winner == 0:
            print("MCTS wins!")
            losses += 1
        else:
            draws += 1

        logger.log(EpisodeRecord(
            episode=i,
            winner=winner,
            first_player=player2.player_id,  # MCTS is always player 0 here
            agent_name="MCTS",
            chosen_q=q0
        ))

    print(f"AI - Wins: {wins}, Losses: {losses}, Draws: {draws}")

    # Save metrics & make plots
    df = logger.to_frame()
    logger.save("episode_metrics.json", str(outdir / "episode_metrics.csv"))
    generate_all_plots(df, outdir)
    print("Metrics saved to episode_metrics.json and episode_metrics.csv")



if __name__ == "__main__":
    main()
