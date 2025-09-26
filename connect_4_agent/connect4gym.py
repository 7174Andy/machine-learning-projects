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


def plot_chosen_q_over_episodes(df: pd.DataFrame, window: int, outdir: Path, title_prefix: str):
    sub = df[df["chosen_q"].notnull()].sort_values("episode")
    if sub.empty:
        return
    q = sub["chosen_q"].to_numpy()
    episodes = sub["episode"].to_numpy()
    y = moving_average(q, window) if len(q) >= window else q
    x = episodes[window-1:] if len(q) >= window else episodes
    plt.figure()
    plt.plot(x, y)
    plt.title(f"{title_prefix} chosen Q at root (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Q (win prob)")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(outdir / f"{title_prefix.lower()}_chosenq.png", dpi=150, bbox_inches="tight")
    plt.close()

def main():
    wins = draws = losses = 0
    outdir = Path("plots")
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
    outdir.mkdir(parents=True, exist_ok=True)
    logger.save("episode_metrics.json", str(outdir / "episode_metrics.csv"))

    plot_chosen_q_over_episodes(df, window=10, outdir=outdir, title_prefix="MCTS")

    print("Metrics saved to episode_metrics.json and episode_metrics.csv")



if __name__ == "__main__":
    main()
