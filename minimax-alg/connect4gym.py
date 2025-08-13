from connect4_gym.env import Connect4Env
from agents.minimax_agent import MiniMaxAgent
import numpy as np


def run_match(player0):
    env = Connect4Env()
    env.reset()
    while env.winner is None:
        if env.current_player == player0.player_id:
            move = player0.get_action(env)
        else:
            move = np.random.choice(env.get_moves())
        env.step(move)
        env.render()

    return env.winner


def main():
    wins = draws = losses = 0
    for i in range(100):
        player1 = MiniMaxAgent(player_id=1, max_depth=6)
        winner = run_match(player1)
        if winner == 1:
            wins += 1
        elif winner == 0:
            losses += 1
        else:
            draws += 1

    print(f"AI - Wins: {wins}, Losses: {losses}, Draws: {draws}")


if __name__ == "__main__":
    main()
