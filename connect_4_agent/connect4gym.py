from connect4_gym.env import Connect4Env
from agents.minimax_agent import MiniMaxAgent
from agents.mcts_agent import MCTS
import numpy as np


def run_match(player0, player1):
    env = Connect4Env()
    env.reset()
    while env.winner is None:
        if env.current_player == player0.player_id:
            move = player0.get_action(env)
        else:
            move = player1.get_action(env)
        env.step(move)
        env.render()

    return env.winner


def main():
    wins = draws = losses = 0
    for i in range(10):
        print(f"Game {i + 1}")
        player1 = MiniMaxAgent(player_id=1, max_depth=6)
        player2 = MCTS(player_id=0)
        winner = run_match(player1, player2)
        if winner == 1:
            print("MiniMaxAgent wins!")
            wins += 1
        elif winner == 0:
            print("MCTS wins!")
            losses += 1
        else:
            draws += 1

    print(f"AI - Wins: {wins}, Losses: {losses}, Draws: {draws}")


if __name__ == "__main__":
    main()
