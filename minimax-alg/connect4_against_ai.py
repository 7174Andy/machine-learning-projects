from connect4_gym.env import Connect4Env
from agents.minimax_agent import MiniMaxAgent
import numpy as np

env = Connect4Env()
env.reset()
ai = MiniMaxAgent(player_id=1, max_depth=6)
env.render()
while env.winner is None:
    if env.current_player == ai.player_id:
        move = ai.get_action(env)
    else:
        move = int(input("Enter your move (0-6): "))
    _, _, done, _ = env.step(move)
    env.render()
