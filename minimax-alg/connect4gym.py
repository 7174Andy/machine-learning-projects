from connect4_gym.env import Connect4Env
from agents.minimax_agent import MiniMaxAgent
import numpy as np

env = Connect4Env()
env.reset()
ai = MiniMaxAgent(player_id=0, max_depth=6)
env.render()
while env.winner is None:
    if env.current_player == ai.player_id:
        move = ai.get_action(env)
    else:
        move = np.random.choice(env.get_moves())  # or your second agent
    _, _, done, _ = env.step(move)
    env.render()
