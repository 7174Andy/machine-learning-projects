from connect4_gym.env import Connect4Env
from agents.minimax_agent import MiniMaxAgent
from agents.mcts_agent import MCTS
import numpy as np
import time

env = Connect4Env()
env.reset()

# Instantiate the agents
agent1 = MCTS(player_id=1)
agent2 = MiniMaxAgent(player_id=-1)

agent1_times = []
agent2_times = []

env.render()
while env.winner is None:
    if env.current_player == agent1.player_id:
        start_time = time.time()
        move = agent1.get_action(env)
        end_time = time.time()
        agent1_times.append(end_time - start_time)
        print(f"Agent 1 (MCTS) took {end_time - start_time:.4f} seconds to move.")
    else:
        start_time = time.time()
        move = agent2.get_action(env)
        end_time = time.time()
        agent2_times.append(end_time - start_time)
        print(f"Agent 2 (MiniMax) took {end_time - start_time:.4f} seconds to move.")
        
    _, _, done, _ = env.step(move)
    env.render()

print("Game Over!")
if env.winner == 1:
    print("Agent 1 (MCTS) wins!")
elif env.winner == -1:
    print("Agent 2 (MiniMax) wins!")
else:
    print("It's a draw!")

print("\n--- Decision Time Statistics ---")
print(f"Agent 1 (MCTS):")
print(f"  Average time per move: {np.mean(agent1_times):.4f} seconds")
print(f"  Total time: {np.sum(agent1_times):.4f} seconds")
print(f"  Max time for a single move: {np.max(agent1_times):.4f} seconds")
print(f"  Min time for a single move: {np.min(agent1_times):.4f} seconds")

print(f"\nAgent 2 (MiniMax):")
print(f"  Average time per move: {np.mean(agent2_times):.4f} seconds")
print(f"  Total time: {np.sum(agent2_times):.4f} seconds")
print(f"  Max time for a single move: {np.max(agent2_times):.4f} seconds")
print(f"  Min time for a single move: {np.min(agent2_times):.4f} seconds")