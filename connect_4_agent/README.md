# Connect 4 AI Agents

This project implements and compares various AI agents for playing the game of Connect 4. It includes a customizable game environment and several agent implementations.

## Explanation

The project is built around a `Connect4Env` environment that follows the `gymnasium` API, making it suitable for reinforcement learning experiments.

### Agents

Three different agents are implemented:

1.  **Random Agent (`agents/random_agent.py`):** A baseline agent that selects a valid move randomly.
2.  **Minimax Agent (`agents/minimax_agent.py`):** A classic game theory agent that uses the Minimax algorithm with alpha-beta pruning to find the optimal move. It evaluates board positions based on a heuristic scoring function.
3.  **MCTS Agent (`agents/mcts_agent.py`):** An agent based on Monte Carlo Tree Search (MCTS), a probabilistic and heuristic search algorithm that is effective in many games.

### Scripts

- **`connect4_against_ai.py`:** Play a game of Connect 4 in your terminal against one of the AI agents.
- **`compare_agents.py`:** Watch two AI agents play against each other and see statistics on their decision-making time.
- **`connect4gym.py`:** Run a large number of games between agents to gather performance metrics. It logs results and generates plots, such as the chosen Q-value for the MCTS agent over time.

## Tech Stack

- **Python**
- **NumPy:** For efficient board representation and calculations.
- **Gymnasium:** For the game environment structure.
- **Pandas & Matplotlib:** For logging and plotting performance metrics.
- **Colorama:** For colored output in the terminal.

## Installation Guide

1.  Install the required libraries using pip:
    ```bash
    pip install numpy gymnasium pandas matplotlib colorama
    ```

## How to Run

You can run the different scripts from the `connect_4_agent` directory.

- **Play against the AI:**
  ```bash
  python connect4_against_ai.py
  ```

- **Compare two AI agents:**
  ```bash
  python compare_agents.py
  ```

- **Run a full match simulation and generate metrics:**
  ```bash
  python connect4gym.py
  ```
