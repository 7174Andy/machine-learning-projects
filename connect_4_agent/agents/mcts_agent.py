from copy import deepcopy
from connect4_gym.env import Connect4Env
from math import log, sqrt
import random

class Node:
    """
    A node in the MCTS tree. Each node contains:
    - state: the current state of the game. A tuple of (player, grid) where player is WHITE or BLACK, and grid is 2D list of the board.
    - num_wins: the number of wins for the player of the parent node
    - num_visits: the number of visits to this node
    - parent: the parent node of this node
    - children: a list of child nodes
    - untried_actions: a list of actions that have not been tried yet
    - is_terminal: bool indicating if the game is over
    """
    def __init__(self, simulator, actions, parent=None):
        self.simulator: Connect4Env = simulator
        self.state = (self.simulator.current_player, self.simulator.board)
        self.num_wins = 0
        self.num_visits = 0
        self.parent = parent
        self.children = []
        self.untried_actions = deepcopy(actions)
        self.is_terminal = self.simulator.winner is not None

RESOURCES = 1500

class MCTS:
    """
    Monte Carlo Tree Search (MCTS) agent for decision making in games.
    """
    def __init__(self, player_id: int = 0, ):
        self.simulator = Connect4Env()
        self.simulator.reset()
        self.player_id = player_id
        self.root = Node(simulator=self.simulator.clone(), actions=self.simulator.get_moves())

    def get_action(self, env: Connect4Env):
        # The tree search itself
        iters = 0
        self.simulator = env.clone()
        self.root = Node(simulator=self.simulator.clone(), actions=self.simulator.get_moves())
        while iters < RESOURCES:
            node = self.select(self.root)
            result = self.rollout(node)
            self.backpropagate(node, result)
            iters += 1
        
        # Return the action that was most visited
        best_child, best_action, _ = self.best_child(self.root, 0)
        return best_action

    def select(self, node: Node):
        while not node.is_terminal:
            if len(node.untried_actions) > 0:
                return self.expand(node)
            else:
                node, _, _ = self.best_child(node)
        return node

    def expand(self, node: Node):
        child_node = None

        action = node.untried_actions.pop()
        if action is not None:
            new_simulator = node.simulator.clone()
            new_simulator.step(action)
            child_node = Node(simulator=new_simulator, actions=new_simulator.get_moves(), parent=node)
            node.children.append((action, child_node))
        
        return child_node

    def best_child(self, node: Node, c=0.8):
        best_child = None
        best_action = None
        best_child_value = float("-inf")

        for action, child in node.children:
            value = float(child.num_wins / child.num_visits) + c * sqrt(
                2 * log(node.num_visits) / child.num_visits
            )
            if best_child is None or value > best_child_value:
                best_child = child
                best_action = action
                best_child_value = value
            
        return best_child, best_action, best_child_value

    def backpropagate(self, node: Node, result):
        while node is not None:
            node.num_visits += 1
            if node.parent is not None:
                node.num_wins += result[node.parent.simulator.current_player]
            node = node.parent

    def rollout(self, node: Node):
        self.simulator = node.simulator.clone()
        while self.simulator.winner is None:
            possible_moves = self.simulator.get_moves()

            if random.random() < 0.4:
                action = self.tectical_move(self.simulator)
            else:
                action = random.choice(possible_moves) if possible_moves else None
            self.simulator.step(action)
        
        reward = {}
        if self.simulator.winner == -1:
            reward = {0: 0.5, 1: 0.5}
        else:
            reward = {self.simulator.winner: 1, 1 - self.simulator.winner: 0}
        return reward

    def tectical_move(self, sim: Connect4Env):
        possible_moves = sim.get_moves()
        
        # Check if there are any immediate winning moves
        if not possible_moves:
            return None
    
        # play a immediate winning move if available
        for move in possible_moves:
            temp_sim = sim.clone()
            temp_sim.step(move)
            if temp_sim.winner == sim.current_player:
                return move
        
        # Block opponent's immediate winning move if available
        opponent = 1 - sim.current_player
        for move in possible_moves:
            temp_sim = sim.clone()
            temp_sim.step(move)
            if temp_sim.winner == opponent:
                return move
        
        # Otherwise, center bias
        order = sorted(possible_moves, key=lambda c: abs(c - (sim.width - 1)/2))
        return order[0]
    
    def root_q_stats(self):
        """ Returns the Q values of the root node's children for analysis purposes. """
        stats = []
        for action, child in self.root.children:
            q_value = child.num_wins / child.num_visits if child.num_visits > 0 else 0
            stats.append((action, q_value, child.num_visits))
        
        stats.sort(key=lambda x: x[0])  # Sort by action (column)
        return stats
