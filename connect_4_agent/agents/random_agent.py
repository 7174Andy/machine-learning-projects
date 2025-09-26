from connect4_gym.env import Connect4Env

class RandomAgent:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def get_action(self, env: Connect4Env):
        import random
        valid_actions = env.get_moves()
        return random.choice(valid_actions) if valid_actions else None