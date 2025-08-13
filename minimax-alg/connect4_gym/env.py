from typing import List
import numpy as np
import gymnasium
from gymnasium.spaces import Box, Discrete, Tuple
from colorama import Fore


class Connect4Env(gymnasium.Env):
    """
    GameState for the Connect 4 game.
    The board is represented as a 2D array (rows and columns).
    Each entry on the array can be:
        0 = empty    (.)
        1 = player 1 (X)
        2 = player 2 (O)

    Winner can be:
         None = No winner (yet)
        -1 = Draw
         1 = player 1 (X)
         2 = player 2 (O)
    """

    def __init__(self, width=7, height=6, connect=4):
        self.num_players = 2
        self.width = width
        self.height = height
        self.connect = connect

        # 3: Channels. Empty cells, p1 chips, p2 chips
        player_observation_space = Box(
            low=0,
            high=1,
            shape=(self.height, self.width, self.num_players),
            dtype=np.int32,
        )

        self.observation_space = Tuple(
            player_observation_space for _ in range(self.num_players)
        )

        # Each player has the column to drop
        self.action_space = Tuple(
            [Discrete(self.width) for _ in range(self.num_players)]
        )

        self.state_space_size = (self.height * self.width) ** 3

        self.reset()

    def reset(self) -> List[np.ndarray]:
        """Initialize the game state."""
        self.board = np.full((self.width, self.height), -1)
        self.current_player = 0
        self.winner = None
        return self.get_player_observations()

    def filter_observation_player_perspective(self, player: int) -> List[np.ndarray]:
        """Filter the observation from the perspective of the current player.

        Args:
            player (int): The player ID (1 or 2).

        Returns:
            List[np.ndarray]: The filtered observation for the player.
        """
        opponent = 0 if player == 1 else 1
        empty_positions = np.where(self.board == -1, 1, 0)
        player_positions = np.where(self.board == player, 1, 0)
        opponent_positions = np.where(self.board == opponent, 1, 0)
        return np.array([empty_positions, player_positions, opponent_positions])

    def get_player_observations(self) -> List[np.ndarray]:
        """Get the observations for each player."""
        p1_state = self.filter_observation_player_perspective(0)
        p2_state = np.array(
            [np.copy(p1_state[0]), np.copy(p1_state[-1]), np.copy(p1_state[-2])]
        )
        return [p1_state, p2_state]

    def clone(self):
        """Creates a deep copy of the environment."""
        clone = Connect4Env(self.width, self.height, self.connect)
        clone.board = np.array([self.board[col][:] for col in range(self.width)])
        clone.current_player = self.current_player
        clone.winner = self.winner
        return clone

    def step(self, movecol):
        """Change the game state based on the player's move.

        Args:
            movecol (int): The column to drop the piece into.
        """
        # Checks if the move is valid
        if not (
            movecol >= 0
            and movecol <= self.width
            and self.board[movecol][self.height - 1] == -1
        ):
            raise IndexError(
                f"Invalid move: {movecol}. Tried to place a chip on column {movecol}, but it's full. The valid moves are: {self.get_moves()}"
            )

        # Make the move
        row = self.height - 1
        while row >= 0 and self.board[movecol][row] == -1:
            row -= 1

        row += 1

        self.board[movecol][row] = self.current_player
        self.current_player = 1 - self.current_player
        self.winner, reward_vector = self.check_for_episode_termination(movecol, row)

        # Create info dictionary
        info = {
            "legal_actions": self.get_moves(),
            "current_player": self.current_player,
        }

        return (
            self.get_player_observations(),
            reward_vector,
            self.winner is not None,
            info,
        )

    def get_moves(self):
        """Get the valid moves for the current player."""
        if self.winner is not None:
            return []
        return [
            col for col in range(self.width) if self.board[col][self.height - 1] == -1
        ]

    def check_for_episode_termination(self, movecol, row):
        """Check if the episode has terminated.

        Args:
            movecol (int): The column where the last move was made.
            row (int): The row where the last move was made.

        Returns:
            Tuple[int, List[int]]: The winner and the reward vector.
        """
        winner, reward_vector = self.winner, [0, 0]
        if self.does_move_win(movecol, row):
            winner = 1 - self.current_player
            if winner == 0:
                reward_vector = [1, -1]
            elif winner == 1:
                reward_vector = [-1, 1]
        elif self.get_moves() == []:  # A draw has happened
            winner = -1
            reward_vector = [0, 0]
        return winner, reward_vector

    def does_move_win(self, movecol, row):
        """Checkes whether the newly dropped piece wins the game.

        Args:
            movecol (int): The column where the piece was dropped.
            row (int): The row where the piece was dropped.

        Returns:
            bool: True if the move wins the game, False otherwise.
        """
        me = self.board[movecol][row]  # The player who made the move
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            p = 1
            while (
                self.is_on_board(movecol + p * dx, row + p * dy)
                and self.board[movecol + p * dx][row + p * dy] == me
            ):
                p += 1
            n = 1
            while (
                self.is_on_board(movecol - n * dx, row - n * dy)
                and self.board[movecol - n * dx][row - n * dy] == me
            ):
                n += 1

            if p + n >= (self.connect + 1):
                return True
        return False

    def is_on_board(self, x, y):
        return x >= 0 and x < self.width and y >= 0 and y < self.height

    def get_result(self, player):
        """Get the result of the game for the specified player.

        Args:
            player (int): The player to get the result for (0 or 1).

        Returns:
            int: 1 if the player won, -1 if they lost, 0 if it was a draw.
        """
        if self.winner == player:
            return 1
        elif self.winner == -1:
            return 0
        else:
            return -1

    def render(self, mode="human"):
        """Render the game board.

        Args:
            mode (str): The mode to render the board in ("human" or "ansi").
        """
        if mode == "human":
            s = ""
            for x in range(self.height - 1, -1, -1):
                for y in range(self.width):
                    s += {
                        -1: Fore.WHITE + ".",
                        0: Fore.RED + "X",
                        1: Fore.YELLOW + "O",
                    }[self.board[y][x]]
                    s += Fore.RESET
                s += "\n"
            print(s)
        else:
            raise NotImplementedError("Unknown render mode")
