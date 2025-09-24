import numpy as np
from connect4_gym.env import Connect4Env


class MiniMaxAgent:
    """
    A simple Minimax agent for playing Connect 4.
    """

    def __init__(self, player_id: int = 0, max_depth: int = 6):
        self.player_id = player_id
        self.max_depth = max_depth

    def get_action(self, env: Connect4Env) -> int:
        legal = env.get_moves()
        assert legal, "No legal moves available"
        best_value = -np.inf
        best_move = None
        alpha = -np.inf
        beta = np.inf

        for col in self._order_center_first(legal, env.width):
            child = env.clone()
            child.step(col)
            value = self.minimax(child, self.max_depth - 1, -np.inf, np.inf)
            if value > best_value:
                best_value = value
                best_move = col
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break

        return best_move

    def minimax(self, env: Connect4Env, depth: int, alpha: float, beta: float) -> float:
        # Check if the game is over
        if env.winner is not None:
            res = env.get_result(self.player_id)  # +1 / 0 / -1 from my POV
            if res > 0:
                return 1e6
            if res < 0:
                return -1e6
            return 0.0

        # Depth cut-off
        if depth == 0:
            return self._score_position_env(env, self.player_id)

        # Find legal moves
        legal = env.get_moves()
        if not legal:
            return float(self._score_position_env(env, self.player_id))

        # Check whose turn it is
        turn_is_max = env.current_player == self.player_id

        # If it's the maximizing player's turn
        if turn_is_max:
            value = -np.inf
            for col in legal:
                child = env.clone()
                child.step(col)
                value = max(value, self.minimax(child, depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = np.inf
            for col in legal:
                child = env.clone()
                child.step(col)
                value = min(value, self.minimax(child, depth - 1, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    @staticmethod
    def _evaluate_window(window, piece: int, empty_val: int, connect: int) -> int:
        """
        window: iterable of length 'connect' with values in {empty_val, 0, 1}
        piece:  player id we're evaluating for (0 or 1)
        """
        opp_piece = 1 - piece
        w = list(int(x) for x in window)

        score = 0
        # NOTE: keep weights exactly as you specified
        if w.count(piece) == 4:
            score += 100
        elif w.count(piece) == 3 and w.count(empty_val) == 1:
            score += 5
        elif w.count(piece) == 2 and w.count(empty_val) == 2:
            score += 2

        if w.count(opp_piece) == 3 and w.count(empty_val) == 1:
            score -= 4

        return score

    def _score_position_env(self, env: Connect4Env, piece: int) -> int:
        """
        Score the env position from the perspective of 'piece' (0 or 1).
        Uses env.board shape (width, height) with empty == -1.
        """
        EMPTY = -1
        connect = getattr(env, "connect", 4)
        board_cols = env.board  # (width, height), column-major
        w, h = env.width, env.height
        B = board_cols.T  # (height, width) for row-wise slicing

        score = 0

        # 1) Center column bias
        center_col = w // 2
        center_array = [int(i) for i in list(board_cols[center_col, :])]
        score += center_array.count(piece) * 3

        # 2) Horizontal windows
        for r in range(h):
            row_array = [int(i) for i in list(B[r, :])]
            for c in range(w - connect + 1):
                window = row_array[c : c + connect]
                score += self._evaluate_window(window, piece, EMPTY, connect)

        # 3) Vertical windows
        for c in range(w):
            col_array = [int(i) for i in list(B[:, c])]
            for r in range(h - connect + 1):
                window = col_array[r : r + connect]
                score += self._evaluate_window(window, piece, EMPTY, connect)

        # 4) Positive diagonals (\)
        for r in range(h - connect + 1):
            for c in range(w - connect + 1):
                window = [B[r + i, c + i] for i in range(connect)]
                score += self._evaluate_window(window, piece, EMPTY, connect)

        # 5) Negative diagonals (/)
        for r in range(connect - 1, h):
            for c in range(w - connect + 1):
                window = [B[r - i, c + i] for i in range(connect)]
                score += self._evaluate_window(window, piece, EMPTY, connect)

        return score

    @staticmethod
    def _order_center_first(moves, width):
        c = width // 2
        return sorted(moves, key=lambda m: abs(m - c))
