import numpy as np
from typing import Tuple, Optional, List
from enum import Enum

class Player(Enum):
    """Enum for players"""
    X = 1
    O = 2

class GameStatus(Enum):
    """Enum for game status"""
    ONGOING = 0
    X_WINS = 1
    O_WINS = 2
    DRAW = 3

class TicTacToeEnv:
    """
    TicTacToe environment for reinforcement learning.
    
    State representation: 3x3 array where 0=empty, 1=X, 2=O
    Action space: 0-8 representing board positions (top-left to bottom-right)
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = Player.X
        self.game_status = GameStatus.ONGOING
        self.move_count = 0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state as a copy of the board"""
        return self.board.copy()
    
    def _is_valid_action(self, action: int) -> bool:
        """Check if action is valid (position is empty)"""
        if not 0 <= action <= 8:
            return False
        row, col = action // 3, action % 3
        return self.board[row, col] == 0
    
    def _get_valid_actions(self) -> List[int]:
        """Get list of valid actions (empty positions)"""
        valid_actions = []
        for i in range(9):
            if self._is_valid_action(i):
                valid_actions.append(i)
        return valid_actions
    
    def _check_winner(self) -> Optional[Player]:
        """Check if there's a winner, return None if no winner"""
        # Check rows
        for row in range(3):
            if self.board[row, 0] != 0 and self.board[row, 0] == self.board[row, 1] == self.board[row, 2]:
                return Player(self.board[row, 0])
        
        # Check columns
        for col in range(3):
            if self.board[0, col] != 0 and self.board[0, col] == self.board[1, col] == self.board[2, col]:
                return Player(self.board[0, col])
        
        # Check diagonals
        if self.board[0, 0] != 0 and self.board[0, 0] == self.board[1, 1] == self.board[2, 2]:
            return Player(self.board[0, 0])
        
        if self.board[0, 2] != 0 and self.board[0, 2] == self.board[1, 1] == self.board[2, 0]:
            return Player(self.board[0, 2])
        
        return None
    
    def _is_draw(self) -> bool:
        """Check if game is a draw (board is full)"""
        return self.move_count == 9
    
    def _update_game_status(self):
        """Update game status based on current board state"""
        winner = self._check_winner()
        if winner:
            self.game_status = GameStatus.X_WINS if winner == Player.X else GameStatus.O_WINS
        elif self._is_draw():
            self.game_status = GameStatus.DRAW
        else:
            self.game_status = GameStatus.ONGOING
    
    def _calculate_reward(self, player: Player) -> float:
        """Calculate reward for the given player"""
        if self.game_status == GameStatus.ONGOING:
            return -0.01  # Small penalty for each move to encourage efficiency
        elif self.game_status == GameStatus.DRAW:
            return 0.0
        elif (self.game_status == GameStatus.X_WINS and player == Player.X) or \
             (self.game_status == GameStatus.O_WINS and player == Player.O):
            return 1.0  # Win
        else:
            return -1.0  # Loss
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment
        
        Args:
            action: Board position (0-8)
            
        Returns:
            (next_state, reward, done, info)
        """
        if not self._is_valid_action(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Make the move
        row, col = action // 3, action % 3
        self.board[row, col] = self.current_player.value
        self.move_count += 1
        
        # Update game status
        self._update_game_status()
        
        # Calculate reward for the player who just moved
        reward = self._calculate_reward(self.current_player)
        
        # Check if game is done
        done = self.game_status != GameStatus.ONGOING
        
        # Switch players if game continues
        if not done:
            self.current_player = Player.O if self.current_player == Player.X else Player.X
        
        # Prepare info dict
        info = {
            'current_player': self.current_player,
            'game_status': self.game_status,
            'move_count': self.move_count,
            'valid_actions': self._get_valid_actions() if not done else []
        }
        
        return self._get_state(), reward, done, info
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions"""
        return self._get_valid_actions()
    
    def is_done(self) -> bool:
        """Check if game is done"""
        return self.game_status != GameStatus.ONGOING
    
    def get_winner(self) -> Optional[Player]:
        """Get the winner if game is done, None otherwise"""
        if self.game_status == GameStatus.X_WINS:
            return Player.X
        elif self.game_status == GameStatus.O_WINS:
            return Player.O
        return None
    
    def render(self):
        """Render the current board state"""
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        
        print("-------------")
        for i in range(3):
            row_str = "|"
            for j in range(3):
                row_str += f" {symbols[self.board[i, j]]} |"
            print(row_str)
            print("-------------")
        
        if self.game_status == GameStatus.ONGOING:
            print(f"Current player: {self.current_player.name}")
        elif self.game_status == GameStatus.DRAW:
            print("Game ended in a draw!")
        else:
            winner = self.get_winner()
            print(f"{winner.name} wins!")
    
    def get_state_hash(self) -> str:
        """Get a hash of the current state for use in Q-learning"""
        return str(self.board.flatten().tolist()) 