import numpy as np
import random
from typing import List, Tuple, Optional, Set
from itertools import product, combinations

class TicTacToeND:
    def __init__(self, dimensions: int = 3, board_size: int = 3, win_length: int = 3):
        """
        Initialize N-dimensional Tic-Tac-Toe
        
        Args:
            dimensions: Number of dimensions (2D, 3D, 4D, etc.)
            board_size: Size of each dimension (3x3, 4x4, etc.)
            win_length: Number of consecutive pieces needed to win
        """
        self.dimensions = dimensions
        self.board_size = board_size
        self.win_length = win_length
        
        # Create N-dimensional board
        self.board = np.zeros([board_size] * dimensions, dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        
        # Pre-compute all possible winning lines
        self.winning_lines = self._generate_winning_lines()
        
        # Total number of cells
        self.total_cells = board_size ** dimensions
    
    def _generate_winning_lines(self) -> List[List[Tuple]]:
        """Generate all possible winning lines in N dimensions"""
        winning_lines = []
        
        # Generate all possible starting positions
        for start_pos in product(range(self.board_size), repeat=self.dimensions):
            # Generate all possible directions (unit vectors in N dimensions)
            for direction in product([-1, 0, 1], repeat=self.dimensions):
                if direction == tuple([0] * self.dimensions):  # Skip zero direction
                    continue
                
                # Check if this direction can form a winning line from this start
                line = []
                pos = list(start_pos)
                
                for i in range(self.win_length):
                    # Check if position is within bounds
                    if any(p < 0 or p >= self.board_size for p in pos):
                        break
                    line.append(tuple(pos))
                    pos = [p + d for p, d in zip(pos, direction)]
                
                # Only add if we have a complete winning line
                if len(line) == self.win_length:
                    winning_lines.append(line)
        
        return winning_lines
    
    def reset(self):
        """Reset the game to initial state"""
        self.board = np.zeros([self.board_size] * self.dimensions, dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        return self.get_state()
    
    def get_state(self) -> str:
        """Get current board state as string"""
        return ''.join(self.board.flatten().astype(str))
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid action indices"""
        return [i for i in range(self.total_cells) if self.board.flatten()[i] == 0]
    
    def _index_to_coordinates(self, index: int) -> Tuple:
        """Convert 1D index to N-dimensional coordinates"""
        coords = []
        remaining = index
        for dim in range(self.dimensions - 1, -1, -1):
            coords.insert(0, remaining // (self.board_size ** dim))
            remaining = remaining % (self.board_size ** dim)
        return tuple(coords)
    
    def _coordinates_to_index(self, coords: Tuple) -> int:
        """Convert N-dimensional coordinates to 1D index"""
        index = 0
        for i, coord in enumerate(coords):
            index += coord * (self.board_size ** (self.dimensions - 1 - i))
        return index
    
    def make_move(self, action: int) -> Tuple[str, int, bool]:
        """Make a move and return (new_state, reward, done)"""
        if self.game_over or action not in self.get_valid_actions():
            return self.get_state(), -10, True
        
        # Convert 1D index to N-dimensional coordinates
        coords = self._index_to_coordinates(action)
        
        # Make the move
        self.board[coords] = self.current_player
        
        # Check for game end
        reward = self._check_game_end()
        
        if not self.game_over:
            self.current_player = 3 - self.current_player
        
        return self.get_state(), reward, self.game_over
    
    def _check_game_end(self) -> int:
        """Check if game has ended and return reward"""
        # Check all winning lines
        for line in self.winning_lines:
            values = [self.board[pos] for pos in line]
            if len(set(values)) == 1 and values[0] != 0:
                self.game_over = True
                self.winner = values[0]
                return 10 if self.winner == 1 else -10
        
        # Check for draw
        if len(self.get_valid_actions()) == 0:
            self.game_over = True
            self.winner = 0
            return 0
        
        return 0
    
    def display(self):
        """Display the board (simplified for N dimensions)"""
        if self.dimensions == 2:
            # 2D display
            symbols = {0: '.', 1: 'X', 2: 'O'}
            for row in self.board:
                print(' '.join([symbols[cell] for cell in row]))
        elif self.dimensions == 3:
            # 3D display - show layers
            symbols = {0: '.', 1: 'X', 2: 'O'}
            for layer in range(self.board_size):
                print(f"Layer {layer}:")
                for row in self.board[layer]:
                    print(' '.join([symbols[cell] for cell in row]))
                print()
        else:
            # N-dimensional display - show flattened view
            print(f"{self.dimensions}D Board ({self.board_size}^{self.dimensions} = {self.total_cells} cells):")
            print(f"Current state: {self.get_state()}")
            print(f"Valid moves: {len(self.get_valid_actions())}")
        
        if self.game_over:
            if self.winner == 0:
                print("Game ended in draw!")
            else:
                print(f"Player {self.winner} wins!")
        else:
            print(f"Current player: {self.current_player}")
        print()
    
    def get_board_info(self) -> dict:
        """Get information about the board configuration"""
        return {
            'dimensions': self.dimensions,
            'board_size': self.board_size,
            'win_length': self.win_length,
            'total_cells': self.total_cells,
            'winning_lines_count': len(self.winning_lines)
        }

# Convenience classes for common configurations
class TicTacToe2D(TicTacToeND):
    """2D Tic-Tac-Toe (3x3)"""
    def __init__(self, board_size: int = 3, win_length: int = 3):
        super().__init__(dimensions=2, board_size=board_size, win_length=win_length)

class TicTacToe3D(TicTacToeND):
    """3D Tic-Tac-Toe (3x3x3)"""
    def __init__(self, board_size: int = 3, win_length: int = 3):
        super().__init__(dimensions=3, board_size=board_size, win_length=win_length)

class TicTacToe4D(TicTacToeND):
    """4D Tic-Tac-Toe (3x3x3x3)"""
    def __init__(self, board_size: int = 3, win_length: int = 3):
        super().__init__(dimensions=4, board_size=board_size, win_length=win_length)

# Backward compatibility
TicTacToe = TicTacToe2D 