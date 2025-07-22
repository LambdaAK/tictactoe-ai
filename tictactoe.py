import numpy as np
import random
from typing import List, Tuple, Optional

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
    
    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        return self.get_state()
    
    def get_state(self) -> str:
        return ''.join(self.board.flatten().astype(str))
    
    def get_valid_actions(self) -> List[int]:
        return [i for i in range(9) if self.board.flatten()[i] == 0]
    
    def make_move(self, action: int) -> Tuple[str, int, bool]:
        if self.game_over or action not in self.get_valid_actions():
            return self.get_state(), -10, True
        
        row, col = action // 3, action % 3
        self.board[row][col] = self.current_player
        
        reward = self._check_game_end()
        
        if not self.game_over:
            self.current_player = 3 - self.current_player
        
        return self.get_state(), reward, self.game_over
    
    def _check_game_end(self) -> int:
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                self.game_over = True
                self.winner = self.board[i][0]
                return 10 if self.winner == 1 else -10
            
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != 0:
                self.game_over = True
                self.winner = self.board[0][i]
                return 10 if self.winner == 1 else -10
        
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            self.game_over = True
            self.winner = self.board[0][0]
            return 10 if self.winner == 1 else -10
        
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            self.game_over = True
            self.winner = self.board[0][2]
            return 10 if self.winner == 1 else -10
        
        if len(self.get_valid_actions()) == 0:
            self.game_over = True
            self.winner = 0
            return 0
        
        return 0
    
    def display(self):
        symbols = {0: '.', 1: 'X', 2: 'O'}
        for row in self.board:
            print(' '.join([symbols[cell] for cell in row]))
        print()