import numpy as np
import random
import pickle
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tictactoe import TicTacToe
except ImportError:
    from ..tictactoe import TicTacToe

class QLearningAgent:
    def __init__(self, player: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        self.player = player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        
    def get_action(self, state: str, valid_actions: List[int], training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        q_values = [self.q_table[state][action] for action in valid_actions]
        max_q = max(q_values) if q_values else 0
        
        best_actions = [action for action, q_val in zip(valid_actions, q_values) if q_val == max_q]
        return random.choice(best_actions)
    
    def update_q_value(self, state: str, action: int, reward: float, 
                      next_state: str, next_valid_actions: List[int]):
        current_q = self.q_table[state][action]
        
        if next_valid_actions:
            max_next_q = max([self.q_table[next_state][a] for a in next_valid_actions])
        else:
            max_next_q = 0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon
    
    def decay_epsilon(self):
        """Decay epsilon according to the decay rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_epsilon(self):
        """Reset epsilon to initial value"""
        self.epsilon = self.initial_epsilon
    
    def get_epsilon(self) -> float:
        """Get current epsilon value"""
        return self.epsilon
    
    def save_q_table(self, filename: str):
        q_dict = {state: dict(actions) for state, actions in self.q_table.items()}
        with open(filename, 'wb') as f:
            pickle.dump(q_dict, f)
    
    def load_q_table(self, filename: str):
        try:
            with open(filename, 'rb') as f:
                q_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: defaultdict(float))
                for state, actions in q_dict.items():
                    for action, q_value in actions.items():
                        self.q_table[state][action] = q_value
        except FileNotFoundError:
            print(f"Q-table file {filename} not found. Starting with empty Q-table.")

def random_agent(valid_actions: List[int]) -> int:
    return random.choice(valid_actions)