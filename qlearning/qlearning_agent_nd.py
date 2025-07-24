import numpy as np
import random
import pickle
import os
from collections import defaultdict
from typing import Dict, List, Tuple

class QLearningAgentND:
    def __init__(self, player: int, learning_rate: float = 0.1, 
                 gamma: float = 0.95, epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, game_config: dict = None):
        """
        Initialize Q-Learning agent for N-dimensional Tic-Tac-Toe
        
        Args:
            player: Player number (1 or 2)
            learning_rate: Learning rate for Q-value updates
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            game_config: Dictionary with game configuration (dimensions, board_size, etc.)
        """
        self.player = player
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Game configuration
        self.game_config = game_config or {'dimensions': 2, 'board_size': 3, 'win_length': 3}
        self.dimensions = self.game_config['dimensions']
        self.board_size = self.game_config['board_size']
        self.total_cells = self.board_size ** self.dimensions
        
        # Q-table: state -> action -> Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Metadata
        self.name = f"QLearning_{self.dimensions}D_Player_{player}"
        
    def get_action(self, state: str, valid_actions: List[int], training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Get Q-values for valid actions
        q_values = {action: self.q_table[state][action] for action in valid_actions}
        
        # Choose action with highest Q-value
        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]
        
        return random.choice(best_actions)
    
    def update_q_value(self, state: str, action: int, reward: float, next_state: str, done: bool):
        """Update Q-value using Q-learning update rule"""
        if done:
            # Terminal state
            target = reward
        else:
            # Non-terminal state
            next_q_values = [self.q_table[next_state][a] for a in range(self.total_cells)]
            max_next_q = max(next_q_values) if next_q_values else 0
            target = reward + self.gamma * max_next_q
        
        # Q-learning update
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.learning_rate * (target - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_q_table(self, filename: str):
        """Save Q-table to file"""
        data = {
            'q_table': dict(self.q_table),
            'player': self.player,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'game_config': self.game_config,
            'name': self.name
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def load_q_table(self, filename: str):
        """Load Q-table from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: defaultdict(float), data['q_table'])
        self.player = data['player']
        self.learning_rate = data['learning_rate']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.epsilon_min = data['epsilon_min']
        self.epsilon_decay = data['epsilon_decay']
        self.game_config = data['game_config']
        self.name = data['name']
        
        # Update derived attributes
        self.dimensions = self.game_config['dimensions']
        self.board_size = self.game_config['board_size']
        self.total_cells = self.board_size ** self.dimensions
    
    def get_epsilon(self) -> float:
        """Get current epsilon value"""
        return self.epsilon
    
    def get_q_values(self, state: str, valid_actions: List[int]) -> Dict[int, float]:
        """Get Q-values for valid actions"""
        return {action: self.q_table[state][action] for action in valid_actions}

def random_agent(valid_actions: List[int]) -> int:
    """Random agent for comparison"""
    return random.choice(valid_actions) 