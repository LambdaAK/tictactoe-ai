import numpy as np
import json
import random
from typing import List, Dict, Any
from base_agent import BaseAgent
from tictactoe_env import TicTacToeEnv, Player

class QLearningAgent(BaseAgent):
    """
    Q-Learning agent for TicTacToe.
    
    Uses a tabular Q-table to store state-action values.
    """
    
    def __init__(self, player: Player, env: TicTacToeEnv, 
                 learning_rate: float = 0.1, 
                 discount_factor: float = 0.9,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent.
        
        Args:
            player: Which player this agent represents
            env: The environment instance
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decays
            epsilon_min: Minimum epsilon value
        """
        super().__init__(player, env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        
    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current game state
            valid_actions: List of valid action indices
            
        Returns:
            Selected action index
        """
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Epsilon-greedy exploration
        if self.training_mode and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Greedy action selection
        best_action = valid_actions[0]
        best_value = self._get_q_value(state, best_action)
        
        for action in valid_actions[1:]:
            q_value = self._get_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        return best_action
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """
        Update Q-values using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        current_q = self._get_q_value(state, action)
        
        if done:
            # Terminal state - no future rewards
            target_q = reward
        else:
            # Non-terminal state - consider future rewards
            next_valid_actions = self.env.get_valid_actions()
            if next_valid_actions:
                # Max Q-value for next state
                max_next_q = max(self._get_q_value(next_state, a) for a in next_valid_actions)
                target_q = reward + self.discount_factor * max_next_q
            else:
                target_q = reward
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self._set_q_value(state, action, new_q)
    
    def _get_q_value(self, state: np.ndarray, action: int) -> float:
        """Get Q-value for state-action pair."""
        key = self.get_action_key(state, action)
        return self.q_table.get(key, 0.0)
    
    def _set_q_value(self, state: np.ndarray, action: int, value: float) -> None:
        """Set Q-value for state-action pair."""
        key = self.get_action_key(state, action)
        self.q_table[key] = value
    
    def save(self, filepath: str) -> None:
        """Save Q-table to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.q_table, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load Q-table from JSON file."""
        with open(filepath, 'r') as f:
            self.q_table = json.load(f)
    
    def decay_epsilon(self) -> None:
        """Decay epsilon for exploration."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information including Q-table size and epsilon."""
        info = super().get_info()
        info.update({
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        })
        return info 