import numpy as np
import random
import json
from typing import List, Dict, Any
from base_agent import BaseAgent
from tictactoe_env import TicTacToeEnv, Player

class RandomAgent(BaseAgent):
    """
    Random agent that selects actions randomly.
    
    Useful for baseline comparison and testing.
    """
    
    def __init__(self, player: Player, env: TicTacToeEnv):
        """Initialize random agent."""
        super().__init__(player, env)
        self.move_count = 0
    
    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """Select a random valid action."""
        if not valid_actions:
            raise ValueError("No valid actions available")
        return random.choice(valid_actions)
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """Random agent doesn't learn, so this is a no-op."""
        pass
    
    def save(self, filepath: str) -> None:
        """Save agent info (random agent has no parameters to save)."""
        info = {
            'agent_type': 'RandomAgent',
            'player': self.player.name,
            'move_count': self.move_count
        }
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load agent info (random agent has no parameters to load)."""
        with open(filepath, 'r') as f:
            info = json.load(f)
            self.move_count = info.get('move_count', 0)
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        info = super().get_info()
        info.update({
            'move_count': self.move_count
        })
        return info 