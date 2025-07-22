from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from tictactoe_env import TicTacToeEnv, Player

class BaseAgent(ABC):
    """
    Abstract base class for all reinforcement learning agents.
    
    This class defines the interface that all RL algorithms must implement.
    """
    
    def __init__(self, player: Player, env: TicTacToeEnv, **kwargs):
        """
        Initialize the agent.
        
        Args:
            player: Which player this agent represents (X or O)
            env: The environment instance
            **kwargs: Additional algorithm-specific parameters
        """
        self.player = player
        self.env = env
        self.training_mode = True
        
    @abstractmethod
    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """
        Select an action given the current state and valid actions.
        
        Args:
            state: Current game state (3x3 numpy array)
            valid_actions: List of valid action indices
            
        Returns:
            Selected action index
        """
        pass
    
    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """
        Update the agent's knowledge based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Save the agent's learned parameters to a file.
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Load the agent's learned parameters from a file.
        
        Args:
            filepath: Path to load the model from
        """
        pass
    
    def set_training_mode(self, training: bool) -> None:
        """
        Set whether the agent is in training mode.
        
        Args:
            training: True for training mode, False for evaluation mode
        """
        self.training_mode = training
    
    def get_state_key(self, state: np.ndarray) -> str:
        """
        Convert state to a string key for use in tabular methods.
        
        Args:
            state: Game state as numpy array
            
        Returns:
            String representation of the state
        """
        return str(state.flatten().tolist())
    
    def get_action_key(self, state: np.ndarray, action: int) -> str:
        """
        Get a key representing a state-action pair.
        
        Args:
            state: Game state as numpy array
            action: Action index
            
        Returns:
            String key for the state-action pair
        """
        state_key = self.get_state_key(state)
        return f"{state_key}:{action}"
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the agent.
        
        Returns:
            Dictionary containing agent information
        """
        return {
            'player': self.player.name,
            'training_mode': self.training_mode,
            'agent_type': self.__class__.__name__
        }
    
    def reset_episode(self) -> None:
        """
        Reset agent state for a new episode.
        Override in subclasses if needed.
        """
        pass 