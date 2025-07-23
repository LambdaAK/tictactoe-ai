import numpy as np
import random
import pickle
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tictactoe import TicTacToe
except ImportError:
    from ..tictactoe import TicTacToe

class PolicyNetwork(nn.Module):
    """Neural network for policy approximation"""
    
    def __init__(self, input_size=9, hidden_size=64, output_size=9):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PolicyGradientAgent:
    def __init__(self, player: int, learning_rate: float = 0.0001, 
                 gamma: float = 0.95, hidden_size: int = 64):
        self.player = player
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.hidden_size = hidden_size
        
        # Neural network for policy
        self.policy_net = PolicyNetwork(input_size=9, hidden_size=hidden_size, output_size=9)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Training data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Metadata
        self.name = f"PolicyGradient_Player_{player}"
        
    def state_to_tensor(self, state: str) -> torch.Tensor:
        """Convert board state string to tensor representation"""
        # Convert state string to numerical representation
        # '0' -> 0 (empty), '1' -> 1 (player 1/X), '2' -> -1 (player 2/O)
        state_map = {'0': 0, '1': 1, '2': -1}
        state_vector = [state_map[cell] for cell in state]
        return torch.FloatTensor(state_vector)
    
    def get_action_probs(self, state: str, valid_actions: List[int]) -> torch.Tensor:
        """Get action probabilities for valid actions"""
        state_tensor = self.state_to_tensor(state)
        logits = self.policy_net(state_tensor)
        
        # Mask invalid actions by setting their logits to -inf
        mask = torch.ones(9) * float('-inf')
        for action in valid_actions:
            mask[action] = 0
        
        masked_logits = logits + mask
        action_probs = F.softmax(masked_logits, dim=0)
        
        return action_probs
    
    def get_action(self, state: str, valid_actions: List[int], training: bool = True) -> int:
        """Select action using current policy"""
        action_probs = self.get_action_probs(state, valid_actions)
        
        if training:
            # Sample action from probability distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            return action.item()
        else:
            # Choose best action (greedy)
            return valid_actions[torch.argmax(action_probs[valid_actions])]
    
    def store_transition(self, state: str, action: int, reward: float):
        """Store state, action, and reward for current episode"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def update_policy(self):
        """Update policy using REINFORCE algorithm"""
        if not self.episode_states:
            return
        
        # Calculate discounted returns
        returns = []
        R = 0
        for reward in reversed(self.episode_rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        # Normalize returns
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for state, action, R in zip(self.episode_states, self.episode_actions, returns):
            state_tensor = self.state_to_tensor(state)
            logits = self.policy_net(state_tensor)
            
            # Create action mask for valid actions
            # We need to reconstruct what the valid actions were at this state
            # For simplicity, we'll assume the action taken was valid
            mask = torch.ones(9) * float('-inf')
            mask[action] = 0  # Only the action that was actually taken
            
            masked_logits = logits + mask
            action_probs = F.softmax(masked_logits, dim=0)
            
            # Calculate log probability of the taken action
            log_prob = torch.log(action_probs[action] + 1e-8)
            
            # REINFORCE loss: -log_prob * return
            policy_loss.append(-log_prob * R)
        
        # Average loss over episode
        policy_loss = torch.stack(policy_loss).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def save_policy(self, filename: str):
        """Save policy network and metadata"""
        metadata = {
            'name': self.name,
            'player': self.player,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'hidden_size': self.hidden_size,
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_policy(self, filename: str):
        """Load policy network and metadata"""
        with open(filename, 'rb') as f:
            metadata = pickle.load(f)
        
        self.name = metadata['name']
        self.player = metadata['player']
        self.learning_rate = metadata['learning_rate']
        self.gamma = metadata['gamma']
        self.hidden_size = metadata['hidden_size']
        
        self.policy_net.load_state_dict(metadata['policy_state_dict'])
        self.optimizer.load_state_dict(metadata['optimizer_state_dict'])
    
    def get_action_probabilities(self, state: str, valid_actions: List[int]) -> Dict[int, float]:
        """Get probability distribution over valid actions"""
        action_probs = self.get_action_probs(state, valid_actions)
        return {action: action_probs[action].item() for action in valid_actions}

def random_agent(valid_actions: List[int]) -> int:
    """Random agent for comparison"""
    return random.choice(valid_actions) 