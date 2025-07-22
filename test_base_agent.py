from base_agent import BaseAgent
from tictactoe_env import TicTacToeEnv, Player
import numpy as np

class MockAgent(BaseAgent):
    """Simple mock agent for testing the base class"""
    
    def __init__(self, player: Player, env: TicTacToeEnv):
        super().__init__(player, env)
        self.q_table = {}
    
    def select_action(self, state, valid_actions):
        # Simple random selection for testing
        return np.random.choice(valid_actions)
    
    def update(self, state, action, reward, next_state, done):
        # Simple update for testing
        key = self.get_action_key(state, action)
        if key not in self.q_table:
            self.q_table[key] = 0.0
        self.q_table[key] += reward * 0.1
    
    def save(self, filepath):
        # Simple save for testing
        with open(filepath, 'w') as f:
            f.write(str(self.q_table))
    
    def load(self, filepath):
        # Simple load for testing
        with open(filepath, 'r') as f:
            self.q_table = eval(f.read())

def test_base_agent_interface():
    """Test the base agent interface"""
    print("Testing base agent interface...")
    
    # Create environment and agent
    env = TicTacToeEnv()
    agent = MockAgent(Player.X, env)
    
    # Test basic properties
    print(f"Agent player: {agent.player}")
    print(f"Training mode: {agent.training_mode}")
    print(f"Agent info: {agent.get_info()}")
    
    # Test state key generation
    state = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0]])
    state_key = agent.get_state_key(state)
    print(f"State key: {state_key}")
    
    # Test action key generation
    action_key = agent.get_action_key(state, 4)
    print(f"Action key: {action_key}")
    
    # Test action selection
    valid_actions = [0, 1, 2, 3, 5, 6, 7, 8]
    action = agent.select_action(state, valid_actions)
    print(f"Selected action: {action}")
    assert action in valid_actions
    
    # Test update
    next_state = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
    agent.update(state, action, 0.5, next_state, False)
    print(f"Q-table size: {len(agent.q_table)}")
    
    # Test training mode toggle
    agent.set_training_mode(False)
    print(f"Training mode after toggle: {agent.training_mode}")
    
    # Test save/load
    agent.save("test_agent.txt")
    new_agent = MockAgent(Player.O, env)
    new_agent.load("test_agent.txt")
    print(f"Loaded Q-table size: {len(new_agent.q_table)}")
    
    print("All base agent tests passed!")

if __name__ == "__main__":
    test_base_agent_interface() 