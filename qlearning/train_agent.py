import argparse
import numpy as np
import sys
import os
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tictactoe import TicTacToe
except ImportError:
    from ..tictactoe import TicTacToe

from qlearning_agent import QLearningAgent, random_agent

def train_q_learning_agent(episodes: int = 50000):
    game = TicTacToe()
    
    # Create agents with unique names and metadata
    agent1 = QLearningAgent(player=1, learning_rate=0.1, discount_factor=0.95, 
                           epsilon=0.8, epsilon_decay=0.9995, epsilon_min=0.01)
    agent1.name = "X_Agent"  # Player 1 is X
    
    agent2 = QLearningAgent(player=2, learning_rate=0.1, discount_factor=0.95, 
                           epsilon=0.8, epsilon_decay=0.9995, epsilon_min=0.01)
    agent2.name = "O_Agent"  # Player 2 is O
    
    wins = {1: 0, 2: 0, 0: 0}
    
    for episode in range(episodes):
        state = game.reset()
        states_actions_rewards = []
        
        while not game.game_over:
            valid_actions = game.get_valid_actions()
            
            if game.current_player == 1:
                action = agent1.get_action(state, valid_actions, training=True)
                current_agent = agent1
            else:
                action = agent2.get_action(state, valid_actions, training=True)
                current_agent = agent2
            
            states_actions_rewards.append((state, action, game.current_player))
            
            next_state, reward, done = game.make_move(action)
            
            if done:
                if game.winner is not None:
                    wins[game.winner] += 1
                
                for i, (s, a, player) in enumerate(reversed(states_actions_rewards)):
                    if player == game.winner:
                        final_reward = 10
                    elif game.winner == 0:
                        final_reward = 0
                    else:
                        final_reward = -10
                    
                    discounted_reward = final_reward * (0.95 ** i)
                    
                    if player == 1:
                        agent1.update_q_value(s, a, discounted_reward, next_state, [])
                    else:
                        agent2.update_q_value(s, a, discounted_reward, next_state, [])
            
            state = next_state
        
        # Decay epsilon after each episode
        agent1.decay_epsilon()
        agent2.decay_epsilon()
        
        if episode % 10000 == 0:
            win_rate_1 = wins[1] / (episode + 1) * 100
            win_rate_2 = wins[2] / (episode + 1) * 100
            draw_rate = wins[0] / (episode + 1) * 100
            print(f"Episode {episode}: {agent1.name} wins: {win_rate_1:.1f}%, {agent2.name} wins: {win_rate_2:.1f}%, Draws: {draw_rate:.1f}%")
            print(f"Epsilon - {agent1.name}: {agent1.get_epsilon():.4f}, {agent2.name}: {agent2.get_epsilon():.4f}")
    
    print(f"\nTraining completed!")
    print(f"Final stats - {agent1.name} wins: {wins[1]}, {agent2.name} wins: {wins[2]}, Draws: {wins[0]}")
    print(f"Final epsilon - {agent1.name}: {agent1.get_epsilon():.4f}, {agent2.name}: {agent2.get_epsilon():.4f}")
    
    # Save agents with metadata
    save_agent_with_metadata(agent1, episodes)
    save_agent_with_metadata(agent2, episodes)
    
    return agent1, agent2

def save_agent_with_metadata(agent: QLearningAgent, episodes: int):
    """Save agent with metadata including training parameters and statistics"""
    import pickle
    from datetime import datetime
    
    # Create agents directory if it doesn't exist
    agents_dir = "agents"
    os.makedirs(agents_dir, exist_ok=True)
    
    metadata = {
        'name': agent.name,
        'player': agent.player,
        'training_episodes': episodes,
        'learning_rate': agent.learning_rate,
        'discount_factor': agent.discount_factor,
        'initial_epsilon': agent.initial_epsilon,
        'epsilon_decay': agent.epsilon_decay,
        'epsilon_min': agent.epsilon_min,
        'final_epsilon': agent.get_epsilon(),
        'training_date': datetime.now().isoformat(),
        'q_table': {state: dict(actions) for state, actions in agent.q_table.items()}
    }
    
    filename = f"{agent.name.lower()}_qtable.pkl"
    filepath = os.path.join(agents_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Saved {agent.name} to {filepath}")

def load_agent_with_metadata(filename: str) -> QLearningAgent:
    """Load agent with metadata from file"""
    import pickle
    
    with open(filename, 'rb') as f:
        metadata = pickle.load(f)
    
    # Create agent with loaded parameters
    agent = QLearningAgent(
        player=metadata['player'],
        learning_rate=metadata['learning_rate'],
        discount_factor=metadata['discount_factor'],
        epsilon=metadata['final_epsilon'],
        epsilon_decay=metadata['epsilon_decay'],
        epsilon_min=metadata['epsilon_min']
    )
    
    # Restore name and q_table
    agent.name = metadata['name']
    agent.q_table = defaultdict(lambda: defaultdict(float))
    for state, actions in metadata['q_table'].items():
        for action, q_value in actions.items():
            agent.q_table[state][action] = q_value
    
    print(f"Loaded {agent.name} from {filename}")
    print(f"Training info: {metadata['training_episodes']} episodes, trained on {metadata['training_date']}")
    
    return agent

def load_agent_by_name(agent_name: str) -> QLearningAgent:
    """Load agent by name from the agents directory"""
    agents_dir = "agents"
    filename = f"{agent_name.lower()}_qtable.pkl"
    filepath = os.path.join(agents_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Agent file not found: {filepath}")
    
    return load_agent_with_metadata(filepath)

def test_against_random(agent: QLearningAgent, games: int = 1000):
    game = TicTacToe()
    wins = {1: 0, 2: 0, 0: 0}
    
    for _ in range(games):
        state = game.reset()
        
        while not game.game_over:
            valid_actions = game.get_valid_actions()
            
            if game.current_player == agent.player:
                action = agent.get_action(state, valid_actions, training=False)
            else:
                action = random_agent(valid_actions)
            
            state, reward, done = game.make_move(action)
        
        wins[game.winner] += 1
    
    agent_wins = wins[agent.player]
    opponent_wins = wins[3 - agent.player]
    draws = wins[0]
    
    print(f"{agent.name} against random opponent ({games} games):")
    print(f"{agent.name} wins: {agent_wins} ({agent_wins/games*100:.1f}%)")
    print(f"Random wins: {opponent_wins} ({opponent_wins/games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/games*100:.1f}%)")

if __name__ == "__main__":
    print("Training Q-Learning agents...")
    agent1, agent2 = train_q_learning_agent()
    
    print("\nTesting trained agent against random player...")
    test_against_random(agent1)