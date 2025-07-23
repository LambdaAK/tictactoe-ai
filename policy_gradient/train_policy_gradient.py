import argparse
import numpy as np
import sys
import os
from collections import defaultdict
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tictactoe import TicTacToe
except ImportError:
    from ..tictactoe import TicTacToe

from policy_gradient_agent import PolicyGradientAgent, random_agent

def train_policy_gradient_agents(episodes: int = 20000):
    game = TicTacToe()
    
    # Create agents with unique names and metadata
    agent1 = PolicyGradientAgent(player=1, learning_rate=0.0005, gamma=0.99, hidden_size=128)
    agent1.name = "PolicyGradient_X_Agent"  # Player 1 is X
    
    agent2 = PolicyGradientAgent(player=2, learning_rate=0.0005, gamma=0.99, hidden_size=128)
    agent2.name = "PolicyGradient_O_Agent"  # Player 2 is O
    
    wins = {1: 0, 2: 0, 0: 0}
    
    for episode in range(episodes):
        state = game.reset()
        
        while not game.game_over:
            valid_actions = game.get_valid_actions()
            
            if game.current_player == 1:
                action = agent1.get_action(state, valid_actions, training=True)
                current_agent = agent1
            else:
                action = agent2.get_action(state, valid_actions, training=True)
                current_agent = agent2
            
            # Store transition for policy gradient update
            # Give small intermediate rewards for strategic moves
            intermediate_reward = calculate_intermediate_reward(game, action)
            current_agent.store_transition(state, action, intermediate_reward)
            
            next_state, reward, done = game.make_move(action)
            state = next_state
        
        # Episode finished, calculate final rewards
        if game.winner is not None:
            wins[game.winner] += 1
            
            # Assign final rewards based on game outcome
            if game.winner == 1:
                # Player 1 won
                agent1.episode_rewards = [r + 20 for r in agent1.episode_rewards]  # Bonus for winning
                agent2.episode_rewards = [r - 20 for r in agent2.episode_rewards]  # Penalty for losing
            elif game.winner == 2:
                # Player 2 won
                agent1.episode_rewards = [r - 20 for r in agent1.episode_rewards]  # Penalty for losing
                agent2.episode_rewards = [r + 20 for r in agent2.episode_rewards]  # Bonus for winning
        else:
            # Draw
            wins[0] += 1
            # Small penalty for draws to encourage winning
            agent1.episode_rewards = [r - 5 for r in agent1.episode_rewards]
            agent2.episode_rewards = [r - 5 for r in agent2.episode_rewards]
        
        # Update policies
        agent1.update_policy()
        agent2.update_policy()
        
        if episode % 1000 == 0:
            win_rate_1 = wins[1] / (episode + 1) * 100
            win_rate_2 = wins[2] / (episode + 1) * 100
            draw_rate = wins[0] / (episode + 1) * 100
            print(f"Episode {episode}: {agent1.name} wins: {win_rate_1:.1f}%, {agent2.name} wins: {win_rate_2:.1f}%, Draws: {draw_rate:.1f}%")
    
    print(f"\nTraining completed!")
    print(f"Final stats - {agent1.name} wins: {wins[1]}, {agent2.name} wins: {wins[2]}, Draws: {wins[0]}")
    
    # Save agents with metadata
    save_agent_with_metadata(agent1, episodes)
    save_agent_with_metadata(agent2, episodes)
    
    return agent1, agent2

def calculate_intermediate_reward(game, action):
    """Calculate intermediate reward for strategic moves"""
    # Check if this move creates a winning opportunity
    row, col = action // 3, action % 3
    temp_board = game.board.copy()
    temp_board[row][col] = game.current_player
    
    # Check if this move wins the game
    if check_win(temp_board, game.current_player):
        return 5  # Bonus for winning move
    
    # Check if this move blocks opponent's win
    opponent = 3 - game.current_player
    if check_win(temp_board, opponent):
        return 3  # Bonus for blocking
    
    # Check if this move creates a fork (two winning opportunities)
    if count_winning_opportunities(temp_board, game.current_player) >= 2:
        return 2  # Bonus for creating fork
    
    return 0  # No special reward

def check_win(board, player):
    """Check if a player has won on the given board"""
    # Check rows, columns, and diagonals
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] == player:
            return True
        if board[0][i] == board[1][i] == board[2][i] == player:
            return True
    
    if board[0][0] == board[1][1] == board[2][2] == player:
        return True
    if board[0][2] == board[1][1] == board[2][0] == player:
        return True
    
    return False

def count_winning_opportunities(board, player):
    """Count how many winning opportunities a player has"""
    count = 0
    
    # Check rows
    for i in range(3):
        if sum(1 for j in range(3) if board[i][j] == player) == 2 and sum(1 for j in range(3) if board[i][j] == 0) == 1:
            count += 1
    
    # Check columns
    for j in range(3):
        if sum(1 for i in range(3) if board[i][j] == player) == 2 and sum(1 for i in range(3) if board[i][j] == 0) == 1:
            count += 1
    
    # Check diagonals
    if sum(1 for i in range(3) if board[i][i] == player) == 2 and sum(1 for i in range(3) if board[i][i] == 0) == 1:
        count += 1
    if sum(1 for i in range(3) if board[i][2-i] == player) == 2 and sum(1 for i in range(3) if board[i][2-i] == 0) == 1:
        count += 1
    
    return count

def save_agent_with_metadata(agent: PolicyGradientAgent, episodes: int):
    """Save agent with metadata including training parameters and statistics"""
    # Create agents directory if it doesn't exist
    agents_dir = "agents"
    os.makedirs(agents_dir, exist_ok=True)
    
    filename = f"{agent.name.lower()}_policy.pkl"
    filepath = os.path.join(agents_dir, filename)
    
    agent.save_policy(filepath)
    print(f"Saved {agent.name} to {filepath}")

def load_agent_with_metadata(filename: str) -> PolicyGradientAgent:
    """Load agent with metadata from file"""
    agent = PolicyGradientAgent(player=1)  # Temporary player number
    agent.load_policy(filename)
    print(f"Loaded {agent.name} from {filename}")
    return agent

def load_agent_by_name(agent_name: str) -> PolicyGradientAgent:
    """Load agent by name from the agents directory"""
    agents_dir = "agents"
    filename = f"{agent_name.lower()}_policy.pkl"
    filepath = os.path.join(agents_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Agent file not found: {filepath}")
    
    return load_agent_with_metadata(filepath)

def test_against_random(agent: PolicyGradientAgent, games: int = 1000):
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

def compare_algorithms():
    """Compare Policy Gradient vs Q-Learning performance"""
    print("Training Policy Gradient agents...")
    pg_agent1, pg_agent2 = train_policy_gradient_agents(episodes=10000)
    
    print("\nTesting Policy Gradient agent against random player...")
    test_against_random(pg_agent1, games=500)
    
    # You can add Q-learning comparison here if needed
    print("\nPolicy Gradient training completed!")

if __name__ == "__main__":
    print("Training Policy Gradient agents...")
    agent1, agent2 = train_policy_gradient_agents()
    
    print("\nTesting trained agent against random player...")
    test_against_random(agent1) 