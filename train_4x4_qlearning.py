#!/usr/bin/env python3
"""
Train Q-Learning Agent for 4x4 Tic-Tac-Toe
Optimized parameters for 4x4 board
"""

import sys
import os
import time
import random
from collections import defaultdict

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tictactoe_nd import TicTacToeND
from qlearning.qlearning_agent_nd import QLearningAgentND, random_agent

def calculate_intermediate_reward(game, action, player):
    """Calculate intermediate reward for strategic moves"""
    # Make a temporary move to check the board state
    temp_game = TicTacToeND(dimensions=game.dimensions, board_size=game.board_size, win_length=game.win_length)
    temp_game.board = game.board.copy()
    temp_game.current_player = game.current_player
    
    # Make the move
    coords = temp_game._index_to_coordinates(action)
    temp_game.board[coords] = player
    
    # Check for immediate win
    for line in temp_game.winning_lines:
        values = [temp_game.board[pos] for pos in line]
        if len(set(values)) == 1 and values[0] == player:
            return 5  # High reward for winning move
    
    # Check for blocking opponent's win
    opponent = 3 - player
    for line in temp_game.winning_lines:
        values = [temp_game.board[pos] for pos in line]
        opponent_count = values.count(opponent)
        empty_count = values.count(0)
        if opponent_count == 2 and empty_count == 1:
            return 3  # Good reward for blocking
    
    # Check for creating winning opportunities
    for line in temp_game.winning_lines:
        values = [temp_game.board[pos] for pos in line]
        player_count = values.count(player)
        empty_count = values.count(0)
        if player_count == 2 and empty_count == 1:
            return 2  # Medium reward for creating opportunity
    
    # Check for center control (for 4x4, positions 5,6,9,10 are center)
    center_positions = [5, 6, 9, 10]
    if action in center_positions:
        return 1  # Small reward for center control
    
    return 0  # No special reward

def train_4x4_qlearning(episodes=30000, learning_rate=0.15, gamma=0.95, 
                        epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9997):
    """Train Q-Learning agent for 4x4 Tic-Tac-Toe"""
    
    print("="*60)
    print("TRAINING Q-LEARNING AGENT FOR 4x4 TIC-TAC-TOE")
    print("="*60)
    
    # Game configuration for 4x4
    game_config = {
        'dimensions': 2,
        'board_size': 4,
        'win_length': 3
    }
    
    # Create game instance
    game = TicTacToeND(**game_config)
    board_info = game.get_board_info()
    
    print(f"Board: {board_info['dimensions']}D {board_info['board_size']}x{board_info['board_size']}")
    print(f"Total cells: {board_info['total_cells']}")
    print(f"Winning lines: {board_info['winning_lines_count']}")
    print(f"Episodes: {episodes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Epsilon decay: {epsilon_decay}")
    print()
    
    # Create Q-Learning agent
    agent = QLearningAgentND(
        player=1,
        game_config=game_config,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay
    )
    
    # Training statistics
    wins = {1: 0, 2: 0, 0: 0}  # Player 1, Player 2, Draws
    win_rates = []
    
    print("Starting training...")
    start_time = time.time()
    
    for episode in range(episodes):
        state = game.reset()
        
        while not game.game_over:
            valid_actions = game.get_valid_actions()
            
            if game.current_player == 1:
                # Q-Learning agent's turn
                action = agent.get_action(state, valid_actions, training=True)
                
                # Make move
                next_state, reward, done = game.make_move(action)
                
                # Calculate intermediate reward for strategic moves
                intermediate_reward = calculate_intermediate_reward(game, action, 1)
                total_reward = reward + intermediate_reward
                
                # Update Q-value
                agent.update_q_value(state, action, total_reward, next_state, done)
                
                state = next_state
                
            else:
                # Random opponent's turn
                action = random_agent(valid_actions)
                next_state, reward, done = game.make_move(action)
                state = next_state
        
        # Episode finished - record result
        if game.winner is not None:
            wins[game.winner] += 1
        else:
            wins[0] += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Print progress every 1000 episodes
        if (episode + 1) % 1000 == 0:
            current_win_rate = wins[1] / (episode + 1) * 100
            win_rates.append(current_win_rate)
            
            elapsed_time = time.time() - start_time
            episodes_per_second = (episode + 1) / elapsed_time
            
            print(f"Episode {episode + 1:5d}: "
                  f"Win rate: {current_win_rate:5.1f}%, "
                  f"Epsilon: {agent.get_epsilon():.3f}, "
                  f"Rate: {episodes_per_second:.1f} eps/s")
    
    # Training completed
    total_time = time.time() - start_time
    final_win_rate = wins[1] / episodes * 100
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Total episodes: {episodes}")
    print(f"Training time: {total_time:.2f} seconds")
    print(f"Episodes per second: {episodes/total_time:.1f}")
    print(f"Final win rate: {final_win_rate:.1f}%")
    print(f"Final epsilon: {agent.get_epsilon():.3f}")
    print(f"Results: Player 1 wins: {wins[1]}, Player 2 wins: {wins[2]}, Draws: {wins[0]}")
    
    # Save the trained agent
    agents_dir = "agents"
    os.makedirs(agents_dir, exist_ok=True)
    
    filename = f"qlearning_2d_4x4_player_1.pkl"
    filepath = os.path.join(agents_dir, filename)
    agent.save_q_table(filepath)
    
    print(f"\nTrained agent saved to: {filepath}")
    
    return agent, final_win_rate, total_time

def test_agent_performance(agent, test_episodes=1000):
    """Test the trained agent's performance"""
    print(f"\n{'='*60}")
    print("TESTING AGENT PERFORMANCE")
    print(f"{'='*60}")
    
    game_config = {
        'dimensions': 2,
        'board_size': 4,
        'win_length': 3
    }
    
    game = TicTacToeND(**game_config)
    
    # Test against random opponent
    wins = {1: 0, 2: 0, 0: 0}
    
    for episode in range(test_episodes):
        state = game.reset()
        
        while not game.game_over:
            valid_actions = game.get_valid_actions()
            
            if game.current_player == 1:
                # Trained agent (no exploration)
                action = agent.get_action(state, valid_actions, training=False)
            else:
                # Random opponent
                action = random_agent(valid_actions)
            
            state, reward, done = game.make_move(action)
        
        if game.winner is not None:
            wins[game.winner] += 1
        else:
            wins[0] += 1
    
    win_rate = wins[1] / test_episodes * 100
    print(f"Test episodes: {test_episodes}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Results: Agent wins: {wins[1]}, Opponent wins: {wins[2]}, Draws: {wins[0]}")
    
    return win_rate

def main():
    """Main training function"""
    print("4x4 TIC-TAC-TOE Q-LEARNING TRAINER")
    print("="*50)
    
    # Training parameters optimized for 4x4
    episodes = 30000
    learning_rate = 0.15
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9997
    
    print(f"Training for {episodes} episodes...")
    print(f"Parameters: lr={learning_rate}, γ={gamma}, ε_decay={epsilon_decay}")
    
    # Train the agent
    agent, train_win_rate, train_time = train_4x4_qlearning(
        episodes=episodes,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay
    )
    
    # Test the agent
    test_win_rate = test_agent_performance(agent, test_episodes=1000)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Training win rate: {train_win_rate:.1f}%")
    print(f"Test win rate: {test_win_rate:.1f}%")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Agent saved to: agents/qlearning_2d_4x4_player_1.pkl")
    
    print(f"\nYou can now play against this agent using:")
    print(f"python play_nd_game.py")
    print(f"Choose: 2D 4x4 → Human vs Q-Learning AI")

if __name__ == "__main__":
    main() 