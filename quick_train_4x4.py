#!/usr/bin/env python3
"""
Quick Train Q-Learning Agent for 4x4 Tic-Tac-Toe
Faster training with fewer episodes
"""

import sys
import os
import time
import random

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tictactoe_nd import TicTacToeND
from qlearning.qlearning_agent_nd import QLearningAgentND, random_agent

def quick_train_4x4(episodes=5000):
    """Quick training for 4x4 Tic-Tac-Toe"""
    
    print("="*50)
    print("QUICK TRAINING: 4x4 TIC-TAC-TOE Q-LEARNING")
    print("="*50)
    
    # Game configuration
    game_config = {
        'dimensions': 2,
        'board_size': 4,
        'win_length': 3
    }
    
    # Create game and agent
    game = TicTacToeND(**game_config)
    agent = QLearningAgentND(
        player=1,
        game_config=game_config,
        learning_rate=0.2,  # Higher learning rate for faster learning
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995  # Faster decay
    )
    
    print(f"Training for {episodes} episodes...")
    print(f"Board: 4x4 (16 cells, 48 winning lines)")
    print()
    
    # Training
    wins = {1: 0, 2: 0, 0: 0}
    start_time = time.time()
    
    for episode in range(episodes):
        state = game.reset()
        
        while not game.game_over:
            valid_actions = game.get_valid_actions()
            
            if game.current_player == 1:
                # Q-Learning agent
                action = agent.get_action(state, valid_actions, training=True)
                next_state, reward, done = game.make_move(action)
                agent.update_q_value(state, action, reward, next_state, done)
                state = next_state
            else:
                # Random opponent
                action = random_agent(valid_actions)
                next_state, reward, done = game.make_move(action)
                state = next_state
        
        # Record result
        if game.winner is not None:
            wins[game.winner] += 1
        else:
            wins[0] += 1
        
        agent.decay_epsilon()
        
        # Print progress every 500 episodes
        if (episode + 1) % 500 == 0:
            win_rate = wins[1] / (episode + 1) * 100
            print(f"Episode {episode + 1:4d}: Win rate: {win_rate:5.1f}%, Epsilon: {agent.get_epsilon():.3f}")
    
    # Results
    total_time = time.time() - start_time
    final_win_rate = wins[1] / episodes * 100
    
    print(f"\nTraining completed in {total_time:.1f} seconds")
    print(f"Final win rate: {final_win_rate:.1f}%")
    print(f"Results: Wins: {wins[1]}, Losses: {wins[2]}, Draws: {wins[0]}")
    
    # Save agent
    agents_dir = "agents"
    os.makedirs(agents_dir, exist_ok=True)
    
    filename = "qlearning_2d_4x4_player_1.pkl"
    filepath = os.path.join(agents_dir, filename)
    agent.save_q_table(filepath)
    
    print(f"Agent saved to: {filepath}")
    
    return agent, final_win_rate

def test_quick_agent(agent, test_episodes=500):
    """Quick test of the trained agent"""
    print(f"\nTesting agent against random opponent ({test_episodes} games)...")
    
    game_config = {
        'dimensions': 2,
        'board_size': 4,
        'win_length': 3
    }
    
    game = TicTacToeND(**game_config)
    wins = {1: 0, 2: 0, 0: 0}
    
    for episode in range(test_episodes):
        state = game.reset()
        
        while not game.game_over:
            valid_actions = game.get_valid_actions()
            
            if game.current_player == 1:
                action = agent.get_action(state, valid_actions, training=False)
            else:
                action = random_agent(valid_actions)
            
            state, reward, done = game.make_move(action)
        
        if game.winner is not None:
            wins[game.winner] += 1
        else:
            wins[0] += 1
    
    win_rate = wins[1] / test_episodes * 100
    print(f"Test win rate: {win_rate:.1f}%")
    print(f"Test results: Wins: {wins[1]}, Losses: {wins[2]}, Draws: {wins[0]}")
    
    return win_rate

def main():
    """Main function"""
    print("QUICK 4x4 TIC-TAC-TOE Q-LEARNING TRAINER")
    print("="*50)
    
    # Quick training with 5000 episodes
    agent, train_win_rate = quick_train_4x4(episodes=5000)
    
    # Quick test
    test_win_rate = test_quick_agent(agent, test_episodes=500)
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Training win rate: {train_win_rate:.1f}%")
    print(f"Test win rate: {test_win_rate:.1f}%")
    print(f"Agent saved to: agents/qlearning_2d_4x4_player_1.pkl")
    
    print(f"\nTo play against this agent:")
    print(f"python play_nd_game.py")
    print(f"Choose: 2D 4x4 → Human vs Q-Learning AI")
    
    print(f"\nOr for quick play:")
    print(f"python quick_play.py")

if __name__ == "__main__":
    main() 