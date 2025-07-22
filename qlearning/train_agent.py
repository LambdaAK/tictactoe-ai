import argparse
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tictactoe import TicTacToe
except ImportError:
    from ..tictactoe import TicTacToe

from qlearning_agent import QLearningAgent, random_agent

def train_q_learning_agent(episodes: int = 50000):
    game = TicTacToe()
    agent1 = QLearningAgent(player=1, learning_rate=0.1, discount_factor=0.95, 
                           epsilon=0.8, epsilon_decay=0.9995, epsilon_min=0.01)
    agent2 = QLearningAgent(player=2, learning_rate=0.1, discount_factor=0.95, 
                           epsilon=0.8, epsilon_decay=0.9995, epsilon_min=0.01)
    
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
            print(f"Episode {episode}: Agent1 wins: {win_rate_1:.1f}%, Agent2 wins: {win_rate_2:.1f}%, Draws: {draw_rate:.1f}%")
            print(f"Epsilon - Agent1: {agent1.get_epsilon():.4f}, Agent2: {agent2.get_epsilon():.4f}")
    
    print(f"\nTraining completed!")
    print(f"Final stats - Agent1 wins: {wins[1]}, Agent2 wins: {wins[2]}, Draws: {wins[0]}")
    print(f"Final epsilon - Agent1: {agent1.get_epsilon():.4f}, Agent2: {agent2.get_epsilon():.4f}")
    
    agent1.save_q_table("agent1_qtable.pkl")
    agent2.save_q_table("agent2_qtable.pkl")
    
    return agent1, agent2

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
    
    print(f"Against random opponent ({games} games):")
    print(f"Agent wins: {agent_wins} ({agent_wins/games*100:.1f}%)")
    print(f"Random wins: {opponent_wins} ({opponent_wins/games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/games*100:.1f}%)")

if __name__ == "__main__":
    print("Training Q-Learning agents...")
    agent1, agent2 = train_q_learning_agent()
    
    print("\nTesting trained agent against random player...")
    test_against_random(agent1)