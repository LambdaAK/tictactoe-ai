#!/usr/bin/env python3
"""
MCTS Training and Optimization Framework
This module provides various methods to improve MCTS agent performance.
"""

import numpy as np
import random
import time
import json
from typing import Dict, List, Tuple, Optional
from mcts_agent import MCTSAgent, RandomAgent, play_game_with_mcts
from tictactoe import TicTacToe
import matplotlib.pyplot as plt
from collections import defaultdict


class MCTSTrainer:
    """Framework for training and optimizing MCTS agents."""
    
    def __init__(self):
        self.training_history = []
        self.best_parameters = {}
    
    def parameter_optimization(self, num_trials=50, games_per_trial=20):
        """Optimize MCTS parameters using grid search."""
        print("Starting MCTS Parameter Optimization")
        print("=" * 40)
        
        # Define parameter ranges to test
        iteration_ranges = [100, 500, 1000, 2000, 3000]
        exploration_constants = [0.5, 1.0, 1.414, 2.0, 3.0]
        
        best_win_rate = 0
        best_params = {}
        results = []
        
        random_agent = RandomAgent(player=2)
        
        for iterations in iteration_ranges:
            for exploration_constant in exploration_constants:
                print(f"\nTesting: iterations={iterations}, exploration_constant={exploration_constant}")
                
                mcts_agent = MCTSAgent(
                    player=1, 
                    iterations=iterations, 
                    exploration_constant=exploration_constant
                )
                
                # Test against random agent
                wins, draws, losses = self._evaluate_agent(
                    mcts_agent, random_agent, num_games=games_per_trial
                )
                
                win_rate = wins / games_per_trial
                results.append({
                    'iterations': iterations,
                    'exploration_constant': exploration_constant,
                    'win_rate': win_rate,
                    'wins': wins,
                    'draws': draws,
                    'losses': losses
                })
                
                print(f"Win rate: {win_rate:.3f} ({wins}/{games_per_trial})")
                
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_params = {
                        'iterations': iterations,
                        'exploration_constant': exploration_constant
                    }
        
        print(f"\nBest parameters found:")
        print(f"Iterations: {best_params['iterations']}")
        print(f"Exploration constant: {best_params['exploration_constant']}")
        print(f"Best win rate: {best_win_rate:.3f}")
        
        self.best_parameters = best_params
        return best_params, results
    
    def self_play_training(self, num_games=1000, save_interval=100):
        """Train MCTS agent through self-play."""
        print("Starting MCTS Self-Play Training")
        print("=" * 40)
        
        # Create two MCTS agents with different parameters
        agent1 = MCTSAgent(player=1, iterations=1000, exploration_constant=1.414)
        agent2 = MCTSAgent(player=2, iterations=1000, exploration_constant=1.414)
        
        training_stats = {
            'games_played': 0,
            'agent1_wins': 0,
            'agent2_wins': 0,
            'draws': 0,
            'avg_game_length': []
        }
        
        for game_num in range(num_games):
            if game_num % save_interval == 0:
                print(f"Training game {game_num}/{num_games}")
                self._print_training_stats(training_stats)
            
            # Play a game between the two agents
            game = TicTacToe()
            moves = 0
            
            while not game.game_over:
                if game.current_player == 1:
                    action = agent1.get_action(game)
                else:
                    action = agent2.get_action(game)
                
                state, reward, game_over = game.make_move(action)
                moves += 1
            
            # Update statistics
            training_stats['games_played'] += 1
            training_stats['avg_game_length'].append(moves)
            
            if game.winner == 1:
                training_stats['agent1_wins'] += 1
            elif game.winner == 2:
                training_stats['agent2_wins'] += 1
            else:
                training_stats['draws'] += 1
            
            # Reset agents for next game
            agent1.reset()
            agent2.reset()
        
        print(f"\nSelf-play training completed!")
        self._print_training_stats(training_stats)
        
        return training_stats
    
    def progressive_training(self, initial_iterations=100, max_iterations=5000, step_size=100):
        """Progressively increase MCTS iterations to find optimal performance."""
        print("Starting Progressive MCTS Training")
        print("=" * 40)
        
        iteration_counts = list(range(initial_iterations, max_iterations + 1, step_size))
        performance_data = []
        
        random_agent = RandomAgent(player=2)
        
        for iterations in iteration_counts:
            print(f"\nTesting with {iterations} iterations...")
            
            mcts_agent = MCTSAgent(player=1, iterations=iterations)
            
            start_time = time.time()
            wins, draws, losses = self._evaluate_agent(mcts_agent, random_agent, num_games=50)
            end_time = time.time()
            
            win_rate = wins / 50
            avg_time = (end_time - start_time) / 50
            
            performance_data.append({
                'iterations': iterations,
                'win_rate': win_rate,
                'avg_time': avg_time,
                'wins': wins,
                'draws': draws,
                'losses': losses
            })
            
            print(f"Win rate: {win_rate:.3f}, Avg time: {avg_time:.3f}s")
        
        return performance_data
    
    def adaptive_training(self, target_win_rate=0.8, max_iterations=10000):
        """Adaptively adjust iterations to achieve target performance."""
        print("Starting Adaptive MCTS Training")
        print("=" * 40)
        
        current_iterations = 100
        random_agent = RandomAgent(player=2)
        
        while current_iterations <= max_iterations:
            print(f"\nTesting with {current_iterations} iterations...")
            
            mcts_agent = MCTSAgent(player=1, iterations=current_iterations)
            
            wins, draws, losses = self._evaluate_agent(mcts_agent, random_agent, num_games=30)
            win_rate = wins / 30
            
            print(f"Win rate: {win_rate:.3f} ({wins}/30)")
            
            if win_rate >= target_win_rate:
                print(f"\nTarget win rate achieved with {current_iterations} iterations!")
                return current_iterations
            
            # Increase iterations for next trial
            current_iterations = int(current_iterations * 1.5)
        
        print(f"\nTarget win rate not achieved within {max_iterations} iterations")
        return max_iterations
    
    def _evaluate_agent(self, agent, opponent, num_games=50):
        """Evaluate an agent against an opponent."""
        wins = 0
        draws = 0
        losses = 0
        
        for i in range(num_games):
            winner = play_game_with_mcts(agent, opponent, display=False)
            
            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1
            
            agent.reset()
            if hasattr(opponent, 'reset'):
                opponent.reset()
        
        return wins, draws, losses
    
    def _print_training_stats(self, stats):
        """Print training statistics."""
        total_games = stats['games_played']
        if total_games == 0:
            return
        
        agent1_rate = stats['agent1_wins'] / total_games
        agent2_rate = stats['agent2_wins'] / total_games
        draw_rate = stats['draws'] / total_games
        avg_length = np.mean(stats['avg_game_length'])
        
        print(f"Games: {total_games}")
        print(f"Agent 1 wins: {agent1_rate:.3f} ({stats['agent1_wins']})")
        print(f"Agent 2 wins: {agent2_rate:.3f} ({stats['agent2_wins']})")
        print(f"Draws: {draw_rate:.3f} ({stats['draws']})")
        print(f"Average game length: {avg_length:.1f} moves")


class MCTSVisualizer:
    """Visualize MCTS training results."""
    
    @staticmethod
    def plot_parameter_optimization(results):
        """Plot parameter optimization results."""
        iterations = list(set([r['iterations'] for r in results]))
        exploration_constants = list(set([r['exploration_constant'] for r in results]))
        
        # Create heatmap data
        heatmap_data = np.zeros((len(exploration_constants), len(iterations)))
        
        for result in results:
            i = exploration_constants.index(result['exploration_constant'])
            j = iterations.index(result['iterations'])
            heatmap_data[i, j] = result['win_rate']
        
        plt.figure(figsize=(10, 6))
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
        plt.colorbar(label='Win Rate')
        plt.xlabel('Iterations')
        plt.ylabel('Exploration Constant')
        plt.title('MCTS Parameter Optimization Results')
        
        # Set tick labels
        plt.xticks(range(len(iterations)), iterations)
        plt.yticks(range(len(exploration_constants)), exploration_constants)
        
        plt.tight_layout()
        plt.savefig('mcts_parameter_optimization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_progressive_training(performance_data):
        """Plot progressive training results."""
        iterations = [d['iterations'] for d in performance_data]
        win_rates = [d['win_rate'] for d in performance_data]
        avg_times = [d['avg_time'] for d in performance_data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Win rate plot
        ax1.plot(iterations, win_rates, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Win Rate')
        ax1.set_title('MCTS Performance vs Iterations')
        ax1.grid(True, alpha=0.3)
        
        # Time plot
        ax2.plot(iterations, avg_times, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Average Time per Game (s)')
        ax2.set_title('MCTS Computation Time vs Iterations')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mcts_progressive_training.png', dpi=300, bbox_inches='tight')
        plt.show()


def save_trained_agent(agent, filename):
    """Save trained agent parameters to file."""
    agent_data = {
        'player': agent.player,
        'iterations': agent.iterations,
        'exploration_constant': agent.exploration_constant
    }
    
    with open(filename, 'w') as f:
        json.dump(agent_data, f, indent=2)
    
    print(f"Agent saved to {filename}")


def load_trained_agent(filename):
    """Load trained agent parameters from file."""
    with open(filename, 'r') as f:
        agent_data = json.load(f)
    
    agent = MCTSAgent(
        player=agent_data['player'],
        iterations=agent_data['iterations'],
        exploration_constant=agent_data['exploration_constant']
    )
    
    print(f"Agent loaded from {filename}")
    return agent


if __name__ == "__main__":
    print("MCTS Training and Optimization Framework")
    print("=" * 50)
    
    trainer = MCTSTrainer()
    visualizer = MCTSVisualizer()
    
    # 1. Parameter optimization
    print("\n1. Running parameter optimization...")
    best_params, optimization_results = trainer.parameter_optimization(
        num_trials=25, games_per_trial=20
    )
    
    # 2. Progressive training
    print("\n2. Running progressive training...")
    progressive_results = trainer.progressive_training(
        initial_iterations=100, max_iterations=2000, step_size=200
    )
    
    # 3. Adaptive training
    print("\n3. Running adaptive training...")
    optimal_iterations = trainer.adaptive_training(target_win_rate=0.85)
    
    # 4. Create and save optimized agent
    optimized_agent = MCTSAgent(
        player=1,
        iterations=best_params['iterations'],
        exploration_constant=best_params['exploration_constant']
    )
    
    save_trained_agent(optimized_agent, 'optimized_mcts_agent.json')
    
    # 5. Visualize results
    print("\n4. Generating visualizations...")
    visualizer.plot_parameter_optimization(optimization_results)
    visualizer.plot_progressive_training(progressive_results)
    
    print("\nTraining completed! Check the generated files:")
    print("- optimized_mcts_agent.json: Best agent parameters")
    print("- mcts_parameter_optimization.png: Parameter optimization heatmap")
    print("- mcts_progressive_training.png: Progressive training results") 