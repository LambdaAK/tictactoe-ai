#!/usr/bin/env python3
"""
Simple MCTS Training Script
This script provides basic training methods for MCTS agents without external dependencies.
"""

import time
import json
from mcts_agent import MCTSAgent, RandomAgent, play_game_with_mcts
from tictactoe import TicTacToe


def optimize_parameters():
    """Find the best MCTS parameters through testing."""
    print("MCTS Parameter Optimization")
    print("=" * 30)
    
    # Test different parameter combinations
    test_configs = [
        {'iterations': 100, 'exploration_constant': 1.414},
        {'iterations': 500, 'exploration_constant': 1.414},
        {'iterations': 1000, 'exploration_constant': 1.414},
        {'iterations': 1000, 'exploration_constant': 1.0},
        {'iterations': 1000, 'exploration_constant': 2.0},
        {'iterations': 2000, 'exploration_constant': 1.414},
        {'iterations': 2000, 'exploration_constant': 1.0},
        {'iterations': 2000, 'exploration_constant': 2.0},
    ]
    
    best_win_rate = 0
    best_config = None
    results = []
    
    random_agent = RandomAgent(player=2)
    
    for config in test_configs:
        print(f"\nTesting: {config}")
        
        mcts_agent = MCTSAgent(
            player=1,
            iterations=config['iterations'],
            exploration_constant=config['exploration_constant']
        )
        
        # Test against random agent
        wins, draws, losses = evaluate_agent(mcts_agent, random_agent, num_games=30)
        win_rate = wins / 30
        
        results.append({
            'config': config,
            'win_rate': win_rate,
            'wins': wins,
            'draws': draws,
            'losses': losses
        })
        
        print(f"Win rate: {win_rate:.3f} ({wins}/30)")
        
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_config = config
    
    print(f"\nBest configuration:")
    print(f"Iterations: {best_config['iterations']}")
    print(f"Exploration constant: {best_config['exploration_constant']}")
    print(f"Win rate: {best_win_rate:.3f}")
    
    return best_config, results


def progressive_training():
    """Test MCTS performance with increasing iterations."""
    print("\nProgressive Training")
    print("=" * 30)
    
    iteration_counts = [100, 300, 500, 1000, 1500, 2000]
    random_agent = RandomAgent(player=2)
    
    for iterations in iteration_counts:
        print(f"\nTesting with {iterations} iterations...")
        
        mcts_agent = MCTSAgent(player=1, iterations=iterations)
        
        start_time = time.time()
        wins, draws, losses = evaluate_agent(mcts_agent, random_agent, num_games=20)
        end_time = time.time()
        
        win_rate = wins / 20
        avg_time = (end_time - start_time) / 20
        
        print(f"Win rate: {win_rate:.3f} ({wins}/20)")
        print(f"Average time per game: {avg_time:.3f}s")


def self_play_training(num_games=500):
    """Train through self-play."""
    print(f"\nSelf-Play Training ({num_games} games)")
    print("=" * 30)
    
    agent1 = MCTSAgent(player=1, iterations=1000)
    agent2 = MCTSAgent(player=2, iterations=1000)
    
    stats = {'agent1_wins': 0, 'agent2_wins': 0, 'draws': 0}
    
    for game_num in range(num_games):
        if game_num % 100 == 0:
            print(f"Game {game_num}/{num_games}")
        
        winner = play_game_with_mcts(agent1, agent2, display=False)
        
        if winner == 1:
            stats['agent1_wins'] += 1
        elif winner == 2:
            stats['agent2_wins'] += 1
        else:
            stats['draws'] += 1
        
        agent1.reset()
        agent2.reset()
    
    total_games = num_games
    print(f"\nSelf-play results:")
    print(f"Agent 1 wins: {stats['agent1_wins']/total_games:.3f} ({stats['agent1_wins']})")
    print(f"Agent 2 wins: {stats['agent2_wins']/total_games:.3f} ({stats['agent2_wins']})")
    print(f"Draws: {stats['draws']/total_games:.3f} ({stats['draws']})")


def adaptive_training(target_win_rate=0.8):
    """Find minimum iterations needed for target performance."""
    print(f"\nAdaptive Training (target: {target_win_rate})")
    print("=" * 30)
    
    current_iterations = 100
    max_iterations = 5000
    random_agent = RandomAgent(player=2)
    
    while current_iterations <= max_iterations:
        print(f"\nTesting with {current_iterations} iterations...")
        
        mcts_agent = MCTSAgent(player=1, iterations=current_iterations)
        
        wins, draws, losses = evaluate_agent(mcts_agent, random_agent, num_games=25)
        win_rate = wins / 25
        
        print(f"Win rate: {win_rate:.3f} ({wins}/25)")
        
        if win_rate >= target_win_rate:
            print(f"\nTarget achieved with {current_iterations} iterations!")
            return current_iterations
        
        current_iterations = int(current_iterations * 1.5)
    
    print(f"\nTarget not achieved within {max_iterations} iterations")
    return max_iterations


def evaluate_agent(agent, opponent, num_games=50):
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


def save_agent(agent, filename):
    """Save agent parameters to file."""
    agent_data = {
        'player': agent.player,
        'iterations': agent.iterations,
        'exploration_constant': agent.exploration_constant
    }
    
    with open(filename, 'w') as f:
        json.dump(agent_data, f, indent=2)
    
    print(f"Agent saved to {filename}")


def load_agent(filename):
    """Load agent parameters from file."""
    with open(filename, 'r') as f:
        agent_data = json.load(f)
    
    agent = MCTSAgent(
        player=agent_data['player'],
        iterations=agent_data['iterations'],
        exploration_constant=agent_data['exploration_constant']
    )
    
    print(f"Agent loaded from {filename}")
    return agent


def demonstrate_trained_agent(agent, num_games=10):
    """Demonstrate the trained agent against a random opponent."""
    print(f"\nDemonstrating trained agent ({num_games} games)")
    print("=" * 30)
    
    random_agent = RandomAgent(player=2)
    wins, draws, losses = evaluate_agent(agent, random_agent, num_games)
    
    print(f"Results:")
    print(f"Wins: {wins} ({wins/num_games:.1%})")
    print(f"Draws: {draws} ({draws/num_games:.1%})")
    print(f"Losses: {losses} ({losses/num_games:.1%})")


if __name__ == "__main__":
    print("Simple MCTS Training")
    print("=" * 50)
    
    # 1. Parameter optimization
    best_config, results = optimize_parameters()
    
    # 2. Progressive training
    progressive_training()
    
    # 3. Adaptive training
    optimal_iterations = adaptive_training(target_win_rate=0.8)
    
    # 4. Self-play training
    self_play_training(num_games=200)
    
    # 5. Create and save optimized agent
    optimized_agent = MCTSAgent(
        player=1,
        iterations=best_config['iterations'],
        exploration_constant=best_config['exploration_constant']
    )
    
    save_agent(optimized_agent, 'trained_mcts_agent.json')
    
    # 6. Demonstrate the trained agent
    demonstrate_trained_agent(optimized_agent)
    
    print("\nTraining completed!")
    print("You can now use the trained agent by loading 'trained_mcts_agent.json'") 