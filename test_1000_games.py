#!/usr/bin/env python3
"""
Test MCTS agent against random opponent for 1000 games
"""

import time
import json
from mcts_agent import MCTSAgent, RandomAgent, play_game_with_mcts
from tictactoe import TicTacToe


def test_1000_games():
    """Test MCTS agent against random opponent for 1000 games."""
    print("Testing MCTS Agent vs Random Agent (1000 games)")
    print("=" * 50)
    
    # Create agents
    mcts_agent = MCTSAgent(player=1, iterations=1000, exploration_constant=1.414)
    random_agent = RandomAgent(player=2)
    
    # Statistics
    wins = 0
    draws = 0
    losses = 0
    game_lengths = []
    mcts_decision_times = []
    
    start_time = time.time()
    
    for game_num in range(1000):
        if game_num % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Game {game_num}/1000 (Elapsed: {elapsed:.1f}s)")
        
        # Play a single game
        game = TicTacToe()
        moves = 0
        
        while not game.game_over:
            if game.current_player == 1:  # MCTS turn
                move_start = time.time()
                action = mcts_agent.get_action(game)
                move_time = time.time() - move_start
                mcts_decision_times.append(move_time)
            else:  # Random turn
                action = random_agent.get_action(game)
            
            state, reward, game_over = game.make_move(action)
            moves += 1
        
        game_lengths.append(moves)
        
        # Record result
        if game.winner == 1:
            wins += 1
        elif game.winner == 2:
            losses += 1
        else:
            draws += 1
        
        # Reset agents
        mcts_agent.reset()
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    win_rate = wins / 1000
    draw_rate = draws / 1000
    loss_rate = losses / 1000
    avg_game_length = sum(game_lengths) / len(game_lengths)
    avg_decision_time = sum(mcts_decision_times) / len(mcts_decision_times)
    
    # Print results
    print(f"\nResults (1000 games):")
    print(f"=" * 30)
    print(f"Wins: {wins} ({win_rate:.1%})")
    print(f"Draws: {draws} ({draw_rate:.1%})")
    print(f"Losses: {losses} ({loss_rate:.1%})")
    print(f"Win rate: {win_rate:.3f}")
    print(f"Average game length: {avg_game_length:.1f} moves")
    print(f"Average MCTS decision time: {avg_decision_time:.3f}s")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per game: {total_time/1000:.3f}s")
    
    # Performance analysis
    print(f"\nPerformance Analysis:")
    print(f"=" * 30)
    if win_rate > 0.7:
        print("Excellent performance! MCTS is dominating the random agent.")
    elif win_rate > 0.6:
        print("Good performance! MCTS is clearly better than random.")
    elif win_rate > 0.55:
        print("Decent performance. MCTS has a slight advantage.")
    else:
        print("Poor performance. MCTS needs more iterations or better parameters.")
    
    # Game length analysis
    short_games = sum(1 for length in game_lengths if length <= 5)
    medium_games = sum(1 for length in game_lengths if 6 <= length <= 7)
    long_games = sum(1 for length in game_lengths if length >= 8)
    
    print(f"\nGame Length Distribution:")
    print(f"Short games (≤5 moves): {short_games} ({short_games/1000:.1%})")
    print(f"Medium games (6-7 moves): {medium_games} ({medium_games/1000:.1%})")
    print(f"Long games (≥8 moves): {long_games} ({long_games/1000:.1%})")
    
    # Save results
    results = {
        'total_games': 1000,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'loss_rate': loss_rate,
        'avg_game_length': avg_game_length,
        'avg_decision_time': avg_decision_time,
        'total_time': total_time,
        'avg_time_per_game': total_time/1000,
        'mcts_parameters': {
            'iterations': mcts_agent.iterations,
            'exploration_constant': mcts_agent.exploration_constant
        }
    }
    
    with open('mcts_1000_games_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to 'mcts_1000_games_results.json'")
    
    return results


def compare_different_iterations():
    """Compare MCTS performance with different iteration counts."""
    print("\nComparing Different Iteration Counts")
    print("=" * 40)
    
    iteration_counts = [100, 500, 1000, 2000]
    random_agent = RandomAgent(player=2)
    
    for iterations in iteration_counts:
        print(f"\nTesting with {iterations} iterations...")
        
        mcts_agent = MCTSAgent(player=1, iterations=iterations)
        
        start_time = time.time()
        wins, draws, losses = 0, 0, 0
        
        for game_num in range(100):  # Test 100 games for each configuration
            winner = play_game_with_mcts(mcts_agent, random_agent, display=False)
            
            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1
            
            mcts_agent.reset()
        
        end_time = time.time()
        win_rate = wins / 100
        avg_time = (end_time - start_time) / 100
        
        print(f"Win rate: {win_rate:.3f} ({wins}/100)")
        print(f"Average time per game: {avg_time:.3f}s")


if __name__ == "__main__":
    # Test 1000 games
    results = test_1000_games()
    
    # Compare different iteration counts
    compare_different_iterations()
    
    print(f"\nTesting completed!")
    print(f"Check 'mcts_1000_games_results.json' for detailed results.") 