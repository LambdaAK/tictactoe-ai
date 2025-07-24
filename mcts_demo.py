#!/usr/bin/env python3
"""
Demo script for Monte Carlo Tree Search (MCTS) agent in Tic-tac-toe.
This script demonstrates the MCTS algorithm and compares it with other agents.
"""

import time
from mcts_agent import MCTSAgent, RandomAgent, play_game_with_mcts, evaluate_mcts_agent
from tictactoe import TicTacToe
import random
import numpy as np

class MinimaxAgent:
    """Simple minimax agent for comparison."""
    
    def __init__(self, player: int = 2, max_depth: int = 9):
        self.player = player
        self.max_depth = max_depth
    
    def get_action(self, game: TicTacToe) -> int:
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            return None
        
        best_score = float('-inf')
        best_action = valid_actions[0]
        
        for action in valid_actions:
            # Make a copy of the game
            game_copy = TicTacToe()
            game_copy.board = game.board.copy()
            game_copy.current_player = game.current_player
            game_copy.game_over = game.game_over
            game_copy.winner = game.winner
            
            # Make the move
            state, reward, game_over = game_copy.make_move(action)
            
            # Get minimax score
            score = self._minimax(game_copy, self.max_depth, False)
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def _minimax(self, game: TicTacToe, depth: int, is_maximizing: bool) -> float:
        if game.game_over or depth == 0:
            if game.winner == self.player:
                return 1.0
            elif game.winner == 3 - self.player:
                return -1.0
            else:
                return 0.0
        
        valid_actions = game.get_valid_actions()
        
        if is_maximizing:
            best_score = float('-inf')
            for action in valid_actions:
                game_copy = TicTacToe()
                game_copy.board = game.board.copy()
                game_copy.current_player = game.current_player
                game_copy.game_over = game.game_over
                game_copy.winner = game.winner
                
                state, reward, game_over = game_copy.make_move(action)
                score = self._minimax(game_copy, depth - 1, False)
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for action in valid_actions:
                game_copy = TicTacToe()
                game_copy.board = game.board.copy()
                game_copy.current_player = game.current_player
                game_copy.game_over = game.game_over
                game_copy.winner = game.winner
                
                state, reward, game_over = game_copy.make_move(action)
                score = self._minimax(game_copy, depth - 1, True)
                best_score = min(best_score, score)
            return best_score
    
    def reset(self):
        pass


def interactive_game():
    """Play an interactive game against the MCTS agent."""
    print("Interactive Game vs MCTS Agent")
    print("=" * 40)
    print("You are Player 1 (X), MCTS is Player 2 (O)")
    print("Enter positions 0-8 (left to right, top to bottom):")
    print("0 1 2")
    print("3 4 5")
    print("6 7 8")
    print()
    
    game = TicTacToe()
    mcts_agent = MCTSAgent(player=2, iterations=500)
    
    while not game.game_over:
        game.display()
        print(f"Player {game.current_player}'s turn")
        
        if game.current_player == 1:  # Human player
            while True:
                try:
                    action = int(input("Enter your move (0-8): "))
                    if action in game.get_valid_actions():
                        break
                    else:
                        print("Invalid move! Try again.")
                except ValueError:
                    print("Please enter a number between 0 and 8.")
        else:  # MCTS agent
            print("MCTS is thinking...")
            start_time = time.time()
            action = mcts_agent.get_action(game)
            end_time = time.time()
            print(f"MCTS chose position {action} (took {end_time - start_time:.2f}s)")
        
        state, reward, game_over = game.make_move(action)
        print()
    
    game.display()
    if game.winner == 0:
        print("It's a draw!")
    elif game.winner == 1:
        print("You win!")
    else:
        print("MCTS wins!")


def benchmark_mcts():
    """Benchmark MCTS with different iteration counts."""
    print("MCTS Performance Benchmark")
    print("=" * 40)
    
    iteration_counts = [100, 500, 1000, 2000]
    random_agent = RandomAgent(player=2)
    
    for iterations in iteration_counts:
        print(f"\nTesting MCTS with {iterations} iterations:")
        mcts_agent = MCTSAgent(player=1, iterations=iterations)
        
        start_time = time.time()
        wins, draws, losses = evaluate_mcts_agent(mcts_agent, random_agent, num_games=20)
        end_time = time.time()
        
        win_rate = wins / 20 * 100
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Total time: {end_time - start_time:.2f}s")
        print(f"Average time per game: {(end_time - start_time) / 20:.2f}s")


def compare_agents():
    """Compare MCTS against different agents."""
    print("Agent Comparison")
    print("=" * 40)
    
    # Create agents
    mcts_agent = MCTSAgent(player=1, iterations=1000)
    random_agent = RandomAgent(player=2)
    minimax_agent = MinimaxAgent(player=2, max_depth=9)
    
    # Test MCTS vs Random
    print("\n1. MCTS vs Random Agent:")
    evaluate_mcts_agent(mcts_agent, random_agent, num_games=50)
    
    # Test MCTS vs Minimax
    print("\n2. MCTS vs Minimax Agent:")
    evaluate_mcts_agent(mcts_agent, minimax_agent, num_games=20)
    
    # Test Minimax vs Random
    print("\n3. Minimax vs Random Agent:")
    wins = 0
    draws = 0
    losses = 0
    
    for i in range(50):
        if i % 10 == 0:
            print(f"Playing game {i+1}/50")
        
        winner = play_game_with_mcts(minimax_agent, random_agent, display=False)
        
        if winner == 1:
            wins += 1
        elif winner == 2:
            losses += 1
        else:
            draws += 1
    
    print(f"\nMinimax Agent Results (vs Random):")
    print(f"Wins: {wins} ({wins/50*100:.1f}%)")
    print(f"Draws: {draws} ({draws/50*100:.1f}%)")
    print(f"Losses: {losses} ({losses/50*100:.1f}%)")


def demonstrate_mcts_decision():
    """Demonstrate MCTS decision-making process."""
    print("MCTS Decision Demonstration")
    print("=" * 40)
    
    # Create a specific game state
    game = TicTacToe()
    # Set up a scenario where MCTS needs to make a strategic decision
    game.board = np.array([
        [1, 2, 1],
        [0, 2, 0],
        [0, 0, 0]
    ])
    game.current_player = 1
    game.game_over = False
    game.winner = None
    
    print("Current board state:")
    game.display()
    print(f"Player {game.current_player}'s turn")
    print("Available moves:", game.get_valid_actions())
    
    # Create MCTS agent with different iteration counts
    for iterations in [100, 500, 1000]:
        print(f"\nMCTS with {iterations} iterations:")
        mcts_agent = MCTSAgent(player=1, iterations=iterations)
        
        start_time = time.time()
        action = mcts_agent.get_action(game)
        end_time = time.time()
        
        print(f"Chosen action: {action}")
        print(f"Decision time: {end_time - start_time:.3f}s")
        
        # Show visit counts for each child
        if mcts_agent.root and mcts_agent.root.children:
            print("Visit counts for each action:")
            for action, child in mcts_agent.root.children.items():
                win_rate = child.value / child.visits if child.visits > 0 else 0
                print(f"  Action {action}: {child.visits} visits, win rate: {win_rate:.3f}")


if __name__ == "__main__":
    print("Monte Carlo Tree Search (MCTS) Demo for Tic-tac-toe")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. Play interactive game vs MCTS")
        print("2. Benchmark MCTS performance")
        print("3. Compare agents")
        print("4. Demonstrate MCTS decision-making")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            interactive_game()
        elif choice == '2':
            benchmark_mcts()
        elif choice == '3':
            compare_agents()
        elif choice == '4':
            demonstrate_mcts_decision()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.") 