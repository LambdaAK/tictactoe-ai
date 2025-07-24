#!/usr/bin/env python3
"""
Demo script for N-dimensional Tic-Tac-Toe
Shows different board configurations and their complexity
"""

from tictactoe_nd import TicTacToeND, TicTacToe2D, TicTacToe3D, TicTacToe4D

def demo_board_configurations():
    """Demonstrate different board configurations"""
    
    print("N-DIMENSIONAL TIC-TAC-TOE DEMO")
    print("="*50)
    
    # Test different configurations
    configs = [
        (2, 3, 3),  # 2D 3x3
        (2, 4, 3),  # 2D 4x4
        (3, 3, 3),  # 3D 3x3x3
        (2, 5, 3),  # 2D 5x5
        (3, 4, 3),  # 3D 4x4x4
        (4, 3, 3),  # 4D 3x3x3x3
        (5, 3, 3),  # 5D 3x3x3x3x3
    ]
    
    for dimensions, board_size, win_length in configs:
        print(f"\n{'='*30}")
        print(f"{dimensions}D Tic-Tac-Toe ({board_size}^{dimensions})")
        print(f"{'='*30}")
        
        try:
            game = TicTacToeND(dimensions=dimensions, board_size=board_size, win_length=win_length)
            info = game.get_board_info()
            
            print(f"Dimensions: {info['dimensions']}")
            print(f"Board size: {info['board_size']}")
            print(f"Win length: {info['win_length']}")
            print(f"Total cells: {info['total_cells']}")
            print(f"Winning lines: {info['winning_lines_count']}")
            
            # Show initial board
            print(f"\nInitial board state:")
            game.display()
            
            # Make a few moves to show gameplay
            print("Making some moves...")
            for i in range(min(3, info['total_cells'] // 2)):
                if game.game_over:
                    break
                    
                valid_actions = game.get_valid_actions()
                if valid_actions:
                    action = valid_actions[0]  # Take first valid action
                    state, reward, done = game.make_move(action)
                    print(f"Move {i+1}: Player {game.current_player + 1} takes position {action}")
                    game.display()
            
        except Exception as e:
            print(f"Error with {dimensions}D {board_size}x{board_size}: {e}")
            continue

def demo_practicality_analysis():
    """Analyze when DQN becomes practical vs Q-Learning"""
    
    print(f"\n{'='*60}")
    print("PRACTICALITY ANALYSIS: Q-LEARNING vs DQN")
    print(f"{'='*60}")
    
    print("\nState Space Complexity:")
    print("-" * 40)
    
    configs = [
        (2, 3),   # 9 cells
        (2, 4),   # 16 cells
        (3, 3),   # 27 cells
        (2, 5),   # 25 cells
        (3, 4),   # 64 cells
        (4, 3),   # 81 cells
        (2, 6),   # 36 cells
        (3, 5),   # 125 cells
        (4, 4),   # 256 cells
        (5, 3),   # 243 cells
    ]
    
    for dimensions, board_size in configs:
        total_cells = board_size ** dimensions
        
        if total_cells <= 100:
            recommendation = "Q-Learning (small state space)"
        elif total_cells <= 1000:
            recommendation = "Both viable"
        else:
            recommendation = "DQN (large state space)"
        
        print(f"{dimensions}D {board_size}x{board_size}: {total_cells:3d} cells → {recommendation}")
    
    print(f"\n{'='*60}")
    print("KEY INSIGHTS:")
    print(f"{'='*60}")
    print("1. Q-Learning excels for small state spaces (≤100 cells)")
    print("   - Fast training, simple implementation")
    print("   - Perfect for 2D 3x3, 2D 4x4, 3D 3x3x3")
    print()
    print("2. DQN becomes practical for larger state spaces (>100 cells)")
    print("   - Handles high-dimensional spaces efficiently")
    print("   - Better for 3D 4x4x4, 4D+, larger boards")
    print()
    print("3. The crossover point is around 100-1000 cells")
    print("   - Below: Q-Learning is faster and simpler")
    print("   - Above: DQN's neural network approximation helps")
    print()
    print("4. N-dimensional Tic-Tac-Toe demonstrates this perfectly:")
    print("   - 2D 3x3 (9 cells): Q-Learning domain")
    print("   - 4D 3x3x3x3 (81 cells): Both viable")
    print("   - 4D 4x4x4x4 (256 cells): DQN domain")

if __name__ == "__main__":
    demo_board_configurations()
    demo_practicality_analysis() 