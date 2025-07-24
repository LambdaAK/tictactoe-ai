#!/usr/bin/env python3
"""
Quick Start N-Dimensional Tic-Tac-Toe
Simple gameplay without complex menus
"""

import sys
import os
import random

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tictactoe_nd import TicTacToeND

def quick_play_2d():
    """Quick 2D Tic-Tac-Toe game"""
    print("\n" + "="*40)
    print("QUICK 2D TIC-TAC-TOE")
    print("="*40)
    
    game = TicTacToeND(dimensions=2, board_size=3, win_length=3)
    state = game.reset()
    
    print("You are X (Player 1), AI is O (Player 2)")
    print("Positions: 0-8 (top-left to bottom-right)")
    print()
    
    game.display()
    
    while not game.game_over:
        valid_actions = game.get_valid_actions()
        
        if game.current_player == 1:
            # Human turn
            print(f"Your turn. Valid moves: {valid_actions}")
            print("Board positions:")
            print("0 | 1 | 2")
            print("3 | 4 | 5")
            print("6 | 7 | 8")
            
            while True:
                try:
                    action = int(input("Your move (0-8): "))
                    if action in valid_actions:
                        break
                    print("Invalid move! Choose from valid actions.")
                except ValueError:
                    print("Please enter a number 0-8.")
        else:
            # AI turn (random for now)
            action = random.choice(valid_actions)
            print(f"AI chooses position {action}")
        
        state, reward, done = game.make_move(action)
        game.display()
    
    if game.winner == 1:
        print("You win! 🎉")
    elif game.winner == 2:
        print("AI wins! 🤖")
    else:
        print("It's a draw! 🤝")

def quick_play_3d():
    """Quick 3D Tic-Tac-Toe game"""
    print("\n" + "="*40)
    print("QUICK 3D TIC-TAC-TOE")
    print("="*40)
    
    game = TicTacToeND(dimensions=3, board_size=3, win_length=3)
    state = game.reset()
    
    print("You are X (Player 1), AI is O (Player 2)")
    print("3D board with 27 positions (0-26)")
    print("Layers: 0 (bottom), 1 (middle), 2 (top)")
    print()
    
    game.display()
    
    while not game.game_over:
        valid_actions = game.get_valid_actions()
        
        if game.current_player == 1:
            # Human turn
            print(f"Your turn. Valid moves: {valid_actions}")
            print("3D positions (layer, row, col):")
            for layer in range(3):
                print(f"Layer {layer}:")
                for row in range(3):
                    positions = []
                    for col in range(3):
                        pos = layer * 9 + row * 3 + col
                        if pos in valid_actions:
                            positions.append(f"{pos:2d}")
                        else:
                            positions.append(" X")
                    print(" ".join(positions))
                print()
            
            while True:
                try:
                    action = int(input("Your move (0-26): "))
                    if action in valid_actions:
                        break
                    print("Invalid move! Choose from valid actions.")
                except ValueError:
                    print("Please enter a number 0-26.")
        else:
            # AI turn (random for now)
            action = random.choice(valid_actions)
            print(f"AI chooses position {action}")
        
        state, reward, done = game.make_move(action)
        game.display()
    
    if game.winner == 1:
        print("You win! 🎉")
    elif game.winner == 2:
        print("AI wins! 🤖")
    else:
        print("It's a draw! 🤝")

def main():
    """Main function"""
    print("N-DIMENSIONAL TIC-TAC-TOE - QUICK PLAY")
    print("="*50)
    
    print("\nChoose game:")
    print("1. 2D Tic-Tac-Toe (3x3)")
    print("2. 3D Tic-Tac-Toe (3x3x3)")
    print("3. Exit")
    
    while True:
        try:
            choice = int(input("\nEnter choice (1-3): "))
            if choice == 1:
                quick_play_2d()
            elif choice == 2:
                quick_play_3d()
            elif choice == 3:
                print("Thanks for playing!")
                break
            else:
                print("Invalid choice! Please enter 1-3.")
        except ValueError:
            print("Please enter a number!")
        
        # Ask if user wants to play again
        play_again = input("\nPlay again? (y/n): ").lower().strip()
        if play_again not in ['y', 'yes']:
            print("Thanks for playing!")
            break

if __name__ == "__main__":
    main() 