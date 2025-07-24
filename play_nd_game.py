#!/usr/bin/env python3
"""
Interactive N-dimensional Tic-Tac-Toe Game
Play against trained agents or watch them play each other
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tictactoe_nd import TicTacToeND, TicTacToe2D, TicTacToe3D, TicTacToe4D

def get_game_config():
    """Let user choose game configuration"""
    print("\n" + "="*50)
    print("N-DIMENSIONAL TIC-TAC-TOE GAME")
    print("="*50)
    
    print("\nChoose game configuration:")
    print("1. 2D 3x3 (Classic Tic-Tac-Toe)")
    print("2. 2D 4x4")
    print("3. 3D 3x3x3")
    print("4. 2D 5x5")
    print("5. 3D 4x4x4")
    print("6. 4D 3x3x3x3")
    print("7. Custom configuration")
    
    while True:
        try:
            choice = int(input("\nEnter choice (1-7): "))
            if choice == 1:
                return {'dimensions': 2, 'board_size': 3, 'win_length': 3}
            elif choice == 2:
                return {'dimensions': 2, 'board_size': 4, 'win_length': 3}
            elif choice == 3:
                return {'dimensions': 3, 'board_size': 3, 'win_length': 3}
            elif choice == 4:
                return {'dimensions': 2, 'board_size': 5, 'win_length': 3}
            elif choice == 5:
                return {'dimensions': 3, 'board_size': 4, 'win_length': 3}
            elif choice == 6:
                return {'dimensions': 4, 'board_size': 3, 'win_length': 3}
            elif choice == 7:
                return get_custom_config()
            else:
                print("Invalid choice! Please enter 1-7.")
        except ValueError:
            print("Please enter a number!")

def get_custom_config():
    """Get custom game configuration from user"""
    print("\nCustom Configuration:")
    
    while True:
        try:
            dimensions = int(input("Number of dimensions (2-5): "))
            if 2 <= dimensions <= 5:
                break
            print("Dimensions must be between 2 and 5!")
        except ValueError:
            print("Please enter a number!")
    
    while True:
        try:
            board_size = int(input("Board size per dimension (3-4): "))
            if 3 <= board_size <= 4:
                break
            print("Board size must be between 3 and 4!")
        except ValueError:
            print("Please enter a number!")
    
    while True:
        try:
            win_length = int(input("Win length (3): "))
            if win_length == 3:
                break
            print("Win length must be 3!")
        except ValueError:
            print("Please enter a number!")
    
    return {'dimensions': dimensions, 'board_size': board_size, 'win_length': win_length}

def get_game_mode():
    """Let user choose game mode"""
    print("\nChoose game mode:")
    print("1. Human vs Random AI")
    print("2. Human vs Q-Learning AI")
    print("3. Human vs DQN AI")
    print("4. Q-Learning vs Random")
    print("5. DQN vs Random")
    print("6. Q-Learning vs DQN")
    print("7. Random vs Random")
    
    while True:
        try:
            choice = int(input("\nEnter choice (1-7): "))
            if 1 <= choice <= 7:
                return choice
            print("Invalid choice! Please enter 1-7.")
        except ValueError:
            print("Please enter a number!")

def get_human_move(game, valid_actions):
    """Get move from human player"""
    if game.dimensions == 2:
        print(f"\nValid moves: {valid_actions}")
        print("Board positions (2D):")
        for i in range(game.board_size):
            row = []
            for j in range(game.board_size):
                pos = i * game.board_size + j
                if pos in valid_actions:
                    row.append(f"{pos:2d}")
                else:
                    row.append(" X")
            print(" ".join(row))
    else:
        print(f"\nValid moves: {valid_actions}")
        print(f"Enter a number from 0 to {game.total_cells - 1}")
    
    while True:
        try:
            action = int(input(f"Your move: "))
            if action in valid_actions:
                return action
            else:
                print("Invalid move! Choose from valid actions.")
        except ValueError:
            print("Please enter a number!")

def random_agent(valid_actions):
    """Random agent"""
    import random
    return random.choice(valid_actions)

def load_agent(agent_type, game_config, player):
    """Load trained agent if available"""
    try:
        if agent_type == "qlearning":
            from qlearning.qlearning_agent_nd import QLearningAgentND
            agent = QLearningAgentND(player=player, game_config=game_config)
            
            # Try to load trained model
            agents_dir = "agents"
            filename = f"qlearning_{game_config['dimensions']}d_{game_config['board_size']}x{game_config['board_size']}_player_{player}.pkl"
            filepath = os.path.join(agents_dir, filename)
            
            if os.path.exists(filepath):
                agent.load_q_table(filepath)
                print(f"Loaded trained Q-Learning agent from {filename}")
            else:
                print(f"No trained Q-Learning agent found. Using untrained agent.")
            
            return agent
            
        elif agent_type == "dqn":
            from dqn.dqn_agent_nd import DQNAgentND
            agent = DQNAgentND(player=player, game_config=game_config)
            
            # Try to load trained model
            agents_dir = "agents"
            filename = f"dqn_{game_config['dimensions']}d_{game_config['board_size']}x{game_config['board_size']}_player_{player}.pkl"
            filepath = os.path.join(agents_dir, filename)
            
            if os.path.exists(filepath):
                agent.load_model(filepath)
                print(f"Loaded trained DQN agent from {filename}")
            else:
                print(f"No trained DQN agent found. Using untrained agent.")
            
            return agent
            
    except Exception as e:
        print(f"Error loading agent: {e}")
        print("Using random agent instead.")
        return None

def play_game(game_config, mode):
    """Play a game with the given configuration and mode"""
    
    # Create game
    game = TicTacToeND(**game_config)
    board_info = game.get_board_info()
    
    print(f"\n{'='*60}")
    print(f"GAME: {game_config['dimensions']}D {game_config['board_size']}x{game_config['board_size']}")
    print(f"Cells: {board_info['total_cells']}, Winning lines: {board_info['winning_lines_count']}")
    print(f"{'='*60}")
    
    # Initialize agents based on mode
    agent1 = None
    agent2 = None
    
    if mode == 1:  # Human vs Random
        agent2 = random_agent
    elif mode == 2:  # Human vs Q-Learning
        agent2 = load_agent("qlearning", game_config, 2)
    elif mode == 3:  # Human vs DQN
        agent2 = load_agent("dqn", game_config, 2)
    elif mode == 4:  # Q-Learning vs Random
        agent1 = load_agent("qlearning", game_config, 1)
        agent2 = random_agent
    elif mode == 5:  # DQN vs Random
        agent1 = load_agent("dqn", game_config, 1)
        agent2 = random_agent
    elif mode == 6:  # Q-Learning vs DQN
        agent1 = load_agent("qlearning", game_config, 1)
        agent2 = load_agent("dqn", game_config, 2)
    elif mode == 7:  # Random vs Random
        agent1 = random_agent
        agent2 = random_agent
    
    # Game loop
    state = game.reset()
    game.display()
    
    while not game.game_over:
        valid_actions = game.get_valid_actions()
        
        if game.current_player == 1:
            if agent1 is None:
                # Human player
                action = get_human_move(game, valid_actions)
            else:
                # AI player
                if callable(agent1):
                    action = agent1(valid_actions)
                else:
                    action = agent1.get_action(state, valid_actions, training=False)
                print(f"Player 1 (X) chooses position {action}")
        else:
            if agent2 is None:
                # Human player
                action = get_human_move(game, valid_actions)
            else:
                # AI player
                if callable(agent2):
                    action = agent2(valid_actions)
                else:
                    action = agent2.get_action(state, valid_actions, training=False)
                print(f"Player 2 (O) chooses position {action}")
        
        state, reward, done = game.make_move(action)
        game.display()
        
        if not done and (agent1 is not None or agent2 is not None):
            input("Press Enter to continue...")
    
    # Game result
    if game.winner == 1:
        print("Player 1 (X) wins!")
    elif game.winner == 2:
        print("Player 2 (O) wins!")
    else:
        print("It's a draw!")

def main():
    """Main game loop"""
    print("Welcome to N-Dimensional Tic-Tac-Toe!")
    
    while True:
        # Get game configuration
        game_config = get_game_config()
        
        # Get game mode
        mode = get_game_mode()
        
        # Play the game
        play_game(game_config, mode)
        
        # Ask if user wants to play again
        play_again = input("\nPlay again? (y/n): ").lower().strip()
        if play_again not in ['y', 'yes']:
            break
    
    print("\nThanks for playing!")

if __name__ == "__main__":
    main() 