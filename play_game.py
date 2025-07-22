import os
import sys
from tictactoe_env import TicTacToeEnv, Player, GameStatus
from q_learning_agent import QLearningAgent
from random_agent import RandomAgent

def print_board_with_positions():
    """Print the board with position numbers for reference."""
    print("\nBoard positions:")
    print("-------------")
    for i in range(3):
        row_str = "|"
        for j in range(3):
            pos = i * 3 + j
            row_str += f" {pos} |"
        print(row_str)
        print("-------------")
    print()

def get_human_move(env):
    """Get move from human player."""
    while True:
        try:
            move = input("Enter your move (0-8): ").strip()
            if move.lower() == 'quit':
                return None
            move = int(move)
            if 0 <= move <= 8 and env._is_valid_action(move):
                return move
            else:
                print("Invalid move! Please enter a number between 0-8 for an empty position.")
        except ValueError:
            print("Please enter a valid number between 0-8.")
        except KeyboardInterrupt:
            print("\nGame interrupted. Goodbye!")
            sys.exit(0)

def play_game(agent_type="random", agent_file=None, human_first=True):
    """
    Play a game against an agent.
    
    Args:
        agent_type: Type of agent ("random", "qlearning", or "trained")
        agent_file: Path to saved agent file (for trained agents)
        human_first: Whether human plays first (X) or second (O)
    """
    env = TicTacToeEnv()
    
    # Create agent
    if agent_type == "random":
        agent = RandomAgent(Player.O if human_first else Player.X, env)
        agent.set_training_mode(False)
    elif agent_type == "qlearning":
        agent = QLearningAgent(Player.O if human_first else Player.X, env)
        agent.set_training_mode(False)
    elif agent_type == "trained":
        if not agent_file or not os.path.exists(agent_file):
            print(f"Error: Trained agent file '{agent_file}' not found!")
            return
        agent = QLearningAgent(Player.O if human_first else Player.X, env)
        agent.load(agent_file)
        agent.set_training_mode(False)
    else:
        print(f"Unknown agent type: {agent_type}")
        return
    
    print(f"\nPlaying against {agent_type} agent")
    print(f"You are playing as {'X' if human_first else 'O'}")
    print("Enter 'quit' to exit the game")
    
    # Game loop
    state = env.reset()
    print_board_with_positions()
    
    while not env.is_done():
        env.render()
        print()
        
        current_player = env.current_player
        valid_actions = env.get_valid_actions()
        
        if current_player == Player.X and human_first or current_player == Player.O and not human_first:
            # Human's turn
            print("Your turn!")
            move = get_human_move(env)
            if move is None:
                print("Game ended by user.")
                return
        else:
            # Agent's turn
            print(f"{agent_type.title()} agent's turn...")
            move = agent.select_action(state, valid_actions)
            print(f"Agent chose position {move}")
        
        # Make the move
        state, reward, done, info = env.step(move)
    
    # Game over
    env.render()
    print("\nGame Over!")
    
    winner = env.get_winner()
    if winner:
        if (winner == Player.X and human_first) or (winner == Player.O and not human_first):
            print("Congratulations! You won!")
        else:
            print(f"The {agent_type} agent won!")
    else:
        print("It's a draw!")

def main():
    """Main game interface."""
    print("Welcome to TicTacToe vs AI!")
    print("=" * 40)
    
    while True:
        print("\nChoose an opponent:")
        print("1. Random Agent")
        print("2. Untrained Q-Learning Agent")
        print("3. Trained Q-Learning Agent")
        print("4. Quit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            agent_type = "random"
            agent_file = None
        elif choice == "2":
            agent_type = "qlearning"
            agent_file = None
        elif choice == "3":
            agent_type = "trained"
            agent_file = input("Enter path to trained agent file: ").strip()
            if not agent_file:
                print("No file specified, using random agent instead.")
                agent_type = "random"
                agent_file = None
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please enter 1-4.")
            continue
        
        # Choose who goes first
        print("\nWho goes first?")
        print("1. You (X)")
        print("2. Agent (O)")
        
        first_choice = input("Enter your choice (1-2): ").strip()
        human_first = first_choice == "1"
        
        # Play the game
        play_game(agent_type, agent_file, human_first)
        
        # Ask if they want to play again
        play_again = input("\nPlay again? (y/n): ").strip().lower()
        if play_again not in ['y', 'yes']:
            print("Thanks for playing!")
            break

if __name__ == "__main__":
    main() 