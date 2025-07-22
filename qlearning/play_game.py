import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tictactoe import TicTacToe
except ImportError:
    from ..tictactoe import TicTacToe

from qlearning_agent import QLearningAgent

def play_human_vs_agent():
    game = TicTacToe()
    agent = QLearningAgent(player=2)
    agent.load_q_table("agent2_qtable.pkl")
    
    print("Welcome to Tic-Tac-Toe!")
    print("You are X (player 1), AI is O (player 2)")
    print("Enter positions 0-8 (top-left to bottom-right):")
    print("0 | 1 | 2")
    print("3 | 4 | 5") 
    print("6 | 7 | 8")
    print()
    
    state = game.reset()
    game.display()
    
    while not game.game_over:
        valid_actions = game.get_valid_actions()
        
        if game.current_player == 1:
            while True:
                try:
                    action = int(input(f"Your turn. Valid moves: {valid_actions}: "))
                    if action in valid_actions:
                        break
                    else:
                        print("Invalid move! Choose an empty position.")
                except ValueError:
                    print("Please enter a number 0-8.")
        else:
            action = agent.get_action(state, valid_actions, training=False)
            print(f"AI chooses position {action}")
        
        state, reward, done = game.make_move(action)
        game.display()
    
    if game.winner == 1:
        print("Congratulations! You won!")
    elif game.winner == 2:
        print("AI wins!")
    else:
        print("It's a draw!")

def play_agent_vs_agent():
    game = TicTacToe()
    agent1 = QLearningAgent(player=1)
    agent2 = QLearningAgent(player=2)
    
    agent1.load_q_table("agent1_qtable.pkl")
    agent2.load_q_table("agent2_qtable.pkl")
    
    print("Agent vs Agent gameplay:")
    state = game.reset()
    game.display()
    
    while not game.game_over:
        valid_actions = game.get_valid_actions()
        
        if game.current_player == 1:
            action = agent1.get_action(state, valid_actions, training=False)
            print(f"Agent 1 (X) chooses position {action}")
        else:
            action = agent2.get_action(state, valid_actions, training=False)
            print(f"Agent 2 (O) chooses position {action}")
        
        state, reward, done = game.make_move(action)
        game.display()
        
        if not done:
            input("Press Enter to continue...")
    
    if game.winner == 1:
        print("Agent 1 (X) wins!")
    elif game.winner == 2:
        print("Agent 2 (O) wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    while True:
        print("\n1. Play against AI")
        print("2. Watch AI vs AI")
        print("3. Exit")
        
        choice = input("Choose option (1-3): ")
        
        if choice == "1":
            play_human_vs_agent()
        elif choice == "2":
            play_agent_vs_agent()
        elif choice == "3":
            break
        else:
            print("Invalid choice!")