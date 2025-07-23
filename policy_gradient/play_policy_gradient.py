import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tictactoe import TicTacToe
except ImportError:
    from ..tictactoe import TicTacToe

from policy_gradient_agent import PolicyGradientAgent
from train_policy_gradient import load_agent_by_name

def play_human_vs_policy_gradient():
    game = TicTacToe()
    
    try:
        agent = load_agent_by_name("PolicyGradient_O_Agent")
    except FileNotFoundError:
        print("Error: PolicyGradient_O_Agent not found. Please train the agents first using train_policy_gradient.py")
        return
    
    print("Welcome to Tic-Tac-Toe vs Policy Gradient AI!")
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
            print(f"Policy Gradient AI ({agent.name}) chooses position {action}")
        
        state, reward, done = game.make_move(action)
        game.display()
    
    if game.winner == 1:
        print("Congratulations! You won!")
    elif game.winner == 2:
        print("Policy Gradient AI wins!")
    else:
        print("It's a draw!")

def play_policy_gradient_vs_policy_gradient():
    game = TicTacToe()
    
    try:
        agent1 = load_agent_by_name("PolicyGradient_X_Agent")
        agent2 = load_agent_by_name("PolicyGradient_O_Agent")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please train the agents first using train_policy_gradient.py")
        return
    
    print("Policy Gradient vs Policy Gradient gameplay:")
    state = game.reset()
    game.display()
    
    while not game.game_over:
        valid_actions = game.get_valid_actions()
        
        if game.current_player == 1:
            action = agent1.get_action(state, valid_actions, training=False)
            print(f"{agent1.name} (X) chooses position {action}")
        else:
            action = agent2.get_action(state, valid_actions, training=False)
            print(f"{agent2.name} (O) chooses position {action}")
        
        state, reward, done = game.make_move(action)
        game.display()
        
        if not done:
            input("Press Enter to continue...")
    
    if game.winner == 1:
        print(f"{agent1.name} (X) wins!")
    elif game.winner == 2:
        print(f"{agent2.name} (O) wins!")
    else:
        print("It's a draw!")

def play_policy_gradient_vs_qlearning():
    game = TicTacToe()
    
    try:
        pg_agent = load_agent_by_name("PolicyGradient_X_Agent")
        from qlearning.train_agent import load_agent_by_name as load_q_agent
        q_agent = load_q_agent("X_Agent")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please train both Policy Gradient and Q-Learning agents first.")
        return
    
    print("Policy Gradient vs Q-Learning gameplay:")
    print(f"{pg_agent.name} (X) vs {q_agent.name} (O)")
    state = game.reset()
    game.display()
    
    while not game.game_over:
        valid_actions = game.get_valid_actions()
        
        if game.current_player == 1:
            action = pg_agent.get_action(state, valid_actions, training=False)
            print(f"{pg_agent.name} (X) chooses position {action}")
        else:
            action = q_agent.get_action(state, valid_actions, training=False)
            print(f"{q_agent.name} (O) chooses position {action}")
        
        state, reward, done = game.make_move(action)
        game.display()
        
        if not done:
            input("Press Enter to continue...")
    
    if game.winner == 1:
        print(f"{pg_agent.name} (X) wins!")
    elif game.winner == 2:
        print(f"{q_agent.name} (O) wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    while True:
        print("\nPolicy Gradient Tic-Tac-Toe")
        print("1. Play against Policy Gradient AI")
        print("2. Watch Policy Gradient vs Policy Gradient")
        print("3. Watch Policy Gradient vs Q-Learning")
        print("4. Exit")
        
        choice = input("Choose option (1-4): ")
        
        if choice == "1":
            play_human_vs_policy_gradient()
        elif choice == "2":
            play_policy_gradient_vs_policy_gradient()
        elif choice == "3":
            play_policy_gradient_vs_qlearning()
        elif choice == "4":
            break
        else:
            print("Invalid choice!") 