import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tictactoe import TicTacToe
except ImportError:
    from ..tictactoe import TicTacToe

from qlearning_agent import QLearningAgent, random_agent

def evaluate_agent_vs_random(q_table_file: str, games: int = 1000, agent_player: int = 1):
    """
    Evaluate a trained Q-learning agent against random opponents
    
    Args:
        q_table_file: Path to the saved Q-table file
        games: Number of games to play
        agent_player: Which player the agent should be (1 or 2)
    """
    game = TicTacToe()
    agent = QLearningAgent(player=agent_player)
    
    try:
        agent.load_q_table(q_table_file)
        print(f"Loaded Q-table from {q_table_file}")
    except FileNotFoundError:
        print(f"Q-table file {q_table_file} not found. Please train an agent first.")
        return
    
    wins = {1: 0, 2: 0, 0: 0}
    
    print(f"Evaluating agent (player {agent_player}) against random opponent over {games} games...")
    
    for game_num in range(games):
        state = game.reset()
        
        while not game.game_over:
            valid_actions = game.get_valid_actions()
            
            if game.current_player == agent.player:
                action = agent.get_action(state, valid_actions, training=False)
            else:
                action = random_agent(valid_actions)
            
            state, reward, done = game.make_move(action)
        
        wins[game.winner] += 1
        
        # Show progress every 100 games
        if (game_num + 1) % 100 == 0:
            progress = (game_num + 1) / games * 100
            agent_wins = wins[agent.player]
            print(f"Progress: {progress:.0f}% - Agent wins so far: {agent_wins}/{game_num + 1} ({agent_wins/(game_num + 1)*100:.1f}%)")
    
    # Final results
    agent_wins = wins[agent.player]
    random_wins = wins[3 - agent.player] if agent.player in [1, 2] else wins[1] + wins[2]
    draws = wins[0]
    
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS ({games} games)")
    print(f"{'='*50}")
    print(f"Agent (Player {agent_player}) wins: {agent_wins} ({agent_wins/games*100:.1f}%)")
    print(f"Random opponent wins: {random_wins} ({random_wins/games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/games*100:.1f}%)")
    
    # Performance assessment
    win_rate = agent_wins / games * 100
    if win_rate >= 95:
        performance = "Excellent"
    elif win_rate >= 85:
        performance = "Very Good"
    elif win_rate >= 70:
        performance = "Good"
    elif win_rate >= 60:
        performance = "Average"
    else:
        performance = "Poor"
    
    print(f"Performance: {performance}")
    
    return agent_wins, random_wins, draws

def evaluate_both_agents(games: int = 1000):
    """Evaluate both trained agents"""
    print("Evaluating Agent 1 (X) vs Random:")
    evaluate_agent_vs_random("agent1_qtable.pkl", games, agent_player=1)
    
    print("\n" + "="*60 + "\n")
    
    print("Evaluating Agent 2 (O) vs Random:")
    evaluate_agent_vs_random("agent2_qtable.pkl", games, agent_player=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Q-learning agent against random player")
    parser.add_argument("--qtable", "-q", default="agent1_qtable.pkl", 
                       help="Path to Q-table file (default: agent1_qtable.pkl)")
    parser.add_argument("--games", "-g", type=int, default=1000,
                       help="Number of games to play (default: 1000)")
    parser.add_argument("--player", "-p", type=int, choices=[1, 2], default=1,
                       help="Agent player number (1=X, 2=O, default: 1)")
    parser.add_argument("--both", "-b", action="store_true",
                       help="Evaluate both trained agents")
    
    args = parser.parse_args()
    
    if args.both:
        evaluate_both_agents(args.games)
    else:
        evaluate_agent_vs_random(args.qtable, args.games, args.player)