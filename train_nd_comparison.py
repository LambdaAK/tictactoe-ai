import sys
import os
import time
from collections import defaultdict

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tictactoe_nd import TicTacToeND, TicTacToe2D, TicTacToe3D, TicTacToe4D
from qlearning.qlearning_agent_nd import QLearningAgentND, random_agent
from dqn.dqn_agent_nd import DQNAgentND

def train_and_compare(dimensions: int, board_size: int = 3, episodes: int = 10000):
    """Train and compare Q-Learning vs DQN for given dimensions"""
    
    print(f"\n{'='*60}")
    print(f"TRAINING FOR {dimensions}D TIC-TAC-TOE ({board_size}^{dimensions} = {board_size**dimensions} cells)")
    print(f"{'='*60}")
    
    # Game configuration
    game_config = {
        'dimensions': dimensions,
        'board_size': board_size,
        'win_length': 3
    }
    
    # Create game instance
    game = TicTacToeND(**game_config)
    board_info = game.get_board_info()
    print(f"Board Info: {board_info}")
    
    # Train Q-Learning Agent
    print(f"\nTraining Q-Learning Agent...")
    start_time = time.time()
    
    ql_agent = QLearningAgentND(
        player=1,
        game_config=game_config,
        learning_rate=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995
    )
    
    ql_wins = {1: 0, 2: 0, 0: 0}
    
    for episode in range(episodes):
        state = game.reset()
        
        while not game.game_over:
            valid_actions = game.get_valid_actions()
            
            if game.current_player == 1:
                action = ql_agent.get_action(state, valid_actions, training=True)
            else:
                action = random_agent(valid_actions)
            
            next_state, reward, done = game.make_move(action)
            
            # Update Q-Learning agent
            if game.current_player == 2:  # After opponent's move
                ql_agent.update_q_value(state, action, reward, next_state, done)
            
            state = next_state
        
        # Episode finished
        if game.winner is not None:
            ql_wins[game.winner] += 1
        else:
            ql_wins[0] += 1
        
        ql_agent.decay_epsilon()
        
        if episode % 1000 == 0:
            win_rate = ql_wins[1] / (episode + 1) * 100
            print(f"Episode {episode}: Q-Learning wins: {win_rate:.1f}%, Epsilon: {ql_agent.get_epsilon():.3f}")
    
    ql_time = time.time() - start_time
    ql_final_win_rate = ql_wins[1] / episodes * 100
    
    print(f"Q-Learning completed in {ql_time:.2f}s")
    print(f"Final Q-Learning win rate: {ql_final_win_rate:.1f}%")
    
    # Train DQN Agent
    print(f"\nTraining DQN Agent...")
    start_time = time.time()
    
    dqn_agent = DQNAgentND(
        player=1,
        game_config=game_config,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9998,
        hidden_size=256,
        replay_buffer_size=20000,
        batch_size=64,
        target_update_freq=200
    )
    
    dqn_wins = {1: 0, 2: 0, 0: 0}
    
    for episode in range(episodes):
        state = game.reset()
        
        while not game.game_over:
            valid_actions = game.get_valid_actions()
            
            if game.current_player == 1:
                action = dqn_agent.get_action(state, valid_actions, training=True)
            else:
                action = random_agent(valid_actions)
            
            next_state, reward, done = game.make_move(action)
            
            # Store experience for DQN
            if game.current_player == 2:  # After opponent's move
                dqn_agent.store_experience(state, action, reward, next_state, done, valid_actions)
            
            state = next_state
        
        # Episode finished
        if game.winner is not None:
            dqn_wins[game.winner] += 1
        else:
            dqn_wins[0] += 1
        
        # Update DQN network
        dqn_agent.update_q_network()
        
        if episode % 1000 == 0:
            win_rate = dqn_wins[1] / (episode + 1) * 100
            print(f"Episode {episode}: DQN wins: {win_rate:.1f}%, Epsilon: {dqn_agent.get_epsilon():.3f}")
    
    dqn_time = time.time() - start_time
    dqn_final_win_rate = dqn_wins[1] / episodes * 100
    
    print(f"DQN completed in {dqn_time:.2f}s")
    print(f"Final DQN win rate: {dqn_final_win_rate:.1f}%")
    
    # Comparison
    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Board: {dimensions}D ({board_size}^{dimensions} = {board_size**dimensions} cells)")
    print(f"Episodes: {episodes}")
    print(f"Q-Learning: {ql_final_win_rate:.1f}% win rate, {ql_time:.2f}s training time")
    print(f"DQN: {dqn_final_win_rate:.1f}% win rate, {dqn_time:.2f}s training time")
    print(f"Speed ratio (Q-Learning/DQN): {ql_time/dqn_time:.2f}x")
    print(f"Performance difference: {dqn_final_win_rate - ql_final_win_rate:.1f}%")
    
    # Determine practicality
    if board_size**dimensions <= 100:  # Small state space
        practicality = "Q-Learning is more practical (small state space)"
    elif board_size**dimensions <= 1000:  # Medium state space
        practicality = "Both approaches are viable"
    else:  # Large state space
        practicality = "DQN is more practical (large state space)"
    
    print(f"Practicality: {practicality}")
    
    return {
        'dimensions': dimensions,
        'board_size': board_size,
        'total_cells': board_size**dimensions,
        'ql_win_rate': ql_final_win_rate,
        'dqn_win_rate': dqn_final_win_rate,
        'ql_time': ql_time,
        'dqn_time': dqn_time,
        'practicality': practicality
    }

def main():
    """Main function to run comparisons across different dimensions"""
    
    print("N-DIMENSIONAL TIC-TAC-TOE: Q-LEARNING vs DQN COMPARISON")
    print("="*80)
    
    # Test configurations
    configs = [
        (2, 3, 5000),   # 2D 3x3 (9 cells) - Q-Learning should dominate
        (2, 4, 5000),   # 2D 4x4 (16 cells) - Both viable
        (3, 3, 3000),   # 3D 3x3x3 (27 cells) - Both viable
        (2, 5, 3000),   # 2D 5x5 (25 cells) - Both viable
        (3, 4, 2000),   # 3D 4x4x4 (64 cells) - DQN advantage
        (4, 3, 1000),   # 4D 3x3x3x3 (81 cells) - DQN advantage
    ]
    
    results = []
    
    for dimensions, board_size, episodes in configs:
        try:
            result = train_and_compare(dimensions, board_size, episodes)
            results.append(result)
        except Exception as e:
            print(f"Error training {dimensions}D {board_size}x{board_size}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY OF ALL RESULTS")
    print(f"{'='*80}")
    
    for result in results:
        print(f"{result['dimensions']}D ({result['board_size']}^{result['dimensions']} = {result['total_cells']} cells):")
        print(f"  Q-Learning: {result['ql_win_rate']:.1f}% win rate, {result['ql_time']:.2f}s")
        print(f"  DQN: {result['dqn_win_rate']:.1f}% win rate, {result['dqn_time']:.2f}s")
        print(f"  {result['practicality']}")
        print()
    
    print("CONCLUSION:")
    print("- For small state spaces (≤100 cells): Q-Learning is faster and simpler")
    print("- For medium state spaces (100-1000 cells): Both approaches work well")
    print("- For large state spaces (>1000 cells): DQN becomes more practical")
    print("- DQN shows its value when state spaces become too large for tabular methods")

if __name__ == "__main__":
    main() 