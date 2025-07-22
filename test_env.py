from tictactoe_env import TicTacToeEnv, Player, GameStatus

def test_basic_game():
    """Test a basic game scenario"""
    print("Testing basic game...")
    env = TicTacToeEnv()
    
    # Initial state
    print("Initial state:")
    env.render()
    print(f"Valid actions: {env.get_valid_actions()}")
    print()
    
    # Make some moves
    moves = [4, 0, 1, 2, 6]  # Center, top-left, top-center, top-right, bottom-left
    
    for i, move in enumerate(moves):
        print(f"Move {i+1}: Player places at position {move}")
        state, reward, done, info = env.step(move)
        env.render()
        print(f"Reward: {reward}, Done: {done}")
        print(f"Valid actions: {info['valid_actions']}")
        print()
        
        if done:
            break

def test_winning_scenario():
    """Test a winning scenario"""
    print("Testing winning scenario...")
    env = TicTacToeEnv()
    
    # X wins with diagonal
    moves = [0, 1, 4, 2, 8]  # X: 0,4,8 (diagonal), O: 1,2
    
    for i, move in enumerate(moves):
        print(f"Move {i+1}: Position {move}")
        state, reward, done, info = env.step(move)
        env.render()
        print(f"Reward: {reward}, Done: {done}")
        print()
        
        if done:
            winner = env.get_winner()
            print(f"Winner: {winner.name if winner else 'None'}")
            break

def test_draw_scenario():
    """Test a draw scenario"""
    print("Testing draw scenario...")
    env = TicTacToeEnv()
    
    # Draw scenario
    moves = [0, 1, 2, 4, 3, 5, 6, 7, 8]
    
    for i, move in enumerate(moves):
        print(f"Move {i+1}: Position {move}")
        state, reward, done, info = env.step(move)
        env.render()
        print(f"Reward: {reward}, Done: {done}")
        print()
        
        if done:
            winner = env.get_winner()
            print(f"Winner: {winner.name if winner else 'Draw'}")
            break

def test_invalid_action():
    """Test invalid action handling"""
    print("Testing invalid action...")
    env = TicTacToeEnv()
    
    # Make a valid move first
    env.step(4)
    print("After valid move at position 4:")
    env.render()
    
    # Try invalid move
    try:
        env.step(4)  # Position already taken
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"Correctly caught invalid action: {e}")

if __name__ == "__main__":
    test_basic_game()
    print("=" * 50)
    test_winning_scenario()
    print("=" * 50)
    test_draw_scenario()
    print("=" * 50)
    test_invalid_action() 