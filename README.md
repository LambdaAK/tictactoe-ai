# TicTacToe Reinforcement Learning Agent

A Python implementation of reinforcement learning agents that learn to play TicTacToe through Q-learning. This project demonstrates how RL algorithms can learn optimal strategies for simple games.

## Project Overview

This project implements:
- **TicTacToe Environment**: A complete game environment with state management, action validation, and reward calculation
- **Base Agent Interface**: An abstract base class for all RL algorithms
- **Q-Learning Agent**: A tabular Q-learning implementation with epsilon-greedy exploration
- **Random Agent**: A baseline random agent for comparison
- **Training System**: A comprehensive trainer that can train agents against different opponents
- **Interactive Game Interface**: Play against trained agents in a user-friendly interface

## Architecture

### Core Components

1. **`tictactoe_env.py`** - Game environment
   - State representation: 3x3 numpy array (0=empty, 1=X, 2=O)
   - Action space: Board positions 0-8 (top-left to bottom-right)
   - Reward function: +1 for win, -1 for loss, 0 for draw, -0.01 per move

2. **`base_agent.py`** - Abstract base class
   - Defines interface for all RL agents
   - Common functionality for state/action key generation
   - Training mode management

3. **`q_learning_agent.py`** - Q-learning implementation
   - Tabular Q-table for state-action values
   - Epsilon-greedy exploration strategy
   - Configurable learning rate, discount factor, and exploration parameters

4. **`random_agent.py`** - Baseline random agent
   - Makes random valid moves
   - Useful for comparison and testing

5. **`train_agent.py`** - Training system
   - Trains agents against random opponents or self-play
   - Tracks training progress and statistics
   - Generates unique filenames for trained agents
   - Creates training progress visualizations

6. **`play_game.py`** - Interactive game interface
   - Play against different types of agents
   - Choose who goes first
   - User-friendly board display

## Quick Start

### Prerequisites
```bash
pip install numpy matplotlib
```

### 1. Train an Agent
```bash
python train_agent.py
```
- Enter number of episodes (e.g., `1000` for quick test, `50000` for strong agent)
- Choose opponent type (`random` or `self`)
- Agent will be saved with a unique filename like `q_agent_random_1000ep_20241201_143022.json`

### 2. Play Against the Trained Agent
```bash
python play_game.py
```
- Choose option 3 (Trained Q-Learning Agent)
- Enter the filename from training (e.g., `q_agent_random_1000ep_20241201_143022.json`)
- Choose who goes first
- Play using board positions 0-8

## How to Play

### Board Positions
```
 0 | 1 | 2 
-----------
 3 | 4 | 5 
-----------
 6 | 7 | 8 
```

### Game Controls
- Enter numbers 0-8 to place your mark
- Type `quit` to exit mid-game
- Press Ctrl+C to interrupt

## Training Details

### Q-Learning Parameters
- **Learning Rate (α)**: 0.1 - How much to update Q-values
- **Discount Factor (γ)**: 0.9 - Importance of future rewards
- **Initial Epsilon (ε)**: 0.3 - Initial exploration rate
- **Epsilon Decay**: 0.9995 - Rate of exploration reduction
- **Minimum Epsilon**: 0.01 - Minimum exploration rate

### Training Strategies
1. **Random Opponent**: Good for learning basic winning patterns
2. **Self-Play**: Creates stronger agents by learning optimal strategies

### Training Progress
The trainer tracks:
- Win rate over time
- Average rewards per episode
- Episode lengths
- Epsilon decay
- Q-table size

## Files Generated

### Trained Agents
- `q_agent_{opponent}_{episodes}ep_{timestamp}.json` - Trained Q-learning agent

### Training Visualizations
- `training_progress_{timestamp}.png` - Training progress plots

## Testing

### Test Environment
```bash
python test_env.py
```
Tests basic game mechanics, winning scenarios, draws, and invalid actions.

### Test Base Agent
```bash
python test_base_agent.py
```
Tests the base agent interface with a mock implementation.

## Project Structure
```
ttt/
├── tictactoe_env.py      # Game environment
├── base_agent.py         # Abstract agent interface
├── q_learning_agent.py   # Q-learning implementation
├── random_agent.py       # Random baseline agent
├── train_agent.py        # Training system
├── play_game.py          # Interactive game interface
├── test_env.py           # Environment tests
├── test_base_agent.py    # Base agent tests
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Agent Performance

### Expected Results
- **Random Agent**: ~33% win rate (baseline)
- **Q-Learning vs Random**: ~80-90% win rate after 10k episodes
- **Q-Learning Self-Play**: ~95%+ win rate after 50k episodes

### Training Time
- 1,000 episodes: ~30 seconds
- 10,000 episodes: ~5 minutes
- 50,000 episodes: ~25 minutes

## Future Enhancements

This project is designed to be easily extensible:

1. **Additional Algorithms**: SARSA, DQN, Policy Gradient
2. **Different Opponents**: Minimax, other trained agents
3. **Larger Boards**: 4x4, 5x5 TicTacToe
4. **Benchmarking**: Compare different algorithms
5. **Web Interface**: Play in browser

## Contributing

Feel free to:
- Add new RL algorithms
- Improve the training system
- Add new game variants
- Enhance the user interface
- Add more comprehensive tests

## License

This project is open source and available under the MIT License.

---

**Happy Learning!** 