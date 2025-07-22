# Tic-Tac-Toe AI

This repository is dedicated to experimenting with various AI algorithms for playing Tic-Tac-Toe. Currently, a **Q-Learning** implementation has been developed using tabular methods.

## Overview

Tic-Tac-Toe serves as an excellent testbed for reinforcement learning algorithms due to its:
- Simple state space (9 positions, 2 players)
- Clear win/lose/draw outcomes
- Perfect information game
- Manageable complexity for learning

## Q-Learning Implementation

### Algorithm Overview

Q-Learning is a model-free reinforcement learning algorithm that learns the quality of actions, telling an agent what action to take under what circumstances. It does not require a model of the environment and can handle problems with stochastic transitions and rewards.

### How Q-Learning Works

1. **Q-Table**: A lookup table that stores Q-values for each state-action pair
   - States: Board configurations (represented as strings)
   - Actions: Available moves (0-8 positions)
   - Q-values: Expected future rewards for taking an action in a state

2. **Learning Process**:
   - Agent explores the environment using ε-greedy strategy
   - Updates Q-values using the Bellman equation:


     ```math
     Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
     ```


   - Where:  
     - $\alpha$ (alpha) = learning rate  
     - $\gamma$ (gamma) = discount factor  
     - $r$ = immediate reward  
     - $s'$ = next state


3. **Training Strategy**:
   - **Exploration**: Random moves with probability ε (epsilon)
   - **Exploitation**: Best known moves with probability 1-ε
   - **Epsilon Decay**: Gradually reduce exploration as learning progresses

### Implementation Details

#### Hyperparameters
- **Learning Rate (α)**: 0.1 - Controls how much new information overrides old information
- **Discount Factor (γ)**: 0.95 - How much future rewards are valued vs immediate rewards
- **Initial Epsilon**: 0.8 - Starting exploration rate
- **Epsilon Decay**: 0.9995 - Rate at which exploration decreases
- **Epsilon Minimum**: 0.01 - Minimum exploration rate

#### Reward Structure
- **Win**: +10 points
- **Loss**: -10 points  
- **Draw**: 0 points
- **Intermediate moves**: 0 points (sparse rewards)

#### State Representation
- Board state converted to string representation
- Example: `"X.O.X.O.."` represents a board with X in positions 0,3,5 and O in positions 2,4

## Project Structure

```
├── tictactoe.py          # Core game logic
├── qlearning/
│   ├── qlearning_agent.py    # Q-Learning agent implementation
│   ├── train_agent.py        # Training script with metadata
│   ├── play_game.py          # Interactive gameplay
│   ├── evaluate_agent.py     # Agent evaluation utilities
│   └── agents/               # Saved trained agents
│       ├── x_agent_qtable.pkl
│       └── o_agent_qtable.pkl
```

## Usage

### Training Agents

```bash
cd qlearning
python train_agent.py
```

This will:
- Train X_Agent (Player 1) and O_Agent (Player 2) for 20,000 episodes
- Save agents with metadata to `agents/` directory
- Display training progress and final statistics
- Test trained agents against random players

### Playing Against AI

```bash
cd qlearning
python play_game.py
```

Choose from:
1. **Play against AI**: Human vs trained O_Agent
2. **Watch AI vs AI**: X_Agent vs O_Agent
3. **Exit**

### Loading Pre-trained Agents

```python
from train_agent import load_agent_by_name

# Load specific agents
x_agent = load_agent_by_name("X_Agent")
o_agent = load_agent_by_name("O_Agent")
```

## Performance

The trained agents typically achieve:
- **~90% win rate** against random players
- **~60% draws** when playing against each other
- **Strategic play** including blocking opponent wins and creating winning opportunities

## Key Features

### Agent Metadata
Each saved agent includes comprehensive metadata:
- Training parameters (learning rate, epsilon values, etc.)
- Number of training episodes
- Training date and time
- Final performance statistics

### Flexible Training
- Configurable number of episodes
- Adjustable hyperparameters
- Progress monitoring during training
- Automatic epsilon decay

### Robust Game Logic
- Valid move checking
- Win condition detection
- Draw detection
- Clean board display

## Future Enhancements

Potential areas for expansion:
- **Deep Q-Learning**: Neural network-based Q-function approximation
- **Monte Carlo Tree Search (MCTS)**: Tree-based search algorithms
- **Minimax with Alpha-Beta Pruning**: Traditional game tree search
- **Policy Gradient Methods**: Direct policy optimization
- **Multi-agent Learning**: Competitive and cooperative scenarios

## Technical Notes

### Dependencies
- Python 3.7+
- NumPy
- Standard library modules (pickle, os, sys, collections, datetime)

### File Formats
- Q-tables saved as pickle files with metadata
- Human-readable training logs
- Configurable save/load paths

### Performance Considerations
- Tabular Q-learning scales well for Tic-Tac-Toe's small state space
- Memory usage is minimal (~100KB per agent)
- Training completes in seconds on modern hardware

## Contributing

This repository is designed for educational and experimental purposes. Feel free to:
- Implement new algorithms
- Optimize existing implementations
- Add new evaluation metrics
- Improve documentation
- Share insights and results

## License

This project is open source and available under the MIT License. 