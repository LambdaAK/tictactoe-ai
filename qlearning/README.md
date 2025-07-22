# Q-Learning Implementation for Tic-Tac-Toe

This directory contains the Q-learning implementation for training AI agents to play Tic-Tac-Toe.

## Files

- `qlearning_agent.py` - The main Q-learning agent implementation
- `train_agent.py` - Script to train Q-learning agents
- `evaluate_agent.py` - Script to evaluate trained agents against random opponents
- `play_game.py` - Script to play games using trained agents
- `agent1_qtable.pkl` - Saved Q-table for player 1 agent
- `agent2_qtable.pkl` - Saved Q-table for player 2 agent

## Usage

To train agents:
```bash
python train_agent.py
```

To evaluate a trained agent:
```bash
python evaluate_agent.py --agent player1 --games 1000
```

To play a game:
```bash
python play_game.py
```

## Dependencies

- Requires `tictactoe.py` from the parent directory
- Uses numpy for numerical operations
- Uses pickle for saving/loading Q-tables 