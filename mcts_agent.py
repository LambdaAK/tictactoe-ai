import numpy as np
import random
import math
from typing import List, Tuple, Optional, Dict, Any
from tictactoe import TicTacToe
import copy

class MCTSNode:
    """Node in the Monte Carlo Tree Search tree."""
    
    def __init__(self, state: str, parent=None, action=None, player: int = 1):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this node
        self.player = player  # Player who made the move to reach this state
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None  # Will be set when needed
        self.is_terminal = False
        self.terminal_value = 0.0
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been explored."""
        if self.untried_actions is None:
            self._initialize_untried_actions()
        return len(self.untried_actions) == 0
    
    def _initialize_untried_actions(self):
        """Initialize the list of untried actions for this node."""
        game = TicTacToe()
        game.board = np.array([int(self.state[i]) for i in range(9)]).reshape(3, 3)
        game.current_player = 3 - self.player  # Next player to move
        game.game_over = False
        game.winner = None
        
        # Check if this is a terminal state
        reward = game._check_game_end()
        if game.game_over:
            self.is_terminal = True
            self.terminal_value = reward
            self.untried_actions = []
        else:
            self.untried_actions = game.get_valid_actions()
    
    def get_ucb_value(self, exploration_constant: float = 1.414) -> float:
        """Calculate UCB1 value for this node."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def select_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """Select the best child node using UCB1."""
        return max(self.children.values(), 
                  key=lambda child: child.get_ucb_value(exploration_constant))
    
    def expand(self) -> 'MCTSNode':
        """Expand the tree by creating a new child node."""
        if self.untried_actions is None:
            self._initialize_untried_actions()
        
        if len(self.untried_actions) == 0:
            return self
        
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        
        # Create new game state
        game = TicTacToe()
        game.board = np.array([int(self.state[i]) for i in range(9)]).reshape(3, 3)
        game.current_player = 3 - self.player
        
        # Make the move
        state, reward, game_over = game.make_move(action)
        
        # Create child node
        child = MCTSNode(state, parent=self, action=action, player=3 - self.player)
        if game_over:
            child.is_terminal = True
            child.terminal_value = reward
        
        self.children[action] = child
        return child
    
    def simulate(self) -> float:
        """Simulate a random game from this state to completion."""
        game = TicTacToe()
        game.board = np.array([int(self.state[i]) for i in range(9)]).reshape(3, 3)
        game.current_player = 3 - self.player
        
        while not game.game_over:
            valid_actions = game.get_valid_actions()
            if len(valid_actions) == 0:
                break
            action = random.choice(valid_actions)
            state, reward, game_over = game.make_move(action)
        
        # Return reward from perspective of the player who made the move to reach this node
        if game.winner == 0:  # Draw
            return 0.0
        elif game.winner == self.player:
            return 1.0
        else:
            return -1.0
    
    def backpropagate(self, reward: float):
        """Backpropagate the simulation result up the tree."""
        node = self
        while node is not None:
            node.visits += 1
            node.value += reward
            reward = -reward  # Flip reward for opponent
            node = node.parent


class MCTSAgent:
    """Monte Carlo Tree Search agent for Tic-tac-toe."""
    
    def __init__(self, player: int = 1, iterations: int = 1000, exploration_constant: float = 1.414):
        self.player = player
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.root = None
    
    def get_action(self, game: TicTacToe) -> int:
        """Get the best action using MCTS."""
        state = game.get_state()
        
        # Check if game is already over
        if game.game_over:
            return None
        
        # Create or update root node
        if self.root is None or self.root.state != state:
            self.root = MCTSNode(state, player=3 - game.current_player)
        else:
            # Update root to current state (if opponent moved)
            if state in self.root.children:
                self.root = self.root.children[state]
            else:
                self.root = MCTSNode(state, player=3 - game.current_player)
        
        # Run MCTS iterations
        for _ in range(self.iterations):
            node = self.root
            
            # Selection: traverse tree until we find a node that can be expanded
            while node.is_fully_expanded() and not node.is_terminal:
                node = node.select_child(self.exploration_constant)
            
            # Expansion: expand the node if it's not terminal
            if not node.is_terminal:
                node = node.expand()
            
            # Simulation: simulate a random game from this node
            if node.is_terminal:
                reward = node.terminal_value
            else:
                reward = node.simulate()
            
            # Backpropagation: update node statistics
            node.backpropagate(reward)
        
        # Choose the action with the most visits
        if not self.root.children:
            # If no children were created (e.g., terminal state), return None
            return None
            
        best_action = max(self.root.children.keys(), 
                         key=lambda action: self.root.children[action].visits)
        
        return best_action
    
    def reset(self):
        """Reset the agent's internal state."""
        self.root = None


def play_game_with_mcts(player1_agent, player2_agent, display=True):
    """Play a game between two agents."""
    game = TicTacToe()
    agents = {1: player1_agent, 2: player2_agent}
    
    if display:
        print("Starting new game!")
        print("Player 1: X, Player 2: O")
        print()
    
    while not game.game_over:
        if display:
            game.display()
            print(f"Player {game.current_player}'s turn")
        
        agent = agents[game.current_player]
        action = agent.get_action(game)
        
        state, reward, game_over = game.make_move(action)
        
        if display:
            print(f"Player {game.current_player} chose position {action}")
            print()
    
    if display:
        game.display()
        if game.winner == 0:
            print("It's a draw!")
        else:
            print(f"Player {game.winner} wins!")
        print()
    
    return game.winner


def evaluate_mcts_agent(mcts_agent, opponent_agent, num_games=100):
    """Evaluate MCTS agent against another agent."""
    wins = 0
    draws = 0
    losses = 0
    
    for i in range(num_games):
        if i % 20 == 0:
            print(f"Playing game {i+1}/{num_games}")
        
        # MCTS plays as player 1
        winner = play_game_with_mcts(mcts_agent, opponent_agent, display=False)
        
        if winner == 1:
            wins += 1
        elif winner == 2:
            losses += 1
        else:
            draws += 1
        
        # Reset agents
        mcts_agent.reset()
        if hasattr(opponent_agent, 'reset'):
            opponent_agent.reset()
    
    print(f"\nMCTS Agent Results (vs {opponent_agent.__class__.__name__}):")
    print(f"Wins: {wins} ({wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print(f"Losses: {losses} ({losses/num_games*100:.1f}%)")
    
    return wins, draws, losses


class RandomAgent:
    """Simple random agent for comparison."""
    
    def __init__(self, player: int = 2):
        self.player = player
    
    def get_action(self, game: TicTacToe) -> int:
        valid_actions = game.get_valid_actions()
        return random.choice(valid_actions)
    
    def reset(self):
        pass


if __name__ == "__main__":
    # Example usage
    print("Monte Carlo Tree Search for Tic-tac-toe")
    print("=" * 40)
    
    # Create agents
    mcts_agent = MCTSAgent(player=1, iterations=1000)
    random_agent = RandomAgent(player=2)
    
    # Play a single game
    print("Playing a single game:")
    play_game_with_mcts(mcts_agent, random_agent, display=True)
    
    # Evaluate MCTS against random agent
    print("\nEvaluating MCTS agent against random agent:")
    evaluate_mcts_agent(mcts_agent, random_agent, num_games=50) 