#!/usr/bin/env python3
"""
Test script for Monte Carlo Tree Search (MCTS) implementation.
"""

import unittest
import numpy as np
from mcts_agent import MCTSNode, MCTSAgent, RandomAgent, play_game_with_mcts
from tictactoe import TicTacToe


class TestMCTSNode(unittest.TestCase):
    """Test cases for MCTSNode class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.empty_state = "000000000"
        self.node = MCTSNode(self.empty_state, player=1)
    
    def test_node_initialization(self):
        """Test that nodes are initialized correctly."""
        self.assertEqual(self.node.state, self.empty_state)
        self.assertEqual(self.node.player, 1)
        self.assertEqual(self.node.visits, 0)
        self.assertEqual(self.node.value, 0.0)
        self.assertIsNone(self.node.parent)
        self.assertEqual(len(self.node.children), 0)
    
    def test_ucb_calculation(self):
        """Test UCB1 value calculation."""
        # Set up a parent node
        parent = MCTSNode("000000000", player=2)
        parent.visits = 10
        
        child = MCTSNode("100000000", parent=parent, player=1)
        child.visits = 5
        child.value = 3.0
        
        ucb_value = child.get_ucb_value()
        self.assertIsInstance(ucb_value, float)
        self.assertGreater(ucb_value, 0)
    
    def test_untried_actions_initialization(self):
        """Test that untried actions are initialized correctly."""
        self.assertIsNone(self.node.untried_actions)
        self.node._initialize_untried_actions()
        self.assertEqual(len(self.node.untried_actions), 9)  # All positions available
    
    def test_terminal_state_detection(self):
        """Test that terminal states are detected correctly."""
        # Create a winning state for player 1
        winning_state = "111000000"
        node = MCTSNode(winning_state, player=1)
        node._initialize_untried_actions()
        
        self.assertTrue(node.is_terminal)
        self.assertEqual(node.terminal_value, 10)  # Win for player 1


class TestMCTSAgent(unittest.TestCase):
    """Test cases for MCTSAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.game = TicTacToe()
        self.agent = MCTSAgent(player=1, iterations=100)
    
    def test_agent_initialization(self):
        """Test that agent is initialized correctly."""
        self.assertEqual(self.agent.player, 1)
        self.assertEqual(self.agent.iterations, 100)
        self.assertEqual(self.agent.exploration_constant, 1.414)
        self.assertIsNone(self.agent.root)
    
    def test_get_action_empty_board(self):
        """Test getting action on empty board."""
        action = self.agent.get_action(self.game)
        self.assertIn(action, range(9))
        self.assertIsNotNone(self.agent.root)
    
    def test_get_action_partial_board(self):
        """Test getting action on partially filled board."""
        # Make a move
        self.game.make_move(4)  # Center
        action = self.agent.get_action(self.game)
        self.assertIn(action, self.game.get_valid_actions())
    
    def test_reset(self):
        """Test agent reset functionality."""
        self.agent.get_action(self.game)  # Create a root
        self.assertIsNotNone(self.agent.root)
        self.agent.reset()
        self.assertIsNone(self.agent.root)


class TestGamePlay(unittest.TestCase):
    """Test cases for game playing functionality."""
    
    def test_play_game_with_mcts(self):
        """Test that games can be played between agents."""
        mcts_agent = MCTSAgent(player=1, iterations=50)
        random_agent = RandomAgent(player=2)
        
        winner = play_game_with_mcts(mcts_agent, random_agent, display=False)
        self.assertIn(winner, [0, 1, 2])  # Draw, Player 1 win, or Player 2 win
    
    def test_random_agent(self):
        """Test random agent functionality."""
        game = TicTacToe()
        agent = RandomAgent(player=1)
        
        # Test multiple actions
        for _ in range(5):
            action = agent.get_action(game)
            self.assertIn(action, game.get_valid_actions())
            if game.get_valid_actions():
                game.make_move(action)


def run_performance_test():
    """Run a simple performance test."""
    print("Running performance test...")
    
    # Test MCTS vs Random
    mcts_agent = MCTSAgent(player=1, iterations=500)
    random_agent = RandomAgent(player=2)
    
    wins = 0
    total_games = 10
    
    for i in range(total_games):
        winner = play_game_with_mcts(mcts_agent, random_agent, display=False)
        if winner == 1:
            wins += 1
        mcts_agent.reset()
    
    win_rate = wins / total_games * 100
    print(f"MCTS win rate against random: {win_rate:.1f}% ({wins}/{total_games})")
    
    # Should be significantly better than random (which would be ~50%)
    assert win_rate > 60, f"MCTS should perform better than random, got {win_rate}%"


def test_edge_cases():
    """Test edge cases and corner cases."""
    print("Testing edge cases...")
    
    # Test with very few iterations
    agent = MCTSAgent(player=1, iterations=1)
    game = TicTacToe()
    action = agent.get_action(game)
    assert action in range(9), f"Invalid action: {action}"
    
    # Test with terminal state
    game = TicTacToe()
    game.board = np.array([
        [1, 1, 1],
        [2, 2, 0],
        [0, 0, 0]
    ])
    game.current_player = 2
    game.game_over = True
    game.winner = 1
    
    # Should handle terminal state gracefully
    agent = MCTSAgent(player=2, iterations=100)
    try:
        action = agent.get_action(game)
        if action is None:
            print("Agent correctly returned None for terminal state")
        else:
            print(f"Agent returned action {action} for terminal state")
    except Exception as e:
        print(f"Error handling terminal state: {e}")
    
    print("Edge case tests completed.")


if __name__ == "__main__":
    print("Running MCTS Tests")
    print("=" * 30)
    
    # Run unit tests
    print("\n1. Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance test
    print("\n2. Running performance test...")
    run_performance_test()
    
    # Run edge case tests
    print("\n3. Running edge case tests...")
    test_edge_cases()
    
    print("\nAll tests completed!") 