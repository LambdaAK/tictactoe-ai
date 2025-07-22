import numpy as np
import matplotlib.pyplot as plt
from tictactoe_env import TicTacToeEnv, Player, GameStatus
from q_learning_agent import QLearningAgent
from random_agent import RandomAgent
import time
from typing import List, Tuple, Dict
import datetime

class Trainer:
    """
    Trainer class for training Q-learning agents.
    """
    
    def __init__(self, env: TicTacToeEnv, agent: QLearningAgent, 
                 opponent_type: str = "random", opponent_agent: QLearningAgent = None):
        """
        Initialize trainer.
        
        Args:
            env: The environment
            agent: The agent to train
            opponent_type: Type of opponent ("random", "self", or "agent")
            opponent_agent: Opponent agent (if opponent_type is "agent")
        """
        self.env = env
        self.agent = agent
        self.opponent_type = opponent_type
        self.opponent_agent = opponent_agent
        
        # Training statistics
        self.episode_rewards = []
        self.win_rates = []
        self.episode_lengths = []
        self.epsilon_history = []
        
    def create_opponent(self, player: Player) -> QLearningAgent:
        """Create opponent agent."""
        if self.opponent_type == "random":
            return RandomAgent(player, self.env)
        elif self.opponent_type == "self":
            # Create a copy of the training agent
            opponent = QLearningAgent(player, self.env)
            opponent.q_table = self.agent.q_table.copy()
            opponent.set_training_mode(False)  # Opponent doesn't learn during training
            return opponent
        elif self.opponent_type == "agent" and self.opponent_agent:
            return self.opponent_agent
        else:
            return RandomAgent(player, self.env)
    
    def play_episode(self, agent_first: bool = True) -> Tuple[float, int, bool]:
        """
        Play a single training episode.
        
        Args:
            agent_first: Whether the training agent goes first
            
        Returns:
            (total_reward, episode_length, agent_won)
        """
        state = self.env.reset()
        total_reward = 0
        episode_length = 0
        
        # Create opponent
        opponent = self.create_opponent(Player.O if agent_first else Player.X)
        
        # Determine agent's player
        agent_player = Player.X if agent_first else Player.O
        
        while not self.env.is_done():
            current_player = self.env.current_player
            valid_actions = self.env.get_valid_actions()
            
            if current_player == agent_player:
                # Training agent's turn
                action = self.agent.select_action(state, valid_actions)
            else:
                # Opponent's turn
                action = opponent.select_action(state, valid_actions)
            
            # Make the move
            next_state, reward, done, info = self.env.step(action)
            episode_length += 1
            
            # Update agent if it was their turn
            if current_player == agent_player:
                self.agent.update(state, action, reward, next_state, done)
                total_reward += reward
            
            state = next_state
        
        # Determine if agent won
        winner = self.env.get_winner()
        agent_won = (winner == agent_player)
        
        return total_reward, episode_length, agent_won
    
    def train(self, num_episodes: int, eval_interval: int = 1000, 
              save_interval: int = 10000, save_path: str = "trained_agent.json") -> Dict:
        """
        Train the agent for specified number of episodes.
        
        Args:
            num_episodes: Number of training episodes
            eval_interval: How often to evaluate performance
            save_interval: How often to save the agent
            save_path: Path to save the trained agent
            
        Returns:
            Training statistics
        """
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Agent: {self.agent.get_info()}")
        print(f"Opponent: {self.opponent_type}")
        
        start_time = time.time()
        wins = 0
        draws = 0
        
        for episode in range(num_episodes):
            # Alternate who goes first
            agent_first = (episode % 2 == 0)
            
            # Play episode
            reward, length, won = self.play_episode(agent_first)
            
            # Track statistics
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            self.epsilon_history.append(self.agent.epsilon)
            
            if won:
                wins += 1
            elif self.env.get_winner() is None:
                draws += 1
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            # Evaluation and logging
            if (episode + 1) % eval_interval == 0:
                win_rate = wins / (episode + 1)
                draw_rate = draws / (episode + 1)
                avg_reward = np.mean(self.episode_rewards[-eval_interval:])
                avg_length = np.mean(self.episode_lengths[-eval_interval:])
                
                self.win_rates.append(win_rate)
                
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Win Rate: {win_rate:.3f}, Draw Rate: {draw_rate:.3f}")
                print(f"  Avg Reward: {avg_reward:.3f}, Avg Length: {avg_length:.1f}")
                print(f"  Epsilon: {self.agent.epsilon:.3f}")
                print(f"  Q-table size: {len(self.agent.q_table)}")
                print()

            # Save agent periodically
            if (episode + 1) % save_interval == 0:
                self.agent.save(save_path)
                print(f"Agent saved to {save_path}")
        
        # Final save
        self.agent.save(save_path)
        
        training_time = time.time() - start_time
        final_win_rate = wins / num_episodes
        
        print(f"\nTraining completed!")
        print(f"Total time: {training_time:.1f} seconds")
        print(f"Final win rate: {final_win_rate:.3f}")
        print(f"Final Q-table size: {len(self.agent.q_table)}")
        print(f"Agent saved to: {save_path}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'win_rates': self.win_rates,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'final_win_rate': final_win_rate,
            'training_time': training_time
        }
    
    def plot_training_progress(self, stats: Dict, plot_filename: str):
        """Plot training progress."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Win rate over time
        ax1.plot(range(0, len(stats['win_rates']) * 1000, 1000), stats['win_rates'])
        ax1.set_title('Win Rate Over Time')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Win Rate')
        ax1.grid(True)
        
        # Average reward over time
        rewards = stats['episode_rewards']
        window = 1000
        avg_rewards = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
        ax2.plot(avg_rewards)
        ax2.set_title('Average Reward Over Time')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.grid(True)
        
        # Episode lengths
        lengths = stats['episode_lengths']
        avg_lengths = [np.mean(lengths[max(0, i-window):i+1]) for i in range(len(lengths))]
        ax3.plot(avg_lengths)
        ax3.set_title('Average Episode Length')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Length')
        ax3.grid(True)
        
        # Epsilon decay
        ax4.plot(stats['epsilon_history'])
        ax4.set_title('Epsilon Decay')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.show()

def main():
    """Main training function."""
    print("Q-Learning Agent Trainer")
    print("=" * 40)
    
    # Create environment and agent
    env = TicTacToeEnv()
    agent = QLearningAgent(
        player=Player.X,
        env=env,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.3,  # Start with higher exploration
        epsilon_decay=0.9995,
        epsilon_min=0.01
    )
    
    # Training parameters
    num_episodes = int(input("Enter number of training episodes (default 50000): ") or "50000")
    opponent_type = input("Choose opponent type (random/self, default random): ").strip() or "random"
    
    # Generate unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"q_agent_{opponent_type}_{num_episodes}ep_{timestamp}.json"
    
    print(f"\nAgent will be saved as: {filename}")
    
    # Create trainer
    trainer = Trainer(env, agent, opponent_type=opponent_type)
    
    # Train the agent
    stats = trainer.train(
        num_episodes=num_episodes,
        eval_interval=1000,
        save_interval=10000,
        save_path=filename
    )
    
    # Plot results
    try:
        plot_filename = f"training_progress_{timestamp}.png"
        trainer.plot_training_progress(stats, plot_filename)
    except ImportError:
        print("Matplotlib not available, skipping plots.")
    
    print(f"\nTraining complete! Agent saved as: {filename}")
    print(f"You can now play against the trained agent using play_game.py")
    print(f"Enter '{filename}' when prompted for the agent file path.")

if __name__ == "__main__":
    main() 