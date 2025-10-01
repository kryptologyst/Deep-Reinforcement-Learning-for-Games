# Project 140. Deep reinforcement learning for games
# Description:
# Deep Reinforcement Learning (DRL) combines reinforcement learning with deep neural networks to learn optimal actions from raw states like pixels or numerical observations. In this project, we implement advanced Deep Q-Network (DQN) variants to play simple games (e.g., CartPole) using the modern Gymnasium environment.

# Python Implementation: Advanced DQN Agent for CartPole-v1
# Install dependencies: pip install gymnasium torch numpy matplotlib seaborn plotly dash sqlite3
 
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sqlite3
import os
from collections import deque
from datetime import datetime
import logging
from typing import Tuple, List, Dict, Any
import pickle
 
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database setup for storing training results
class TrainingDatabase:
    def __init__(self, db_path: str = "training_results.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Training sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                total_episodes INTEGER,
                final_score REAL,
                model_path TEXT,
                hyperparameters TEXT
            )
        ''')
        
        # Episode results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episode_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                episode INTEGER,
                reward REAL,
                epsilon REAL,
                loss REAL,
                steps INTEGER,
                FOREIGN KEY (session_id) REFERENCES training_sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_session(self, session_name: str, hyperparameters: Dict[str, Any]) -> int:
        """Create a new training session and return session ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_sessions (session_name, hyperparameters)
            VALUES (?, ?)
        ''', (session_name, json.dumps(hyperparameters)))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return session_id
    
    def log_episode(self, session_id: int, episode: int, reward: float, 
                   epsilon: float, loss: float, steps: int):
        """Log episode results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO episode_results (session_id, episode, reward, epsilon, loss, steps)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, episode, reward, epsilon, loss, steps))
        
        conn.commit()
        conn.close()
    
    def get_session_results(self, session_id: int) -> List[Dict]:
        """Get all episode results for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT episode, reward, epsilon, loss, steps
            FROM episode_results
            WHERE session_id = ?
            ORDER BY episode
        ''', (session_id,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'episode': row[0],
                'reward': row[1],
                'epsilon': row[2],
                'loss': row[3],
                'steps': row[4]
            })
        
        conn.close()
        return results

# Advanced Dueling DQN Network
class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DuelingDQN, self).__init__()
        
        # Shared feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage streams
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

# Prioritized Experience Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.beta_increment = 0.001
    
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            return None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Extract batch
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), indices, weights)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

# Advanced DQN Agent with Double DQN and Dueling Architecture
class AdvancedDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.01, target_update: int = 10,
                 buffer_size: int = 10000, batch_size: int = 64,
                 device: str = None):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.batch_size = batch_size
        
        # Device setup
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Networks
        self.q_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        
        # Training metrics
        self.training_losses = []
        self.training_rewards = []
        self.episode_count = 0
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self) -> float:
        """Train the agent using Double DQN with prioritized experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return 0.0
        
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (~dones)
        
        # Calculate TD errors for priority updates
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors + 1e-6)
        
        # Calculate loss with importance sampling weights
        loss = F.mse_loss(current_q_values, target_q_values, reduction='none')
        weighted_loss = (loss * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        return weighted_loss.item()
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']

# Training and Visualization Functions
class TrainingVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('DQN Training Progress', fontsize=16)
    
    def plot_training_progress(self, rewards: List[float], losses: List[float], 
                             epsilons: List[float], episode_lengths: List[int]):
        """Plot training progress"""
        episodes = range(len(rewards))
        
        # Plot rewards
        self.axes[0, 0].clear()
        self.axes[0, 0].plot(episodes, rewards, alpha=0.6, color='blue')
        self.axes[0, 0].plot(episodes, self._moving_average(rewards, 10), color='red', linewidth=2)
        self.axes[0, 0].set_title('Episode Rewards')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].set_ylabel('Reward')
        self.axes[0, 0].grid(True)
        
        # Plot losses
        self.axes[0, 1].clear()
        self.axes[0, 1].plot(episodes, losses, alpha=0.6, color='green')
        self.axes[0, 1].plot(episodes, self._moving_average(losses, 10), color='red', linewidth=2)
        self.axes[0, 1].set_title('Training Loss')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Loss')
        self.axes[0, 1].grid(True)
        
        # Plot epsilon
        self.axes[1, 0].clear()
        self.axes[1, 0].plot(episodes, epsilons, color='orange')
        self.axes[1, 0].set_title('Exploration Rate (Epsilon)')
        self.axes[1, 0].set_xlabel('Episode')
        self.axes[1, 0].set_ylabel('Epsilon')
        self.axes[1, 0].grid(True)
        
        # Plot episode lengths
        self.axes[1, 1].clear()
        self.axes[1, 1].plot(episodes, episode_lengths, alpha=0.6, color='purple')
        self.axes[1, 1].plot(episodes, self._moving_average(episode_lengths, 10), color='red', linewidth=2)
        self.axes[1, 1].set_title('Episode Length')
        self.axes[1, 1].set_xlabel('Episode')
        self.axes[1, 1].set_ylabel('Steps')
        self.axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """Calculate moving average"""
        if len(data) < window:
            return data
        return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]
    
    def save_plots(self, filepath: str):
        """Save plots to file"""
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

# Main training function
def train_dqn_agent(episodes: int = 1000, render: bool = False, save_plots: bool = True):
    """Train the DQN agent"""
    logger.info("Starting DQN training...")
    
    # Environment setup
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = AdvancedDQNAgent(state_dim, action_dim)
    
    # Initialize database
    db = TrainingDatabase()
    session_name = f"DQN_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    hyperparameters = {
        'episodes': episodes,
        'lr': agent.lr,
        'gamma': agent.gamma,
        'epsilon_decay': agent.epsilon_decay,
        'min_epsilon': agent.min_epsilon,
        'batch_size': agent.batch_size,
        'target_update': agent.target_update
    }
    session_id = db.create_session(session_name, hyperparameters)
    
    # Initialize visualizer
    visualizer = TrainingVisualizer()
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    episode_epsilons = []
    episode_lengths = []
    
    # Training loop
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        episode_loss = 0
        loss_count = 0
        
        done = False
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train()
            if loss > 0:
                episode_loss += loss
                loss_count += 1
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Update metrics
        avg_loss = episode_loss / max(loss_count, 1)
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)
        episode_epsilons.append(agent.epsilon)
        episode_lengths.append(steps)
        
        # Log to database
        db.log_episode(session_id, episode, total_reward, agent.epsilon, avg_loss, steps)
        
        # Update agent
        agent.decay_epsilon()
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        # Update visualization
        if episode % 10 == 0:
            visualizer.plot_training_progress(episode_rewards, episode_losses, 
                                            episode_epsilons, episode_lengths)
        
        # Log progress
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            logger.info(f"Episode {episode+1}/{episodes} - "
                       f"Avg Reward: {avg_reward:.2f}, "
                       f"Epsilon: {agent.epsilon:.3f}, "
                       f"Loss: {avg_loss:.4f}")
    
    # Save final model
    model_path = f"models/dqn_model_{session_name}.pth"
    os.makedirs("models", exist_ok=True)
    agent.save_model(model_path)
    
    # Update database with final results
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE training_sessions 
        SET end_time = CURRENT_TIMESTAMP, 
            total_episodes = ?, 
            final_score = ?, 
            model_path = ?
        WHERE id = ?
    ''', (episodes, np.mean(episode_rewards[-100:]), model_path, session_id))
    conn.commit()
    conn.close()
    
    # Save plots
    if save_plots:
        os.makedirs("plots", exist_ok=True)
        visualizer.save_plots(f"plots/training_progress_{session_name}.png")
    
    env.close()
    logger.info(f"Training completed! Final average reward: {np.mean(episode_rewards[-100:]):.2f}")
    
    return agent, episode_rewards, episode_losses

# Evaluation function
def evaluate_agent(agent: AdvancedDQNAgent, episodes: int = 100, render: bool = False):
    """Evaluate the trained agent"""
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    
    total_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        total_rewards.append(total_reward)
        logger.info(f"Evaluation Episode {episode+1}: Reward = {total_reward}")
    
    env.close()
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    logger.info(f"Evaluation Results - Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    
    return total_rewards

if __name__ == "__main__":
    # Train the agent
    agent, rewards, losses = train_dqn_agent(episodes=500, render=False, save_plots=True)
    
    # Evaluate the trained agent
    evaluation_rewards = evaluate_agent(agent, episodes=50, render=False)
    
    print(f"\n🎮 Training Complete!")
    print(f"📊 Final Training Performance: {np.mean(rewards[-100:]):.2f}")
    print(f"🎯 Evaluation Performance: {np.mean(evaluation_rewards):.2f}")
    print(f"💾 Model saved to models/ directory")
    print(f"📈 Plots saved to plots/ directory")
    print(f"🗄️ Training data saved to training_results.db")