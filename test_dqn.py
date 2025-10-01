"""
Unit Tests for Deep Reinforcement Learning DQN Implementation
Tests cover all major components including networks, replay buffer, agent, and training.
"""

import unittest
import numpy as np
import torch
import sqlite3
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our DQN components
from 0140 import (
    DuelingDQN, PrioritizedReplayBuffer, AdvancedDQNAgent, 
    TrainingDatabase, TrainingVisualizer
)

class TestDuelingDQN(unittest.TestCase):
    """Test cases for Dueling DQN network"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state_dim = 4
        self.action_dim = 2
        self.hidden_dim = 128
        self.network = DuelingDQN(self.state_dim, self.action_dim, self.hidden_dim)
    
    def test_network_initialization(self):
        """Test network initialization"""
        self.assertIsInstance(self.network, DuelingDQN)
        self.assertEqual(self.network.feature_layer[0].in_features, self.state_dim)
        self.assertEqual(self.network.value_stream[-1].out_features, 1)
        self.assertEqual(self.network.advantage_stream[-1].out_features, self.action_dim)
    
    def test_forward_pass(self):
        """Test forward pass through the network"""
        batch_size = 32
        state = torch.randn(batch_size, self.state_dim)
        
        q_values = self.network(state)
        
        self.assertEqual(q_values.shape, (batch_size, self.action_dim))
        self.assertTrue(torch.isfinite(q_values).all())
    
    def test_single_state_forward(self):
        """Test forward pass with single state"""
        state = torch.randn(1, self.state_dim)
        q_values = self.network(state)
        
        self.assertEqual(q_values.shape, (1, self.action_dim))
    
    def test_gradient_flow(self):
        """Test that gradients flow properly"""
        state = torch.randn(10, self.state_dim, requires_grad=True)
        q_values = self.network(state)
        loss = q_values.sum()
        loss.backward()
        
        # Check that gradients exist for all parameters
        for param in self.network.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())

class TestPrioritizedReplayBuffer(unittest.TestCase):
    """Test cases for Prioritized Experience Replay Buffer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.capacity = 1000
        self.buffer = PrioritizedReplayBuffer(capacity=self.capacity)
    
    def test_buffer_initialization(self):
        """Test buffer initialization"""
        self.assertEqual(self.buffer.capacity, self.capacity)
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.position, 0)
    
    def test_push_transition(self):
        """Test pushing transitions to buffer"""
        state = np.array([1, 2, 3, 4])
        action = 1
        reward = 1.0
        next_state = np.array([2, 3, 4, 5])
        done = False
        
        self.buffer.push(state, action, reward, next_state, done)
        
        self.assertEqual(len(self.buffer), 1)
        self.assertEqual(self.buffer.position, 1)
    
    def test_buffer_capacity_limit(self):
        """Test that buffer respects capacity limit"""
        # Fill buffer beyond capacity
        for i in range(self.capacity + 100):
            state = np.random.randn(4)
            action = i % 2
            reward = float(i)
            next_state = np.random.randn(4)
            done = i % 10 == 0
            
            self.buffer.push(state, action, reward, next_state, done)
        
        self.assertEqual(len(self.buffer), self.capacity)
    
    def test_sample_batch(self):
        """Test sampling batch from buffer"""
        # Add some transitions
        for i in range(50):
            state = np.random.randn(4)
            action = i % 2
            reward = float(i)
            next_state = np.random.randn(4)
            done = i % 10 == 0
            
            self.buffer.push(state, action, reward, next_state, done)
        
        batch_size = 16
        batch = self.buffer.sample(batch_size)
        
        self.assertIsNotNone(batch)
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        self.assertEqual(len(states), batch_size)
        self.assertEqual(len(actions), batch_size)
        self.assertEqual(len(rewards), batch_size)
        self.assertEqual(len(next_states), batch_size)
        self.assertEqual(len(dones), batch_size)
        self.assertEqual(len(indices), batch_size)
        self.assertEqual(len(weights), batch_size)
    
    def test_empty_buffer_sample(self):
        """Test sampling from empty buffer"""
        batch = self.buffer.sample(16)
        self.assertIsNone(batch)
    
    def test_priority_update(self):
        """Test priority updates"""
        # Add some transitions
        for i in range(10):
            state = np.random.randn(4)
            action = i % 2
            reward = float(i)
            next_state = np.random.randn(4)
            done = False
            
            self.buffer.push(state, action, reward, next_state, done)
        
        # Sample and update priorities
        batch = self.buffer.sample(5)
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        new_priorities = np.random.rand(5)
        self.buffer.update_priorities(indices, new_priorities)
        
        # Check that priorities were updated
        for idx, priority in zip(indices, new_priorities):
            self.assertEqual(self.buffer.priorities[idx], priority)

class TestAdvancedDQNAgent(unittest.TestCase):
    """Test cases for Advanced DQN Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state_dim = 4
        self.action_dim = 2
        self.agent = AdvancedDQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lr=0.001,
            gamma=0.99,
            epsilon=0.1,
            batch_size=32
        )
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertEqual(self.agent.epsilon, 0.1)
        self.assertEqual(self.agent.gamma, 0.99)
    
    def test_action_selection_training(self):
        """Test action selection during training"""
        state = np.random.randn(self.state_dim)
        
        # Test multiple times to check both random and greedy actions
        actions = []
        for _ in range(100):
            action = self.agent.select_action(state, training=True)
            actions.append(action)
            self.assertIn(action, range(self.action_dim))
        
        # Should have some variety due to epsilon-greedy
        unique_actions = set(actions)
        self.assertGreater(len(unique_actions), 1)
    
    def test_action_selection_evaluation(self):
        """Test action selection during evaluation (no exploration)"""
        state = np.random.randn(self.state_dim)
        
        # During evaluation, should always select greedy action
        actions = []
        for _ in range(10):
            action = self.agent.select_action(state, training=False)
            actions.append(action)
            self.assertIn(action, range(self.action_dim))
        
        # All actions should be the same (greedy)
        self.assertEqual(len(set(actions)), 1)
    
    def test_store_transition(self):
        """Test storing transitions"""
        state = np.random.randn(self.state_dim)
        action = 1
        reward = 1.0
        next_state = np.random.randn(self.state_dim)
        done = False
        
        initial_length = len(self.agent.replay_buffer)
        self.agent.store_transition(state, action, reward, next_state, done)
        
        self.assertEqual(len(self.agent.replay_buffer), initial_length + 1)
    
    def test_train_with_insufficient_data(self):
        """Test training with insufficient replay buffer data"""
        # Try to train with empty buffer
        loss = self.agent.train()
        self.assertEqual(loss, 0.0)
        
        # Add some data but not enough for batch
        for i in range(10):  # Less than batch_size (32)
            state = np.random.randn(self.state_dim)
            action = i % self.action_dim
            reward = float(i)
            next_state = np.random.randn(self.state_dim)
            done = i % 5 == 0
            
            self.agent.store_transition(state, action, reward, next_state, done)
        
        loss = self.agent.train()
        self.assertEqual(loss, 0.0)
    
    def test_train_with_sufficient_data(self):
        """Test training with sufficient replay buffer data"""
        # Add enough data for training
        for i in range(50):  # More than batch_size (32)
            state = np.random.randn(self.state_dim)
            action = i % self.action_dim
            reward = float(i)
            next_state = np.random.randn(self.state_dim)
            done = i % 10 == 0
            
            self.agent.store_transition(state, action, reward, next_state, done)
        
        loss = self.agent.train()
        self.assertGreater(loss, 0.0)
        self.assertTrue(np.isfinite(loss))
    
    def test_epsilon_decay(self):
        """Test epsilon decay"""
        initial_epsilon = self.agent.epsilon
        self.agent.decay_epsilon()
        
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.min_epsilon)
    
    def test_target_network_update(self):
        """Test target network update"""
        # Get initial parameters
        initial_params = list(self.agent.target_network.parameters())
        
        # Modify main network parameters
        for param in self.agent.q_network.parameters():
            param.data += torch.randn_like(param.data) * 0.1
        
        # Update target network
        self.agent.update_target_network()
        
        # Check that target network parameters match main network
        main_params = list(self.agent.q_network.parameters())
        target_params = list(self.agent.target_network.parameters())
        
        for main_param, target_param in zip(main_params, target_params):
            self.assertTrue(torch.allclose(main_param.data, target_param.data))
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            # Save model
            self.agent.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Create new agent and load model
            new_agent = AdvancedDQNAgent(self.state_dim, self.action_dim)
            new_agent.load_model(model_path)
            
            # Check that parameters match
            main_params = list(self.agent.q_network.parameters())
            loaded_params = list(new_agent.q_network.parameters())
            
            for main_param, loaded_param in zip(main_params, loaded_params):
                self.assertTrue(torch.allclose(main_param.data, loaded_param.data))
            
            # Check other attributes
            self.assertEqual(self.agent.epsilon, new_agent.epsilon)
            self.assertEqual(self.agent.episode_count, new_agent.episode_count)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)

class TestTrainingDatabase(unittest.TestCase):
    """Test cases for Training Database"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db = TrainingDatabase(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test database initialization"""
        self.assertTrue(os.path.exists(self.temp_db.name))
        
        # Check that tables exist
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        self.assertIn('training_sessions', tables)
        self.assertIn('episode_results', tables)
        
        conn.close()
    
    def test_create_session(self):
        """Test creating a training session"""
        session_name = "test_session"
        hyperparameters = {"lr": 0.001, "gamma": 0.99}
        
        session_id = self.db.create_session(session_name, hyperparameters)
        
        self.assertIsInstance(session_id, int)
        self.assertGreater(session_id, 0)
        
        # Verify session was created
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_name, hyperparameters FROM training_sessions 
            WHERE id = ?
        ''', (session_id,))
        
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], session_name)
        
        loaded_hyperparams = json.loads(result[1])
        self.assertEqual(loaded_hyperparams, hyperparameters)
        
        conn.close()
    
    def test_log_episode(self):
        """Test logging episode results"""
        # Create a session first
        session_id = self.db.create_session("test_session", {"lr": 0.001})
        
        # Log an episode
        episode = 1
        reward = 100.0
        epsilon = 0.5
        loss = 0.1
        steps = 200
        
        self.db.log_episode(session_id, episode, reward, epsilon, loss, steps)
        
        # Verify episode was logged
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT episode, reward, epsilon, loss, steps 
            FROM episode_results WHERE session_id = ?
        ''', (session_id,))
        
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result, (episode, reward, epsilon, loss, steps))
        
        conn.close()
    
    def test_get_session_results(self):
        """Test getting session results"""
        # Create a session and log multiple episodes
        session_id = self.db.create_session("test_session", {"lr": 0.001})
        
        episodes_data = [
            (1, 50.0, 0.9, 0.2, 100),
            (2, 75.0, 0.8, 0.15, 150),
            (3, 100.0, 0.7, 0.1, 200)
        ]
        
        for episode, reward, epsilon, loss, steps in episodes_data:
            self.db.log_episode(session_id, episode, reward, epsilon, loss, steps)
        
        # Get results
        results = self.db.get_session_results(session_id)
        
        self.assertEqual(len(results), 3)
        
        for i, (episode, reward, epsilon, loss, steps) in enumerate(episodes_data):
            result = results[i]
            self.assertEqual(result['episode'], episode)
            self.assertEqual(result['reward'], reward)
            self.assertEqual(result['epsilon'], epsilon)
            self.assertEqual(result['loss'], loss)
            self.assertEqual(result['steps'], steps)

class TestTrainingVisualizer(unittest.TestCase):
    """Test cases for Training Visualizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.visualizer = TrainingVisualizer()
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization"""
        self.assertIsNotNone(self.visualizer.fig)
        self.assertEqual(len(self.visualizer.axes), 2)
        self.assertEqual(len(self.visualizer.axes[0]), 2)
    
    def test_moving_average_calculation(self):
        """Test moving average calculation"""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        window = 3
        
        moving_avg = self.visualizer._moving_average(data, window)
        
        self.assertEqual(len(moving_avg), len(data))
        
        # Check first few values
        self.assertEqual(moving_avg[0], 1.0)  # Only one value
        self.assertEqual(moving_avg[1], 1.5)  # Average of [1, 2]
        self.assertEqual(moving_avg[2], 2.0)   # Average of [1, 2, 3]
        self.assertEqual(moving_avg[3], 3.0)   # Average of [2, 3, 4]
    
    def test_moving_average_short_data(self):
        """Test moving average with data shorter than window"""
        data = [1, 2]
        window = 5
        
        moving_avg = self.visualizer._moving_average(data, window)
        
        self.assertEqual(moving_avg, data)  # Should return original data
    
    @patch('matplotlib.pyplot.savefig')
    def test_save_plots(self, mock_savefig):
        """Test saving plots"""
        filepath = "test_plot.png"
        self.visualizer.save_plots(filepath)
        
        mock_savefig.assert_called_once_with(filepath, dpi=300, bbox_inches='tight')

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    @patch('gymnasium.make')
    def test_full_training_cycle(self, mock_gym_make):
        """Test a complete training cycle"""
        # Mock environment
        mock_env = MagicMock()
        mock_env.observation_space.shape = [4]
        mock_env.action_space.n = 2
        mock_env.reset.return_value = (np.random.randn(4), {})
        mock_env.step.return_value = (np.random.randn(4), 1.0, False, False, {})
        mock_gym_make.return_value = mock_env
        
        # Create agent
        agent = AdvancedDQNAgent(state_dim=4, action_dim=2, batch_size=16)
        
        # Create database
        db = TrainingDatabase(self.temp_db.name)
        session_id = db.create_session("integration_test", {"lr": 0.001})
        
        # Run a few training steps
        for episode in range(5):
            state, _ = mock_env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(10):  # Short episodes for testing
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = mock_env.step(action)
                done = terminated or truncated
                
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Log episode
            db.log_episode(session_id, episode, total_reward, agent.epsilon, 0.1, steps)
            agent.decay_epsilon()
        
        # Verify training data
        results = db.get_session_results(session_id)
        self.assertEqual(len(results), 5)
        
        # Verify agent learned something (epsilon decreased)
        self.assertLess(agent.epsilon, 1.0)

def run_performance_tests():
    """Run performance benchmarks"""
    import time
    
    print("Running performance tests...")
    
    # Test network forward pass speed
    network = DuelingDQN(4, 2)
    state = torch.randn(1000, 4)
    
    start_time = time.time()
    for _ in range(100):
        _ = network(state)
    forward_time = time.time() - start_time
    
    print(f"Network forward pass (1000 samples x 100): {forward_time:.4f}s")
    
    # Test replay buffer sampling speed
    buffer = PrioritizedReplayBuffer(10000)
    
    # Fill buffer
    for i in range(10000):
        buffer.push(np.random.randn(4), i % 2, float(i), np.random.randn(4), i % 10 == 0)
    
    start_time = time.time()
    for _ in range(1000):
        _ = buffer.sample(64)
    sampling_time = time.time() - start_time
    
    print(f"Replay buffer sampling (64 samples x 1000): {sampling_time:.4f}s")

if __name__ == "__main__":
    # Run unit tests
    unittest.main(verbosity=2)
    
    # Run performance tests
    run_performance_tests()
