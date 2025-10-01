"""
Demo script for Deep Reinforcement Learning DQN
Demonstrates the key features and capabilities of the implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from config import get_config, setup_directories, validate_config
from 0140 import (
    AdvancedDQNAgent, TrainingDatabase, TrainingVisualizer,
    train_dqn_agent, evaluate_agent
)

def demo_basic_training():
    """Demonstrate basic DQN training"""
    print("🎮 Demo 1: Basic DQN Training")
    print("-" * 40)
    
    # Get configuration
    config = get_config()
    
    # Override for quick demo
    config["training"]["episodes"] = 100
    config["agent"]["batch_size"] = 32
    
    print(f"Training for {config['training']['episodes']} episodes...")
    
    # Train the agent
    start_time = time.time()
    agent, rewards, losses = train_dqn_agent(
        episodes=config["training"]["episodes"],
        render=False,
        save_plots=True
    )
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.2f} seconds")
    print(f"📊 Final average reward: {np.mean(rewards[-10:]):.2f}")
    print(f"📈 Best reward: {np.max(rewards):.2f}")
    
    return agent, rewards, losses

def demo_agent_evaluation(agent):
    """Demonstrate agent evaluation"""
    print("\n🎯 Demo 2: Agent Evaluation")
    print("-" * 40)
    
    print("Evaluating trained agent...")
    evaluation_rewards = evaluate_agent(agent, episodes=20, render=False)
    
    avg_reward = np.mean(evaluation_rewards)
    std_reward = np.std(evaluation_rewards)
    success_rate = np.mean([r >= 475 for r in evaluation_rewards]) * 100
    
    print(f"✅ Evaluation completed!")
    print(f"📊 Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"🎯 Success rate: {success_rate:.1f}%")
    print(f"🏆 Best evaluation reward: {np.max(evaluation_rewards):.2f}")
    
    return evaluation_rewards

def demo_database_features():
    """Demonstrate database features"""
    print("\n🗄️ Demo 3: Database Features")
    print("-" * 40)
    
    # Create database
    db = TrainingDatabase("demo_results.db")
    
    # Create a demo session
    session_name = "Demo_Session"
    hyperparameters = {
        "lr": 0.001,
        "gamma": 0.99,
        "episodes": 100,
        "batch_size": 32
    }
    
    session_id = db.create_session(session_name, hyperparameters)
    print(f"✅ Created session: {session_name} (ID: {session_id})")
    
    # Log some demo episodes
    demo_episodes = [
        (1, 50.0, 0.9, 0.2, 100),
        (2, 75.0, 0.8, 0.15, 150),
        (3, 100.0, 0.7, 0.1, 200),
        (4, 125.0, 0.6, 0.08, 250),
        (5, 150.0, 0.5, 0.05, 300)
    ]
    
    for episode, reward, epsilon, loss, steps in demo_episodes:
        db.log_episode(session_id, episode, reward, epsilon, loss, steps)
    
    print(f"✅ Logged {len(demo_episodes)} demo episodes")
    
    # Retrieve and display results
    results = db.get_session_results(session_id)
    print(f"📊 Retrieved {len(results)} episode results")
    
    # Display summary statistics
    rewards = [r["reward"] for r in results]
    losses = [r["loss"] for r in results]
    steps = [r["steps"] for r in results]
    
    print(f"📈 Average reward: {np.mean(rewards):.2f}")
    print(f"📉 Average loss: {np.mean(losses):.4f}")
    print(f"👣 Average steps: {np.mean(steps):.1f}")
    
    return db, session_id

def demo_visualization():
    """Demonstrate visualization features"""
    print("\n📊 Demo 4: Visualization Features")
    print("-" * 40)
    
    # Create visualizer
    visualizer = TrainingVisualizer()
    
    # Generate demo data
    episodes = 100
    rewards = []
    losses = []
    epsilons = []
    episode_lengths = []
    
    # Simulate training progress
    for episode in range(episodes):
        # Simulate learning curve
        base_reward = 200 + episode * 2 + np.random.normal(0, 20)
        reward = max(0, min(500, base_reward))
        rewards.append(reward)
        
        # Simulate decreasing loss
        loss = max(0.01, 0.5 * np.exp(-episode / 50) + np.random.normal(0, 0.02))
        losses.append(loss)
        
        # Simulate epsilon decay
        epsilon = max(0.01, 0.99 ** episode)
        epsilons.append(epsilon)
        
        # Simulate increasing episode length
        length = min(500, 50 + episode * 3 + np.random.normal(0, 10))
        episode_lengths.append(length)
    
    print("📈 Creating training progress visualization...")
    visualizer.plot_training_progress(rewards, losses, epsilons, episode_lengths)
    
    # Save plots
    visualizer.save_plots("demo_training_progress.png")
    print("✅ Visualization saved as 'demo_training_progress.png'")
    
    return rewards, losses, epsilons, episode_lengths

def demo_performance_comparison():
    """Demonstrate performance comparison between different configurations"""
    print("\n⚡ Demo 5: Performance Comparison")
    print("-" * 40)
    
    configs = [
        {"name": "Standard DQN", "lr": 0.001, "batch_size": 32},
        {"name": "High LR", "lr": 0.01, "batch_size": 32},
        {"name": "Large Batch", "lr": 0.001, "batch_size": 128},
    ]
    
    results = {}
    
    for config in configs:
        print(f"🧪 Testing {config['name']}...")
        
        # Create agent with specific config
        agent = AdvancedDQNAgent(
            state_dim=4,
            action_dim=2,
            lr=config["lr"],
            batch_size=config["batch_size"]
        )
        
        # Quick training simulation (just a few episodes for demo)
        episode_rewards = []
        for episode in range(20):  # Short demo
            # Simulate episode
            total_reward = 50 + episode * 5 + np.random.normal(0, 10)
            episode_rewards.append(total_reward)
            
            # Simulate training
            for _ in range(10):  # Simulate steps
                agent.store_transition(
                    np.random.randn(4), 0, 1.0, np.random.randn(4), False
                )
                agent.train()
            
            agent.decay_epsilon()
        
        results[config["name"]] = {
            "final_reward": np.mean(episode_rewards[-5:]),
            "convergence": len(episode_rewards),
            "config": config
        }
        
        print(f"✅ {config['name']}: Final reward = {results[config['name']]['final_reward']:.2f}")
    
    # Display comparison
    print("\n📊 Performance Comparison Results:")
    for name, result in results.items():
        print(f"  {name}: {result['final_reward']:.2f}")
    
    return results

def demo_model_save_load():
    """Demonstrate model saving and loading"""
    print("\n💾 Demo 6: Model Save/Load")
    print("-" * 40)
    
    # Create and train a simple agent
    agent = AdvancedDQNAgent(state_dim=4, action_dim=2)
    
    # Simulate some training
    for episode in range(10):
        for step in range(20):
            agent.store_transition(
                np.random.randn(4), 0, 1.0, np.random.randn(4), False
            )
            agent.train()
        agent.decay_epsilon()
    
    print("✅ Agent trained (simulated)")
    
    # Save model
    model_path = "demo_model.pth"
    agent.save_model(model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Create new agent and load model
    new_agent = AdvancedDQNAgent(state_dim=4, action_dim=2)
    new_agent.load_model(model_path)
    print("✅ Model loaded successfully")
    
    # Verify parameters match
    original_params = list(agent.q_network.parameters())
    loaded_params = list(new_agent.q_network.parameters())
    
    params_match = all(
        torch.allclose(orig.data, loaded.data) 
        for orig, loaded in zip(original_params, loaded_params)
    )
    
    if params_match:
        print("✅ Model parameters verified - save/load successful!")
    else:
        print("❌ Model parameters don't match!")
    
    # Clean up
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
        print("🧹 Demo model file cleaned up")

def run_complete_demo():
    """Run the complete demonstration"""
    print("🚀 Deep Reinforcement Learning DQN - Complete Demo")
    print("=" * 60)
    
    # Setup
    setup_directories()
    config = get_config()
    
    if not validate_config(config):
        print("❌ Configuration validation failed!")
        return
    
    print("✅ Configuration validated successfully!")
    
    try:
        # Run all demos
        agent, rewards, losses = demo_basic_training()
        evaluation_rewards = demo_agent_evaluation(agent)
        db, session_id = demo_database_features()
        demo_rewards, demo_losses, demo_epsilons, demo_lengths = demo_visualization()
        performance_results = demo_performance_comparison()
        demo_model_save_load()
        
        print("\n🎉 Demo completed successfully!")
        print("=" * 60)
        print("📋 Summary:")
        print(f"  • Training episodes: {len(rewards)}")
        print(f"  • Final training reward: {np.mean(rewards[-10:]):.2f}")
        print(f"  • Evaluation reward: {np.mean(evaluation_rewards):.2f}")
        print(f"  • Database sessions: 1")
        print(f"  • Performance comparisons: {len(performance_results)}")
        print(f"  • Visualizations created: 1")
        
        print("\n🎯 Next Steps:")
        print("  • Run 'python dashboard.py' for interactive web UI")
        print("  • Run 'python test_dqn.py' for comprehensive testing")
        print("  • Modify config.py for custom experiments")
        print("  • Check the README.md for detailed documentation")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_complete_demo()
