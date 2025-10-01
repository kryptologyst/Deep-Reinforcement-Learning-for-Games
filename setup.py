#!/usr/bin/env python3
"""
Setup script for Deep Reinforcement Learning DQN project
Handles installation, configuration, and initial setup.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        "models",
        "plots", 
        "logs",
        "data",
        "checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ Created directory: {directory}")

def check_gpu_support():
    """Check for GPU/CUDA support"""
    print("🔍 Checking GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {gpu_count} GPU(s) - {gpu_name}")
            return True
        else:
            print("ℹ️  CUDA not available, will use CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False

def run_tests():
    """Run basic tests to verify installation"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        import gymnasium
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import sqlite3
        print("✅ All core imports successful")
        
        # Test environment creation
        env = gymnasium.make("CartPole-v1")
        state, _ = env.reset()
        print(f"✅ Environment test successful - State shape: {state.shape}")
        env.close()
        
        # Test database creation
        from 0140 import TrainingDatabase
        db = TrainingDatabase("test_setup.db")
        print("✅ Database test successful")
        
        # Clean up test database
        if os.path.exists("test_setup.db"):
            os.remove("test_setup.db")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def create_sample_config():
    """Create a sample configuration file"""
    print("⚙️  Creating sample configuration...")
    
    config_content = '''# Sample configuration for Deep Reinforcement Learning DQN
# Copy this file to config.py and modify as needed

# Environment Configuration
ENVIRONMENT_CONFIG = {
    "name": "CartPole-v1",
    "render_mode": None,
    "max_episode_steps": 500,
    "reward_threshold": 475.0
}

# Agent Configuration  
AGENT_CONFIG = {
    "lr": 0.001,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "min_epsilon": 0.01,
    "target_update": 10,
    "buffer_size": 10000,
    "batch_size": 64
}

# Training Configuration
TRAINING_CONFIG = {
    "episodes": 1000,
    "eval_episodes": 100,
    "save_frequency": 100,
    "log_frequency": 50
}
'''
    
    with open("config_sample.py", "w") as f:
        f.write(config_content)
    
    print("✅ Sample configuration created: config_sample.py")

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎉 Setup completed successfully!")
    print("=" * 50)
    print("📋 Next Steps:")
    print("  1. Run demo: python demo.py")
    print("  2. Start training: python 0140.py")
    print("  3. Launch dashboard: python dashboard.py")
    print("  4. Run tests: python test_dqn.py")
    print("  5. Read documentation: README.md")
    print("\n🔧 Configuration:")
    print("  • Edit config.py for custom settings")
    print("  • Modify hyperparameters as needed")
    print("  • Check GPU support in the logs above")
    print("\n📚 Documentation:")
    print("  • README.md - Complete documentation")
    print("  • config.py - Configuration options")
    print("  • demo.py - Usage examples")

def main():
    """Main setup function"""
    print("🚀 Deep Reinforcement Learning DQN - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Setup directories
    setup_directories()
    
    # Check GPU support
    check_gpu_support()
    
    # Run tests
    if not run_tests():
        print("⚠️  Some tests failed, but setup may still work")
    
    # Create sample config
    create_sample_config()
    
    # Print next steps
    print_next_steps()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
