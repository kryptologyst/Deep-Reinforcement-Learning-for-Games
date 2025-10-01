"""
Configuration file for Deep Reinforcement Learning DQN
Contains all hyperparameters and settings for easy modification.
"""

import os
from typing import Dict, Any

# Environment Configuration
ENVIRONMENT_CONFIG = {
    "name": "CartPole-v1",
    "render_mode": None,  # "human" for visualization
    "max_episode_steps": 500,
    "reward_threshold": 475.0
}

# Agent Configuration
AGENT_CONFIG = {
    "state_dim": 4,  # Will be set automatically from environment
    "action_dim": 2,  # Will be set automatically from environment
    "lr": 0.001,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "min_epsilon": 0.01,
    "target_update": 10,
    "buffer_size": 10000,
    "batch_size": 64,
    "hidden_dim": 128
}

# Training Configuration
TRAINING_CONFIG = {
    "episodes": 1000,
    "eval_episodes": 100,
    "save_frequency": 100,  # Save model every N episodes
    "log_frequency": 50,    # Log progress every N episodes
    "plot_frequency": 10,   # Update plots every N episodes
    "early_stopping": {
        "enabled": True,
        "patience": 200,     # Stop if no improvement for N episodes
        "min_delta": 10.0    # Minimum improvement threshold
    }
}

# Network Architecture Configuration
NETWORK_CONFIG = {
    "architecture": "dueling",  # "dueling" or "standard"
    "hidden_layers": [128, 128],
    "activation": "relu",
    "dropout": 0.0,
    "batch_norm": False
}

# Replay Buffer Configuration
REPLAY_BUFFER_CONFIG = {
    "type": "prioritized",  # "prioritized" or "standard"
    "capacity": 10000,
    "alpha": 0.6,  # Prioritization exponent
    "beta": 0.4,   # Importance sampling exponent
    "beta_increment": 0.001
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    "enabled": True,
    "real_time": True,
    "save_plots": True,
    "plot_style": "seaborn-v0_8",
    "figure_size": (15, 10),
    "dpi": 300
}

# Database Configuration
DATABASE_CONFIG = {
    "enabled": True,
    "path": "training_results.db",
    "backup_frequency": 1000,  # Backup every N episodes
    "retention_days": 30        # Keep data for N days
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    "enabled": True,
    "host": "localhost",
    "port": 8050,
    "debug": False,
    "auto_refresh": True,
    "refresh_interval": 2000  # milliseconds
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "training.log",
    "max_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# Device Configuration
DEVICE_CONFIG = {
    "auto_detect": True,
    "preferred_device": "cuda",  # "cuda", "cpu", or "auto"
    "mixed_precision": False,    # Use automatic mixed precision
    "compile_model": False       # Use torch.compile (PyTorch 2.0+)
}

# Path Configuration
PATHS_CONFIG = {
    "models_dir": "models",
    "plots_dir": "plots",
    "logs_dir": "logs",
    "data_dir": "data",
    "checkpoints_dir": "checkpoints"
}

# Experiment Configuration
EXPERIMENT_CONFIG = {
    "name": "DQN_Experiment",
    "description": "Deep Q-Network training experiment",
    "tags": ["dqn", "cartpole", "reinforcement_learning"],
    "seed": 42,
    "reproducible": True
}

def get_config() -> Dict[str, Any]:
    """Get the complete configuration dictionary"""
    return {
        "environment": ENVIRONMENT_CONFIG,
        "agent": AGENT_CONFIG,
        "training": TRAINING_CONFIG,
        "network": NETWORK_CONFIG,
        "replay_buffer": REPLAY_BUFFER_CONFIG,
        "visualization": VISUALIZATION_CONFIG,
        "database": DATABASE_CONFIG,
        "dashboard": DASHBOARD_CONFIG,
        "logging": LOGGING_CONFIG,
        "device": DEVICE_CONFIG,
        "paths": PATHS_CONFIG,
        "experiment": EXPERIMENT_CONFIG
    }

def setup_directories():
    """Create necessary directories"""
    paths = PATHS_CONFIG
    for dir_name in paths.values():
        os.makedirs(dir_name, exist_ok=True)

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters"""
    try:
        # Validate agent config
        agent = config["agent"]
        assert 0 < agent["lr"] <= 1, "Learning rate must be between 0 and 1"
        assert 0 < agent["gamma"] <= 1, "Gamma must be between 0 and 1"
        assert 0 <= agent["epsilon"] <= 1, "Epsilon must be between 0 and 1"
        assert agent["epsilon_decay"] < 1, "Epsilon decay must be less than 1"
        assert agent["min_epsilon"] >= 0, "Min epsilon must be non-negative"
        assert agent["batch_size"] > 0, "Batch size must be positive"
        
        # Validate training config
        training = config["training"]
        assert training["episodes"] > 0, "Episodes must be positive"
        assert training["eval_episodes"] > 0, "Eval episodes must be positive"
        
        # Validate replay buffer config
        buffer = config["replay_buffer"]
        assert buffer["capacity"] > 0, "Buffer capacity must be positive"
        assert 0 <= buffer["alpha"] <= 1, "Alpha must be between 0 and 1"
        assert 0 <= buffer["beta"] <= 1, "Beta must be between 0 and 1"
        
        return True
    except AssertionError as e:
        print(f"Configuration validation error: {e}")
        return False
    except KeyError as e:
        print(f"Missing configuration key: {e}")
        return False

def print_config(config: Dict[str, Any]):
    """Print configuration in a readable format"""
    print("🔧 Deep Reinforcement Learning Configuration")
    print("=" * 50)
    
    for section_name, section_config in config.items():
        print(f"\n📋 {section_name.upper()}:")
        for key, value in section_config.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")

if __name__ == "__main__":
    # Example usage
    config = get_config()
    
    if validate_config(config):
        print("✅ Configuration is valid!")
        setup_directories()
        print("📁 Directories created successfully!")
        
        # Print configuration
        print_config(config)
    else:
        print("❌ Configuration validation failed!")
