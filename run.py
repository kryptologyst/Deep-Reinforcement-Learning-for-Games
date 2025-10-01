#!/usr/bin/env python3
"""
Quick start script for Deep Reinforcement Learning DQN
Provides easy commands to run different components of the project.
"""

import sys
import argparse
import subprocess
import os
from pathlib import Path

def run_training(episodes=500, render=False, config_file=None):
    """Run DQN training"""
    print(f"🎮 Starting DQN training for {episodes} episodes...")
    
    cmd = [sys.executable, "0140.py"]
    if config_file:
        cmd.extend(["--config", config_file])
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return False
    
    return True

def run_dashboard(port=8050, debug=False):
    """Run the web dashboard"""
    print(f"🌐 Starting web dashboard on port {port}...")
    
    cmd = [sys.executable, "dashboard.py"]
    if debug:
        cmd.append("--debug")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Dashboard failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
        return False
    
    return True

def run_demo():
    """Run the demo script"""
    print("🎯 Running DQN demo...")
    
    try:
        subprocess.run([sys.executable, "demo.py"], check=True)
        print("✅ Demo completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed: {e}")
        return False
    
    return True

def run_tests(verbose=False, coverage=False):
    """Run unit tests"""
    print("🧪 Running unit tests...")
    
    cmd = [sys.executable, "-m", "pytest", "test_dqn.py"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=html"])
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ All tests passed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        return False
    
    return True

def setup_project():
    """Setup the project"""
    print("🔧 Setting up project...")
    
    try:
        subprocess.run([sys.executable, "setup.py"], check=True)
        print("✅ Project setup completed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Setup failed: {e}")
        return False
    
    return True

def check_requirements():
    """Check if requirements are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        "torch", "gymnasium", "numpy", "matplotlib", 
        "seaborn", "dash", "sqlite3"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "sqlite3":
                import sqlite3
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run 'python setup.py' to install requirements")
        return False
    
    print("✅ All requirements satisfied!")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Deep Reinforcement Learning DQN - Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py train                    # Start training
  python run.py train --episodes 1000   # Train for 1000 episodes
  python run.py dashboard                # Start web dashboard
  python run.py demo                     # Run demo
  python run.py test                     # Run tests
  python run.py setup                    # Setup project
  python run.py check                    # Check requirements
        """
    )
    
    parser.add_argument(
        "command",
        choices=["train", "dashboard", "demo", "test", "setup", "check"],
        help="Command to run"
    )
    
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=500,
        help="Number of training episodes (default: 500)"
    )
    
    parser.add_argument(
        "--render", "-r",
        action="store_true",
        help="Render environment during training"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8050,
        help="Dashboard port (default: 8050)"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--config", "-f",
        type=str,
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    print("🎮 Deep Reinforcement Learning DQN - Quick Start")
    print("=" * 50)
    
    # Execute command
    if args.command == "train":
        return run_training(args.episodes, args.render, args.config)
    
    elif args.command == "dashboard":
        return run_dashboard(args.port, args.debug)
    
    elif args.command == "demo":
        return run_demo()
    
    elif args.command == "test":
        return run_tests(args.verbose, args.coverage)
    
    elif args.command == "setup":
        return setup_project()
    
    elif args.command == "check":
        return check_requirements()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
