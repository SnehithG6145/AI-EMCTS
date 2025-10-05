#!/usr/bin/env python3
"""
Run script for Elastic MCTS project
This script ensures all directories exist and runs the simulation
"""
import os
import sys
import subprocess

def main():
    # Ensure all directories exist
    for directory in ['env', 'mcts', 'analysis']:
        os.makedirs(directory, exist_ok=True)
    
    # Check if all required files exist
    required_files = [
        'main.py',
        'simulation_ktk_multi.py',
        'env/ktk.py',
        'env/__init__.py',
        'mcts/mcts_standard.py',
        'mcts/mcts_random_group.py',
        'mcts/mcts_elastic_unit.py',
        'mcts/__init__.py',
        'analysis/plot_graphs_multi.py',
        'analysis/__init__.py'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Error: The following required files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure all project files are in the correct locations.")
        return 1
    
    # Run the main script
    print("Starting Elastic MCTS simulation...")
    try:
        subprocess.run([sys.executable, 'main.py'], check=True)
        return 0
    except subprocess.CalledProcessError:
        print("Error: The simulation failed to run.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
