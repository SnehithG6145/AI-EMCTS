import os
import sys

# Ensure all directories are in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create directories if they don't exist
for directory in ['env', 'mcts', 'analysis']:
    os.makedirs(directory, exist_ok=True)

from simulation_ktk_multi import simulate_game

if __name__ == "__main__":
    print("Starting Elastic MCTS Simulation...")
    try:
        simulate_game()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the README.md file for troubleshooting information.")
