# location: simulation_ktk_multi.py

from env.ktk import KTK
from mcts.mcts_standard import mcts_standard
from mcts.mcts_random_group import mcts_random_group
from mcts.mcts_elastic_unit import mcts_elastic_unit
from analysis.plot_graphs_multi import plot_results
import time
import traceback
import numpy as np
import random
import os
import datetime

def get_user_input(prompt, default_value, value_type=int):
    """Get user input with a default value"""
    try:
        user_input = input(f"{prompt} [{default_value}]: ")
        if user_input.strip() == "":
            return default_value
        return value_type(user_input)
    except ValueError:
        print(f"Invalid input. Using default value: {default_value}")
        return default_value

def format_action(action):
    """Format action for display, handling numpy types"""
    if action is None:
        return "None"
    
    unit_id, action_type, target = action
    
    # Convert numpy integers to Python integers
    if isinstance(unit_id, np.integer):
        unit_id = int(unit_id)
    
    if isinstance(target, np.integer):
        target = int(target)
    elif isinstance(target, np.ndarray):
        target = tuple(int(x) for x in target)
    
    return (unit_id, action_type, target)

def randomize_parameters():
    """Generate random parameters for the simulation"""
    batch_size = random.randint(10, 30)
    alpha_abs = batch_size * random.randint(4, 10)
    iterations = random.randint(30, 70)
    max_turns = random.randint(15, 25)
    eta_r = random.uniform(0.05, 0.2)
    eta_t = random.uniform(0.5, 1.5)
    board_size = random.randint(4, 6)
    
    return {
        "batch_size": batch_size,
        "alpha_abs": alpha_abs,
        "iterations": iterations,
        "max_turns": max_turns,
        "eta_r": eta_r,
        "eta_t": eta_t,
        "board_size": board_size
    }

def simulate_game(use_random_params=False, random_seed=None):
    """
    Run the KTK game simulation with MCTS variants
    
    Parameters:
    - use_random_params: Whether to use randomized parameters (default: False)
    - random_seed: Seed for random number generation (default: None)
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    else:
        # Use current time as seed
        current_time = int(time.time())
        random.seed(current_time)
        np.random.seed(current_time)
    
    print("=== KTK Multi-Agent Simulation (Elastic MCTS) ===\n")
    
    # Get parameters - either from user input or randomized
    if use_random_params:
        params = randomize_parameters()
        BATCH_SIZE = params["batch_size"]
        ALPHA_ABS = params["alpha_abs"]
        ITERATIONS = params["iterations"]
        MAX_TURNS = params["max_turns"]
        ETA_R = params["eta_r"]
        ETA_T = params["eta_t"]
        BOARD_SIZE = params["board_size"]
        
        print("Using randomized parameters:")
    else:
        # Get user input for parameters with defaults
        print("Please enter parameters (press Enter to use default values):")
        BATCH_SIZE = get_user_input("Batch Size - Number of iterations between abstractions", 20)
        ALPHA_ABS = get_user_input("Alpha ABS - Iteration threshold for abstraction", 160)
        ITERATIONS = get_user_input("MCTS Iterations - Number of MCTS iterations per decision", 50)
        MAX_TURNS = get_user_input("Max Turns - Maximum number of turns before counting pieces", 20)
        ETA_R = get_user_input("Eta R - Reward function error threshold", 0.1, float)
        ETA_T = get_user_input("Eta T - Transition probability error threshold", 1.0, float)
        BOARD_SIZE = get_user_input("Board Size - Size of the game board", 4)
    
    print(f"\nParameters:")
    print(f"- BATCH_SIZE: {BATCH_SIZE} (iterations between abstractions)")
    print(f"- ALPHA_ABS: {ALPHA_ABS} (iteration threshold for abstraction)")
    print(f"- ITERATIONS: {ITERATIONS} (MCTS iterations per decision)")
    print(f"- MAX_TURNS: {MAX_TURNS} (maximum turns before counting pieces)")
    print(f"- ETA_R: {ETA_R:.2f} (reward function error threshold)")
    print(f"- ETA_T: {ETA_T:.2f} (transition probability error threshold)")
    print(f"- BOARD_SIZE: {BOARD_SIZE} (size of the game board)")
    print("\n")
    
    # Create the environment with random setup
    env = KTK(board_size=BOARD_SIZE, max_turns=MAX_TURNS, random_setup=True)
    players = [0, 1]
    turn = 1
    results = {
        "iterations": [],
        "standard_nodes": [],
        "random_nodes": [],
        "elastic_nodes": [],
        "standard_choices": [],
        "random_choices": [],
        "elastic_choices": [],
        "elastic_ground_nodes": [],
        "elastic_abs_nodes": []
    }

    print("Initial state:")
    env.display()
    print("\n")

    try:
        while not env.is_done() and turn <= MAX_TURNS:
            for player in players:
                if env.is_done():
                    break
                    
                print(f"\n=== Turn {turn}: Player {player}'s turn ===")
                
                # Run all three MCTS variants with error handling
                try:
                    start_time = time.time()
                    print("Running Standard MCTS...")
                    std_action, std_nodes = mcts_standard(env.copy(), player, ITERATIONS)
                    std_time = time.time() - start_time
                    
                    start_time = time.time()
                    print("Running Random Grouping MCTS...")
                    rg_action, rg_nodes = mcts_random_group(env.copy(), player, ITERATIONS, BATCH_SIZE, ALPHA_ABS)
                    rg_time = time.time() - start_time
                    
                    start_time = time.time()
                    print("Running Elastic MCTS with Unit Ordering...")
                    elastic_action, elastic_nodes, ground_nodes, abs_nodes = mcts_elastic_unit(
                        env.copy(), player, ITERATIONS, BATCH_SIZE, ALPHA_ABS, ETA_R, ETA_T
                    )
                    elastic_time = time.time() - start_time

                    # Display the actions recommended by each algorithm
                    print(f"\nRecommended actions:")
                    print(f"Standard MCTS: {format_action(std_action)} (computed in {std_time:.2f}s)")
                    print(f"Random Grouping MCTS: {format_action(rg_action)} (computed in {rg_time:.2f}s)")
                    print(f"Elastic MCTS: {format_action(elastic_action)} (computed in {elastic_time:.2f}s)")
                    
                    # Randomly choose which algorithm's action to use (adds more variability)
                    if random.random() < 0.2:  # 20% chance to use a different algorithm
                        chosen_algo = random.choice(["Standard", "Random Grouping", "Elastic"])
                        if chosen_algo == "Standard":
                            chosen_action = std_action
                        elif chosen_algo == "Random Grouping":
                            chosen_action = rg_action
                        else:
                            chosen_action = elastic_action
                        print(f"\nRandomly selected {chosen_algo} MCTS action for execution")
                    else:
                        chosen_action = elastic_action
                        chosen_algo = "Elastic"
                    
                    print(f"\nPlayer {player} executes action: {format_action(chosen_action)}")
                    env, done = env.step(chosen_action)
                    
                    print("\nNew game state:")
                    # Only show kings' status
                    env.display(show_kings_only=True)

                    # Record metrics for plotting
                    next_choices = len(env.get_possible_actions())
                    results["iterations"].append(turn)
                    results["standard_nodes"].append(std_nodes)
                    results["random_nodes"].append(rg_nodes)
                    results["elastic_nodes"].append(elastic_nodes)
                    results["standard_choices"].append(next_choices)
                    results["random_choices"].append(next_choices)
                    results["elastic_choices"].append(next_choices)
                    results["elastic_ground_nodes"].append(ground_nodes)
                    results["elastic_abs_nodes"].append(abs_nodes)

                    print(f"\nMetrics:")
                    print(f"Next turn choices: {next_choices}")
                    print(f"Node counts: Standard: {std_nodes}, RG: {rg_nodes}, Elastic: {elastic_nodes}")
                    print(f"Elastic - Ground nodes: {ground_nodes}, Abstract nodes: {abs_nodes}")
                    
                    if abs_nodes > 0:
                        compression = ground_nodes / abs_nodes
                        print(f"Compression rate: {compression:.2f}")

                    if done:
                        print("\n=== Game Over! ===")
                        env.display()
                        winner = env.winner
                        print(f"Player {winner} wins!")
                        break
                        
                except Exception as e:
                    print(f"Error during MCTS computation: {e}")
                    traceback.print_exc()
                    # Continue with a default action
                    actions = env.get_possible_actions()
                    if actions:
                        action = actions[0]
                        print(f"Using default action: {format_action(action)}")
                        env, _ = env.step(action)
                        env.display(show_kings_only=True)
            
            turn += 1

        if not env.is_done():
            print("\n=== Game reached maximum turns! ===")
            # Count pieces to determine winner
            player0_pieces = sum(1 for i in range(1, 5) if env.alive[i])
            player1_pieces = sum(1 for i in range(5, 9) if env.alive[i])
            
            if player0_pieces > player1_pieces:
                print(f"Player 0 wins with {player0_pieces} pieces vs {player1_pieces}!")
            elif player1_pieces > player0_pieces:
                print(f"Player 1 wins with {player1_pieces} pieces vs {player0_pieces}!")
            else:
                print(f"The game is a draw with {player0_pieces} pieces each!")
        
        # Create a timestamp for the output file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"elastic_mcts_results_{timestamp}.png"
        
        print(f"\nGenerating result plots as {output_filename}...")
        plot_results(results, ALPHA_ABS, output_filename)
        print(f"Simulation complete! Results saved to {output_filename}")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        traceback.print_exc()
        # Still try to generate plots with data collected so far
        if results["iterations"]:
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"elastic_mcts_results_{timestamp}.png"
                plot_results(results, ALPHA_ABS, output_filename)
            except Exception as plot_error:
                print(f"Error generating plots: {plot_error}")

if __name__ == "__main__":
    simulate_game()
