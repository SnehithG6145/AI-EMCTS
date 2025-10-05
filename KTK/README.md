# Elastic MCTS for Kill The King (KTK) Game

This project implements the Elastic Monte Carlo Tree Search (MCTS) algorithm for the Kill The King (KTK) game, based on the research paper "Elastic Monte Carlo Tree Search with State Abstraction". The implementation compares three MCTS variants:

1. **Standard MCTS**: The traditional MCTS algorithm
2. **Random Grouping MCTS**: MCTS with random state grouping
3. **Elastic MCTS**: MCTS with dynamic state abstraction based on approximate MDP homomorphism

## Project Structure

- `main.py`: Entry point for running the simulation
- `simulation_ktk_multi.py`: Main simulation logic that runs all three MCTS variants and compares their performance
- `env/ktk.py`: Implementation of the Kill The King (KTK) game environment
- `mcts/mcts_standard.py`: Implementation of the standard MCTS algorithm
- `mcts/mcts_random_group.py`: Implementation of MCTS with random state grouping
- `mcts/mcts_elastic_unit.py`: Implementation of Elastic MCTS with unit ordering
- `analysis/plot_graphs_multi.py`: Functions for plotting and analyzing the results

## Key Concepts Implemented

1. **Approximate MDP Homomorphism**: Groups states that have similar reward functions and transition dynamics
2. **Dynamic State Abstraction**: Constructs abstractions during the search process
3. **Unit Ordering**: Prioritizes attack actions to improve search efficiency
4. **Compression Rate Analysis**: Measures how effectively the algorithm reduces the search space

## How to Run

\`\`\`bash
python main.py
\`\`\`

The simulation will prompt you for parameters (or use defaults) and then run a game where two agents compete using the Elastic MCTS algorithm for decision-making. The results are visualized in graphs that compare the performance of all three MCTS variants.

## Parameters

- **Batch Size (B)**: Number of iterations between abstractions (default: 20)
- **Alpha ABS (α_ABS)**: Iteration threshold for abstraction (default: 160)
- **MCTS Iterations**: Number of MCTS iterations per decision (default: 50)
- **Max Turns**: Maximum number of turns before counting pieces (default: 20)
- **Eta R (η_R)**: Reward function error threshold (default: 0.1)
- **Eta T (η_T)**: Transition probability error threshold (default: 1.0)
- **Board Size**: Size of the game board (default: 4)

## Game Rules

The Kill The King (KTK) game is a turn-based strategy game where:

1. Each player controls 4 units: King, Warrior, Archer, and Healer
2. The goal is to kill the opponent's King
3. Units take turns in a fixed order: King, Warrior, Archer, Healer
4. Units can move, attack, heal, or wait
5. If the maximum number of turns is reached, the player with more units wins

## Output

The simulation generates a graph file `elastic_mcts_results.png` with four plots:

1. **Nodes vs Iterations**: Shows how many nodes each algorithm explores
2. **Choices for Next Player**: Shows the number of available actions
3. **Compression Rate**: Shows the ratio of ground nodes to abstract nodes for Elastic MCTS
4. **Algorithm Efficiency**: Shows the nodes explored per available choice

## References

This implementation is based on the research paper "Elastic Monte Carlo Tree Search with State Abstraction" which introduces the concept of dynamic state abstraction in MCTS.
\`\`\`

## Now, let's fix the KTK environment to avoid the initialization message repetition
