# Ktk and XO Games Execution Guide

This repository contains two games: Ktk and XO. Below are the instructions on how to execute each game.

---

## Ktk Game

The Ktk game can be executed by running the main script located at:

```
KTK/main.py
```

### How to run

1. Open a terminal or command prompt.
2. Navigate to the root directory of the project.
3. Run the following command:

```bash
python KTK/main.py
```

### Description

- This script sets up necessary directories (`env`, `mcts`, `analysis`) if they do not exist.
- It then starts the Elastic MCTS simulation by calling the `simulate_game` function from `simulation_ktk_multi`.
- If the simulation is interrupted or encounters an error, appropriate messages will be displayed.
- For troubleshooting, please refer to this README.

---

## XO Game

The XO game can be executed by running the simulation script located at:

```
XO/simulation.py
```

### How to run

1. Open a terminal or command prompt.
2. Navigate to the root directory of the project.
3. Run the following command:

```bash
python XO/simulation.py
```

### Description

- This script runs a Tic-Tac-Toe simulation using three variants of Monte Carlo Tree Search (MCTS).
- It prints the board state and player moves at each iteration.
- At the end of the simulation, it displays the result (win or draw) and plots graphs showing choices and nodes over iterations.
- The script requires `matplotlib` for plotting. Ensure it is installed via:

```bash
pip install matplotlib
```

---

## Requirements

- Python 3.x
- Required Python packages (install via pip if needed):
  - matplotlib (for XO game plotting)

---

## Troubleshooting

- If you encounter errors related to missing packages, install them using pip.
- For Ktk game errors, check the printed traceback for details.
- Ensure you are running the scripts from the root directory of the project.

---

This README provides the basic instructions to run both games. For more detailed information, please refer to the respective directories and source code.
