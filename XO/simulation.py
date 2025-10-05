from environment import TicTacToeEnv
from mcts_standard import mcts_standard
from mcts_random_group import mcts_random_group
from mcts_elastic import mcts_elastic, get_canonical_state
import matplotlib.pyplot as plt

def print_board(state):
    board = ['X' if x == 1 else 'O' if x == -1 else ' ' for x in state]
    print(f" {' | '.join(board[0:3])}\n {' | '.join(board[3:6])}\n {' | '.join(board[6:9])}")

def get_unique_move_groups(env, player):
    actions = env.get_possible_actions()
    canonical_states = set()
    for action in actions:
        new_env = env.copy()
        new_state, _, _, _ = new_env.step(action)
        canonical_states.add(get_canonical_state(new_state))
    return len(canonical_states)

def simulate_game():
    env = TicTacToeEnv()
    state = env.reset()
    done = False
    
    mcts_choices = []
    rg_choices = []
    emcts_choices = []
    mcts_nodes = []
    rg_nodes = []
    emcts_nodes = []
    
    print("\n=== Tic-Tac-Toe Simulation ===")
    iteration = 0
    while not done:
        player = env.get_current_player()
        player_name = 'X' if player == 0 else 'O'
        
        # Run all three MCTS variants
        mcts_action, mcts_node_count = mcts_standard(env, player, 100)
        rg_action, rg_node_count = mcts_random_group(env, player, 100)
        emcts_action, emcts_node_count = mcts_elastic(env, player, 100)
        
        # Use EMCTS action for progression
        state, reward, done, _ = env.step(emcts_action)
        
        # Next player's choices
        next_player = 1 - player
        mcts_next = len(env.get_possible_actions())
        emcts_next = get_unique_move_groups(env, next_player)
        rg_next = mcts_next  # RG doesn't reduce choices inherently
        
        # Store data
        mcts_choices.append(mcts_next)
        rg_choices.append(rg_next)
        emcts_choices.append(emcts_next)
        mcts_nodes.append(mcts_node_count)
        rg_nodes.append(rg_node_count)
        emcts_nodes.append(emcts_node_count)
        
        # Print simulation
        print(f"\nIteration {iteration + 1}: Player {player_name} chooses {emcts_action}")
        print("Board:")
        print_board(state)
        print(f"Choices for Next Player - MCTS: {mcts_next}, RG MCTS: {rg_next}, EMCTS: {emcts_next}")
        print(f"Nodes - MCTS: {mcts_node_count}, RG MCTS: {rg_node_count}, EMCTS: {emcts_node_count}")
        iteration += 1
    
    # Result
    if reward == 1:
        print(f"Player {player_name} wins!")
    else:
        print("Draw!")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Choices
    ax1.plot(range(1, len(mcts_choices) + 1), mcts_choices, label='MCTS Choices', marker='o')
    ax1.plot(range(1, len(rg_choices) + 1), rg_choices, label='RG MCTS Choices', marker='o')
    ax1.plot(range(1, len(emcts_choices) + 1), emcts_choices, label='EMCTS Choices', marker='o')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Next Player Choices')
    ax1.set_title('Next Player Choices Over Iterations')
    ax1.legend()
    ax1.grid(True)
    
    # Nodes
    ax2.plot(range(1, len(mcts_nodes) + 1), mcts_nodes, label='MCTS Nodes', marker='o')
    ax2.plot(range(1, len(rg_nodes) + 1), rg_nodes, label='RG MCTS Nodes', marker='o')
    ax2.plot(range(1, len(emcts_nodes) + 1), emcts_nodes, label='EMCTS Nodes', marker='o')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Number of Nodes')
    ax2.set_title('Nodes vs. Iterations (MCTS vs. RG MCTS vs. EMCTS)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_game()