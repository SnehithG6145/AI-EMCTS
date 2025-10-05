# location: mcts/mcts_elastic_unit.py

import math
import random
import numpy as np

class Group:
    """Group of nodes that share statistics"""
    def __init__(self, nodes):
        self.nodes = nodes
        self.visits = 0
        self.value = 0
        for node in nodes:
            node.group = self

    def ucb1(self, parent_visits):
        """UCB1 value for this group (Equation 1 in the paper)"""
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + math.sqrt(2 * math.log(parent_visits) / self.visits)

class Node:
    """Tree node for MCTS"""
    def __init__(self, state, parent=None, action=None):
        self.state = state  # (board, alive, player, unit_type_idx)
        self.parent = parent
        self.action = action
        self.children = []
        self.group = None
        self.visits = 0
        self.value = 0

    def ucb1(self, parent_visits):
        """UCB1 value for this node (Equation 1 in the paper)"""
        if self.group:
            return self.group.ucb1(parent_visits)
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + math.sqrt(2 * math.log(parent_visits) / self.visits)

    def fully_expand(self, env):
        """Expand this node with all possible actions"""
        if not self.children:
            actions = env.get_possible_actions()
            for action in actions:
                sim_env = env.copy()
                next_state, _ = sim_env.step(action)
                self.children.append(Node(next_state.get_state(), self, action))

    def is_fully_expanded(self, env):
        """Check if this node is fully expanded"""
        return len(self.children) == len(env.get_possible_actions())

# Approximate MDP Homomorphism
def state_similarity(s1, s2, eta_r=0.1, eta_t=1.0):
    """
    Compute similarity between two states
    s1 and s2 are tuples of (board, alive, player, unit_type_idx)
    Implements approximate MDP homomorphism as described in the paper
    
    Parameters:
    - s1, s2: States to compare
    - eta_r: Reward function error threshold (default: 0.1)
    - eta_t: Transition probability error threshold (default: 1.0)
    """
    board1, alive1, player1, unit_idx1 = s1
    board2, alive2, player2, unit_idx2 = s2
    
    # If different players or unit types, states are not similar
    if player1 != player2 or unit_idx1 != unit_idx2:
        return False
    
    # Compare board states (simplified)
    board_diff = np.sum(board1 != board2)
    if board_diff > 2:  # Allow small differences
        return False
    
    # Compare alive status
    alive_diff = sum(alive1[k] != alive2[k] for k in alive1 if k in alive2)
    if alive_diff > 1:  # Allow small differences
        return False
    
    # Check if both states have similar attack opportunities
    attack_opportunities1 = has_attack_opportunity(board1, player1)
    attack_opportunities2 = has_attack_opportunity(board2, player2)
    if attack_opportunities1 != attack_opportunities2:
        return False
    
    return True

def has_attack_opportunity(board, player):
    """Check if the player has an opportunity to attack in the given board state"""
    player_units = []
    enemy_units = []
    
    # Find player and enemy units
    for i in range(len(board)):
        for j in range(len(board[i])):
            unit_id = board[i, j]
            if unit_id > 0:
                unit_owner = 0 if unit_id < 5 else 1
                if unit_owner == player:
                    player_units.append((i, j, unit_id))
                else:
                    enemy_units.append((i, j, unit_id))
    
    # Check if any player unit is adjacent to an enemy unit
    for pi, pj, _ in player_units:
        for ei, ej, _ in enemy_units:
            if abs(pi - ei) <= 2 and abs(pj - ej) <= 2:  # Within attack range
                return True
    
    return False

def construct_abstraction(state_node, env, eta_r=0.1, eta_t=1.0):
    """
    Implement Algorithm 3 from the paper: ConstructAbstraction
    Group similar nodes based on approximate MDP homomorphism
    
    Parameters:
    - state_node: The node to construct abstraction for
    - env: Game environment
    - eta_r: Reward function error threshold (default: 0.1)
    - eta_t: Transition probability error threshold (default: 1.0)
    """
    if not state_node.is_fully_expanded(env):
        state_node.fully_expand(env)
    
    if not state_node.children:
        return []
    
    # Separate attack actions to prioritize them (Unit Ordering - MCTSu)
    attack_children = [c for c in state_node.children if c.action[1] == "attack"]
    other_children = [c for c in state_node.children if c.action[1] != "attack"]
    
    groups = []
    
    # Create a group for attack actions if any exist
    if attack_children:
        groups.append(Group(attack_children))
    
    # Group other actions based on state similarity (Dynamic State Grouping)
    ungrouped = other_children.copy()
    
    while ungrouped:
        child = ungrouped.pop(0)
        matched = False
        
        # Try to find a matching group
        for group in groups:
            rep = group.nodes[0]  # Representative node of the group
            
            # Check if states are similar using approximate MDP homomorphism
            if state_similarity(rep.state, child.state, eta_r, eta_t):
                group.nodes.append(child)
                child.group = group
                matched = True
                break
        
        # If no matching group, create a new one
        if not matched:
            groups.append(Group([child]))
    
    return groups

def mcts_elastic_unit(env, player, iterations, batch_size=20, alpha_abs=160, eta_r=0.1, eta_t=1.0):
    """
    Implement Algorithm 1 and 2 from the paper: Elastic MCTS
    
    Parameters:
    - env: Game environment
    - player: Current player (0 or 1)
    - iterations: Number of MCTS iterations
    - batch_size: Number of iterations between abstractions (default: 20)
    - alpha_abs: Iteration threshold for abstraction (default: 160)
    - eta_r: Reward function error threshold (default: 0.1)
    - eta_t: Transition probability error threshold (default: 1.0)
    """
    root = Node(env.get_state())
    actions = env.get_possible_actions()
    
    if not actions:
        return (0, "wait", None), 0, 0, 0

    ground_nodes = 0
    abs_nodes = 0
    current_iteration = 0
    
    # Main loop (Algorithm 1)
    while current_iteration < iterations:
        # MCTS Iteration (Algorithm 2)
        node = root
        sim_env = env.copy()
        
        # Selection phase
        while node.is_fully_expanded(sim_env) and node.children:
            if node.children[0].group:
                # Get unique groups
                groups = list({c.group for c in node.children if c.group})
                abs_nodes = len(groups)
                
                if groups:
                    # Select best group by UCB1
                    group = max(groups, key=lambda g: g.ucb1(node.visits))
                    # Randomly select a node from the group
                    node = random.choice(group.nodes)
                else:
                    # If no groups, use standard UCB1
                    node = max(node.children, key=lambda c: c.ucb1(node.visits))
            else:
                # Standard UCB1 selection
                node = max(node.children, key=lambda c: c.ucb1(node.visits))
            
            # Apply the action
            sim_env, _ = sim_env.step(node.action)
        
        # Expansion phase
        if not node.is_fully_expanded(sim_env):
            node.fully_expand(sim_env)
            ground_nodes += len(node.children)
            
            # Apply abstraction based on iteration threshold
            if current_iteration < alpha_abs and current_iteration % batch_size == 0 and node.children:
                construct_abstraction(node, sim_env, eta_r, eta_t)
            elif current_iteration >= alpha_abs and node.children:
                # Split groups after threshold (as per paper)
                for child in node.children:
                    child.group = None
            
            if node.children:
                # Select an unexplored child
                unexplored = [c for c in node.children if c.visits == 0]
                if unexplored:
                    # Prioritize attack actions
                    attack_nodes = [c for c in unexplored if c.action[1] == "attack"]
                    if attack_nodes:
                        node = random.choice(attack_nodes)
                    else:
                        node = random.choice(unexplored)
                    sim_env, _ = sim_env.step(node.action)
        
        # Simulation phase
        sim_count = 0
        max_sim_depth = 20  # Reduced from 40 to speed up simulation
        
        while not sim_env.is_done() and sim_count < max_sim_depth:
            actions = sim_env.get_possible_actions()
            if not actions:
                break
            
            # Aggressive rollout policy - prefer attacks
            attack_actions = [a for a in actions if a[1] == "attack"]
            if attack_actions:
                action = random.choice(attack_actions)
            else:
                action = random.choice(actions)
            sim_env, _ = sim_env.step(action)
            sim_count += 1
        
        # Reward calculation
        if sim_env.is_done():
            # Win/loss reward
            reward = 1 if sim_env.winner == player else -1
        else:
            # Heuristic reward based on piece count difference
            if player == 0:
                player_pieces = sum(1 for i in range(1, 5) if sim_env.alive[i])
                opponent_pieces = sum(1 for i in range(5, 9) if sim_env.alive[i])
            else:
                player_pieces = sum(1 for i in range(5, 9) if sim_env.alive[i])
                opponent_pieces = sum(1 for i in range(1, 5) if sim_env.alive[i])
                
            reward = (player_pieces - opponent_pieces) / 8.0  # Normalize
            
            # Bonus for killing enemy king
            enemy_king_id = 5 if player == 0 else 1
            if not sim_env.alive[enemy_king_id]:
                reward += 0.5
        
        # Back-propagation
        while node:
            if node.group:
                node.group.visits += 1
                node.group.value += reward
            
            node.visits += 1
            node.value += reward
            node = node.parent
        
        current_iteration += 1

    # Return best action using the recommendation policy
    if not root.children:
        return actions[0] if actions else None, 0, 0, 0
    
    # Prioritize attack actions in final selection
    attack_children = [c for c in root.children if c.action[1] == "attack"]
    if attack_children:
        best_child = max(attack_children, key=lambda c: c.group.visits if c.group else c.visits)
    else:
        best_child = max(root.children, key=lambda c: c.group.visits if c.group else c.visits)
    
    total_nodes = len(root.children) + sum(len(n.children) for n in root.children if hasattr(n, 'children'))
    
    return best_child.action, total_nodes, ground_nodes, abs_nodes





# Compression Rate, 
# abs_nodes = len(groups) L199, 
# ground_nodes += len(node.children) L219, 
# return best_child.action, total_nodes, ground_nodes, abs_nodes L304, 

# compression = ground_nodes / abs_nodes @simulation_ktk_multi.py L209, 




# grouping and ungrouping L224
# Unit Ordering



