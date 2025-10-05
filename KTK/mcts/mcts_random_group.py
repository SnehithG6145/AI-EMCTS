# location: mcts/mcts_random_group.py

import math
import random

class Group:
    """Group of nodes that share statistics"""
    def __init__(self, nodes):
        self.nodes = nodes
        self.visits = 0
        self.value = 0
        for node in nodes:
            node.group = self

    def ucb1(self, parent_visits):
        """UCB1 value for this group"""
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + math.sqrt(2 * math.log(parent_visits) / self.visits)

class Node:
    """Tree node for Random Grouping MCTS"""
    def __init__(self, state, parent=None, action=None):
        self.state = state  # (board, alive, player, unit_type_idx)
        self.parent = parent
        self.action = action
        self.children = []
        self.group = None
        self.visits = 0
        self.value = 0

    def ucb1(self, parent_visits):
        """UCB1 value for this node"""
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

def construct_random_groups(children):
    """
    Randomly group nodes as described in the paper for RG MCTS_u
    """
    if not children:
        return []
    
    # Separate attack actions to prioritize them
    attack_children = [c for c in children if c.action[1] == "attack"]
    other_children = [c for c in children if c.action[1] != "attack"]
    
    groups = []
    
    # Create a group for attack actions if any exist
    if attack_children:
        groups.append(Group(attack_children))
    
    # Randomly group the remaining actions
    if other_children:
        random.shuffle(other_children)
        # Create random groups with approximately equal sizes
        group_size = max(1, len(other_children) // 3)  # Divide into approximately 3 groups
        
        for i in range(0, len(other_children), group_size):
            group_nodes = other_children[i:i + group_size]
            if group_nodes:
                groups.append(Group(group_nodes))
    
    return groups

def mcts_random_group(env, player, iterations, batch_size=20, alpha_abs=160):
    """
    Random Grouping MCTS implementation as described in the paper
    
    Parameters:
    - env: Game environment
    - player: Current player (0 or 1)
    - iterations: Number of MCTS iterations
    - batch_size: Number of iterations between abstractions (default: 20)
    - alpha_abs: Iteration threshold for abstraction (default: 160)
    """
    root = Node(env.get_state())
    actions = env.get_possible_actions()
    
    if not actions:
        return actions[0] if actions else None, 0

    current_iteration = 0

    while current_iteration < iterations:
        node = root
        sim_env = env.copy()
        
        # Selection phase
        while node.is_fully_expanded(sim_env) and node.children:
            if node.children[0].group:
                # Get unique groups
                groups = list({c.group for c in node.children if c.group})
                
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
            
            # Apply random grouping based on iteration threshold
            if current_iteration < alpha_abs and current_iteration % batch_size == 0 and node.children:
                construct_random_groups(node.children)
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
        return actions[0] if actions else None, 0
    
    # Prioritize attack actions in final selection
    attack_children = [c for c in root.children if c.action[1] == "attack"]
    if attack_children:
        best_child = max(attack_children, key=lambda c: c.group.visits if c.group else c.visits)
    else:
        best_child = max(root.children, key=lambda c: c.group.visits if c.group else c.visits)
    
    total_nodes = len(root.children) + sum(len(n.children) for n in root.children if hasattr(n, 'children'))
    
    return best_child.action, total_nodes
