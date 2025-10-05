# location: mcts/mcts_standard.py

import math
import random

class Node:
    """Tree node for standard MCTS"""
    def __init__(self, state, parent=None, action=None):
        self.state = state  # (board, alive, player, unit_type_idx)
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0

    def ucb1(self, parent_visits):
        """UCB1 value for this node (Equation 1 in the paper)"""
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

def mcts_standard(env, player, iterations):
    """
    Standard MCTS implementation as described in the paper
    """
    root = Node(env.get_state())
    actions = env.get_possible_actions()
    
    if not actions:
        return actions[0] if actions else None, 0

    for _ in range(iterations):
        node = root
        sim_env = env.copy()
        
        # Selection phase
        while node.is_fully_expanded(sim_env) and node.children:
            node = max(node.children, key=lambda c: c.ucb1(node.visits))
            sim_env, _ = sim_env.step(node.action)
        
        # Expansion phase
        if not node.is_fully_expanded(sim_env):
            node.fully_expand(sim_env)
            
            if node.children:
                # Select an unexplored child
                unexplored = [c for c in node.children if c.visits == 0]
                if unexplored:
                    # Prioritize attack actions during expansion
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
            node.visits += 1
            node.value += reward
            node = node.parent

    # Return best action using the recommendation policy
    if not root.children:
        return actions[0] if actions else None, 0
    
    # Prioritize attack actions in final selection
    attack_children = [c for c in root.children if c.action[1] == "attack"]
    if attack_children:
        best_child = max(attack_children, key=lambda c: c.visits)
    else:
        best_child = max(root.children, key=lambda c: c.visits)
    
    total_nodes = len(root.children) + sum(len(n.children) for n in root.children if hasattr(n, 'children'))
    
    return best_child.action, total_nodes



# unit ordering
# Prioritize attack actions during expansion, rollout, final selection, L62, 79, 118
