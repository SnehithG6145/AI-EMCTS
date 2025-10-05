import numpy as np
import math
import random
from environment import TicTacToeEnv

BATCH_SIZE = 10
ALPHA_ABS = 30
C_PARAM = 1.4

SYMMETRY_PERMUTATIONS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 5, 8, 1, 4, 7, 0, 3, 6],
    [8, 7, 6, 5, 4, 3, 2, 1, 0], [6, 3, 0, 7, 4, 1, 8, 5, 2],
    [0, 3, 6, 1, 4, 7, 2, 5, 8], [2, 1, 0, 5, 4, 3, 8, 7, 6],
    [8, 5, 2, 7, 4, 1, 6, 3, 0], [6, 7, 8, 3, 4, 5, 0, 1, 2]
]

def apply_permutation(state, perm):
    return tuple(state[i] for i in perm)

def get_canonical_state(state):
    return min(apply_permutation(state, perm) for perm in SYMMETRY_PERMUTATIONS)

class Group:
    def __init__(self, nodes):
        self.nodes = nodes
        self.value = 0
        self.visits = 0
        for node in nodes:
            node.group = self

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        self.group = None

    def is_fully_expanded(self, env):
        return len(self.children) == len(env.get_possible_actions())

    def best_child(self, c_param=C_PARAM):
        weights = []
        for child in self.children:
            if child.group:
                value, visits = child.group.value, child.group.visits
            else:
                value, visits = child.value, child.visits
            weights.append(
                (value / (visits + 1e-5)) +
                c_param * math.sqrt((2 * math.log(self.visits + 1)) / (visits + 1e-5))
            )
        return self.children[np.argmax(weights)]

    def expand(self, env):
        actions = env.get_possible_actions()
        tried = [child.action for child in self.children]
        untried = [a for a in actions if a not in tried]
        if untried:
            action = random.choice(untried)
            new_env = env.copy()
            next_state, _, _, _ = new_env.step(action)
            child = Node(next_state, self, action)
            self.children.append(child)
            return child
        return self

    def fully_expand(self, env):
        while not self.is_fully_expanded(env):
            self.expand(env)

    def backpropagate(self, reward):
        if self.group:
            self.group.value += reward
            self.group.visits += 1
        else:
            self.value += reward
            self.visits += 1
        if self.parent:
            self.parent.backpropagate(reward)

def collect_nodes(node):
    nodes = [node]
    for child in node.children:
        nodes.extend(collect_nodes(child))
    return nodes

def construct_abstraction_symmetry(node, env):
    if not node.is_fully_expanded(env):
        node.fully_expand(env)
    canonical_to_group = {}
    for child in node.children:
        canonical = get_canonical_state(child.state)
        if canonical not in canonical_to_group:
            canonical_to_group[canonical] = Group([child])
        else:
            canonical_to_group[canonical].nodes.append(child)
            child.group = canonical_to_group[canonical]
    return list(canonical_to_group.values())

def tree_policy(node, env):
    current = node
    sim_env = env.copy()
    while not sim_env.is_done():
        if not current.is_fully_expanded(sim_env):
            return current.expand(sim_env)
        current = current.best_child()
        sim_env.step(current.action)
    return current

def default_policy(env, player):
    sim_env = env.copy()
    while not sim_env.is_done():
        action = random.choice(sim_env.get_possible_actions())
        sim_env.step(action)
    if sim_env.check_win(player):
        return 1
    elif sim_env.check_win(1 - player):
        return -1
    return 0

def mcts_elastic(env, player, num_simulations):
    root = Node(env.get_state())
    root.fully_expand(env)
    groups = []
    for i in range(num_simulations):
        sim_env = env.copy()
        node = tree_policy(root, sim_env)
        reward = default_policy(sim_env, player)
        node.backpropagate(reward)
        if (i + 1) % BATCH_SIZE == 0 and i < ALPHA_ABS:
            groups = construct_abstraction_symmetry(root, env)
        elif i == ALPHA_ABS:
            for group in groups:
                for n in group.nodes:
                    n.value = group.value
                    n.visits = group.visits
                    n.group = None
            groups = []
    final_groups = construct_abstraction_symmetry(root, env)
    best_group = max(final_groups, key=lambda g: g.visits)
    best_child = max(best_group.nodes, key=lambda n: n.visits)
    return best_child.action, len(collect_nodes(root))