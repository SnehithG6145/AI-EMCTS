import numpy as np
import math
import random
from environment import TicTacToeEnv

C_PARAM = 1.4

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self, env):
        return len(self.children) == len(env.get_possible_actions())

    def best_child(self, c_param=C_PARAM):
        weights = [
            (child.value / (child.visits + 1e-5)) +
            c_param * math.sqrt((2 * math.log(self.visits + 1)) / (child.visits + 1e-5))
            for child in self.children
        ]
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
        self.value += reward
        self.visits += 1
        if self.parent:
            self.parent.backpropagate(reward)

def collect_nodes(node):
    nodes = [node]
    for child in node.children:
        nodes.extend(collect_nodes(child))
    return nodes

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

def mcts_standard(env, player, num_simulations):
    root = Node(env.get_state())
    root.fully_expand(env)
    for _ in range(num_simulations):
        sim_env = env.copy()
        node = tree_policy(root, sim_env)
        reward = default_policy(sim_env, player)
        node.backpropagate(reward)
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.action, len(collect_nodes(root))