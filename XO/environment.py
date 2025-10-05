import random

class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = [0] * 9  # 0=empty, 1=X, -1=O
        return tuple(self.state)

    def get_current_player(self):
        num_X = sum(1 for x in self.state if x == 1)
        num_O = sum(1 for x in self.state if x == -1)
        return 0 if num_X == num_O else 1

    def get_possible_actions(self):
        return [i for i in range(9) if self.state[i] == 0]

    def step(self, action):
        assert self.state[action] == 0, "Invalid action"
        player = self.get_current_player()
        mark = 1 if player == 0 else -1
        self.state[action] = mark
        done = self.is_done()
        reward = 0
        if done:
            if self.check_win(player):
                reward = 1
            elif self.check_draw():
                reward = 0
        return tuple(self.state), reward, done, {}

    def is_done(self):
        return self.check_win(0) or self.check_win(1) or self.check_draw()

    def check_win(self, player):
        mark = 1 if player == 0 else -1
        win_combinations = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        return any(self.state[a] == self.state[b] == self.state[c] == mark
                   for a, b, c in win_combinations)

    def check_draw(self):
        return all(x != 0 for x in self.state)

    def copy(self):
        new_env = TicTacToeEnv()
        new_env.state = self.state.copy()
        return new_env

    def get_state(self):
        return tuple(self.state)

    def set_state(self, state):
        self.state = list(state)