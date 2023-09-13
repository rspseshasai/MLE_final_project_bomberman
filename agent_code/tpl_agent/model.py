import numpy as np


class QLearning:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)  # Exploration: choose a random action
        else:
            return self.get_best_action(state)  # Exploitation: choose the best action

    def get_best_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)

        current_q_value = self.q_table[state][action]
        max_next_q_value = np.max(self.q_table[next_state])
        new_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * (
                    reward + self.discount_factor * max_next_q_value)
        self.q_table[state][action] = new_q_value
