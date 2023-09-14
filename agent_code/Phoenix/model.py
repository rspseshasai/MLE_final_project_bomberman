import numpy as np
import torch
import torch.nn as nn


import torch
import torch.nn as nn
import numpy as np

class QLearning(nn.Module):
    def __init__(self, num_actions, input_size, hidden_size, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        super(QLearning, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Define the neural network layers here
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_actions)

        # Initialize other Q-learning related variables here
        self.q_table = {}

    def forward(self, state):
        # Implement the forward pass of your Q-learning network
        x = torch.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

    def get_action(self, state):
        # Your exploration-exploitation logic here
        state = torch.tensor(state, dtype=torch.float32)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)  # Exploration: choose a random action
        else:
            state = state.unsqueeze(0)
            q_values = self.forward(state)
            return torch.argmax(q_values).item()  # Exploitation: choose the best action

    def update_q_value(self, state, action, reward, next_state):
        # Your Q-value update logic here
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        state_action_value = self.forward(state)[action]
        next_state_value = torch.max(self.forward(next_state))
        expected_state_action_value = reward + self.discount_factor * next_state_value

        loss = nn.MSELoss()(state_action_value, expected_state_action_value)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
