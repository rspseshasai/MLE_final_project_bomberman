import torch
import torch.nn as nn

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        self.input_dim = 23
        self.output_dim = len(ACTIONS)

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.output_dim)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class QLearningAgent:
    def __init__(self):
        self.q_network = QNetwork()

    def update_q_table(self, state, action, reward, next_state):
        # Convert state and next_state to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32) if next_state is not None else None

        # Get Q-values for the current state and next state
        q_values = self.q_network(state)
        next_q_values = self.q_network(next_state) if next_state is not None else None

        # Calculate the target Q-value using the Q-learning update rule
        if next_state is not None:
            max_next_q_value, _ = next_q_values.max(0)
            target_q_value = reward + self.DISCOUNT_FACTOR * max_next_q_value
        else:
            target_q_value = reward

        # Define a dictionary to map action strings to indices
        action_to_index = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'BOMB': 4, 'WAIT': 5}

        if action is not None:
            # Convert the action string to its index
            action_index = action_to_index[action]

            # Calculate the loss (Mean Squared Error)
            loss = torch.mean((q_values[0, action_index] - target_q_value) ** 2)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_model(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load_model(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
        self.q_network.eval()
