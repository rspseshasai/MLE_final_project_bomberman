import torch
import torch.nn as nn
import torch.optim as optim
import random

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, state, action, reward, next_state):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Add a time step dimension (seq_len=1) to the input tensor
        x = x.unsqueeze(1)

        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out


class RNNLearningAgent:
    def __init__(self, input_dim, capacity=10000, batch_size=64):
        self.input_dim = input_dim
        self.output_dim = len(ACTIONS)
        self.hidden_dim = 128  # You can adjust this as needed
        self.batch_size = batch_size
        self.capacity = capacity
        self.buffer = ExperienceReplayBuffer(self.capacity)

        # Create the RNN model
        self.rnn_model = RNNModel(self.input_dim, self.hidden_dim, self.output_dim)

        # Define an optimizer
        self.optimizer = None

    def train_model(self):
        if len(self.buffer.buffer) >= self.batch_size:
            # Sample a batch of experiences from the replay buffer
            batch = self.buffer.sample(self.batch_size)

            # Convert batch data to tensors
            states = torch.tensor([exp[0] for exp in batch], dtype=torch.float32)
            actions = torch.tensor([exp[1] for exp in batch])
            rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32)
            next_states = torch.tensor([exp[3] for exp in batch], dtype=torch.float32)

            # Get Q-values for the current state and next state
            current_state_values = self.rnn_model(states)
            next_state_values = self.rnn_model(next_states)

            # Calculate the target Q-values using the Q-learning update rule
            max_next_state_values, _ = next_state_values.max(1)
            target_state_values = rewards + self.DISCOUNT_FACTOR * max_next_state_values

            # Calculate the loss (Mean Squared Error)
            action_indices = torch.arange(self.batch_size, dtype=torch.int64) * self.output_dim + actions
            predicted_state_values = current_state_values.view(-1)[action_indices]
            loss = torch.mean((predicted_state_values - target_state_values) ** 2)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_rnn_model(self, filename):
        torch.save(self.rnn_model.state_dict(), filename)

