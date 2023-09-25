import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque

# Defined a named tuple for storing experience transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.input_dim = 8
        self.output_dim = len(ACTIONS)
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.output_dim)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self):
        self.q_network = QNetwork()
        DISCOUNT_FACTOR=None
        self.target_network = QNetwork().eval()  # Target network for stable DQN
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = deque(maxlen=3)  # Experience replay buffer
        self.batch_size = 10
        self.discount_factor = DISCOUNT_FACTOR
        self.EXPLORATION_PROB = 0.9
        self.target_update_interval = 0.4
        self.steps_done = 0

    def update_q_network(self, state, action, reward, next_state):
        # Store transition in the replay buffer
        self.memory.append(Transition(state, action, next_state, reward))

        # Perform a DQN update if there are enough samples in the replay buffer
        if len(self.memory) >= self.batch_size:
            self._optimize_model()

        # Update the target network periodically for stability
        if self.steps_done % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _optimize_model(self):
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.q_network(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load_model(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
        self.q_network.eval()
