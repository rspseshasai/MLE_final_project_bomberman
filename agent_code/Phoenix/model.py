import torch.nn as nn
import torch.nn.functional as F

class Phoenix(nn.Module):
    def __init__(self):
        super(Phoenix, self).__init__()
        self.dense = nn.Linear(in_features=23, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=6)

    def forward(self, x):
        x = F.relu(self.dense(x))
        q_values = self.out(x)
        return q_values

    def initialize_training(self,
                            alpha,
                            gamma,
                            epsilon,
                            buffer_size,
                            batch_size,
                            loss_function,
                            optimizer,
                            training_episodes):
        self.gamma = gamma
        self.epsilon_begin = epsilon[0]
        self.epsilon_end = epsilon[1]
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.optimizer = optimizer(self.parameters(), lr=alpha)
        self.loss_function = loss_function
        self.training_episodes = training_episodes
