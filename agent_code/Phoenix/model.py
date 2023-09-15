import torch
import torch.nn as nn


class Phoenix(nn.Module):
    def __init__(self):
        super(Phoenix, self).__init__()

        self.number_of_in_features = 4
        self.number_of_actions = 6

        # LAYERS
        # self.dense1 = nn.Linear(in_features=self.number_of_in_features, out_features=24)
        # self.dense2 = nn.Linear(in_features=24, out_features=36)

        # self.dense3 = nn.Linear(in_features=36, out_features=27)
        self.out = nn.Linear(in_features=self.number_of_in_features, out_features=self.number_of_actions)

    def forward(self, x):
        out = self.out(x)

        return out

    def initialize_training(self, learning_rate, batch_size):
        # Set up training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()
