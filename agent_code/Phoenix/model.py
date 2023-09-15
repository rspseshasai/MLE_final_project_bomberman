import torch
import torch.nn as nn


class Phoenix(nn.Module):
    def __init__(self):
        super(Phoenix, self).__init__()
        self.input_features = 10
        self.output_features = 6
        # Define the layers of the neural network
        self.output_layer = nn.Linear(in_features=self.input_features, out_features=self.output_features)

    def forward(self, x):
        # Define the forward pass of the network
        x = self.output_layer(x)
        return x

    def initialize_training(self, learning_rate, batch_size):
        # Set up training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()
