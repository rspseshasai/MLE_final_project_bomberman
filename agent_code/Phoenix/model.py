import torch
import torch.nn as nn


class Phoenix(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Phoenix, self).__init__()
        self.input_features = 1734
        self.hidden_features = 246
        self.output_features = 6
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Define the layers of the neural network
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.float()  # Cast the input tensor to float if it's not already
        x = torch.relu(self.fc1(x))
        x = self.output_layer(x)
        return x

    def initialize_training(self, learning_rate, batch_size):
        # Set up training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()
