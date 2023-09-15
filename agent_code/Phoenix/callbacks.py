import os
import pickle
import random

import numpy as np
import torch

from agent_code.Phoenix.model import Phoenix

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        # Initialize your model here if training or create a default model
        self.model = Phoenix(1734, 256, len(ACTIONS))
    else:
        self.logger.info("Loading model from saved state.")
        with open("saved/my-phoenix-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    if self.train:  # Exploration vs exploitation
        if random.random() <= 0.3:  # Choose a random action for exploration
            return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])  # Exploration probabilities

    features = state_to_features(game_state)
    q_values = self.model(torch.tensor(features, dtype=torch.float32))
    action_prob = np.array(torch.argmax(q_values).item())
    best_action = ACTIONS[np.argmax(action_prob)]

    self.logger.debug("Action selected: " + best_action)
    return best_action


def state_to_features(game_state: dict) -> np.array:

    # Extract relevant information from game_state
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    self_info = game_state['self']
    self_x, self_y = self_info[3]  # Get the coordinates of your agent

    # Define the size of the game board
    width, height = field.shape[0], field.shape[1]

    # Create an empty feature vector (you should adjust its size)
    feature_vector = []

    # Iterate over the game board and add relevant information to the feature vector
    for x in range(width):
        for y in range(height):
            tile = field[x][y]
            if tile == 1:
                # Add a feature indicating a crate
                feature_vector.append(1)
            elif tile == -1:
                # Add a feature indicating a stone wall
                feature_vector.append(-1)
            else:
                # Add a feature indicating a free tile
                feature_vector.append(0)

            # Add features based on the bomb countdown and explosion map
            for bomb_coords, bomb_countdown in bombs:
                if bomb_coords == (x, y):
                    # Add a feature indicating a bomb with countdown
                    feature_vector.append(bomb_countdown)
                    break
            else:
                # Add a feature indicating no bomb at this location
                feature_vector.append(0)

            # Add features based on the explosion map
            feature_vector.append(explosion_map[x][y])

            # Add a feature indicating the presence of a coin
            if (x, y) in coins:
                feature_vector.append(1)
            else:
                feature_vector.append(0)

            # Add features indicating the relative position of your agent
            feature_vector.append(x - self_x)
            feature_vector.append(y - self_y)

    # Convert the feature vector to a numpy array
    return np.array(feature_vector)
