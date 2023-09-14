import os
import pickle
import random

import numpy as np
import torch

from .model import QLearning

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Determine the input size based on your state representation
    input_size = 576  # Replace with the actual input size

    self.q_learning = QLearning(num_actions=4, input_size=input_size, hidden_size=64, learning_rate=0.1,
                                discount_factor=0.9)

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
    else:
        self.logger.info("Loading model from saved state.")
        self.q_learning.load_state_dict(torch.load(f'my-saved-model.pt'))
        self.q_learning.eval()


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Check if there are revealed coins in the current game state

    # If there are no revealed coins, use Q-learning to choose the action
    state = state_to_features(game_state)
    action = self.q_learning.get_action(state)

    # # Random exploration during training
    # random_prob = 0.1
    # if self.train and random.random() < random_prob:
    #     self.logger.debug("Choosing action purely at random.")
    #     # 80%: walk in any direction. 10% wait. 10% bomb.
    #     return np.random.choice(ACTIONS, p=[0.25, 0.25, 0.25, 0.25])

    return action


import numpy as np

def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e., a feature vector.

    :param game_state: A dictionary describing the current game board.
    :return: np.array
    """
    global field
    if game_state is None:
        return None

    # Initialize feature channels as empty lists
    channels = []

    if 'field' in game_state:
        # Create a feature channel for the game field
        field = game_state['field']

        # Preprocess the field data by normalizing it to values between 0 and 1
        normalized_field = (field + 1) / 2.0  # Normalize from [-1, 1] to [0, 1]
        channels.append(normalized_field)

    if 'self' in game_state:
        # Create a feature channel for the agent's position
        agent_position = game_state['self'][3]

        # Create an empty grid for the agent's position
        agent_channel = np.zeros_like(field)

        # Set the agent's position to a specific value (e.g., 1) to highlight it
        agent_channel[agent_position] = 1
        channels.append(agent_channel)

    # Concatenate feature channels into a feature tensor
    stacked_channels = np.stack(channels)

    # Flatten the feature tensor into a 1D feature vector
    feature_vector = stacked_channels.reshape(-1)

    return feature_vector
