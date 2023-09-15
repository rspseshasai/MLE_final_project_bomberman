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
        self.model = Phoenix()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:

    if self.train:  # Exploration vs exploitation
        if random.random() <= 0.3:  # Choose a random action for exploration
            return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])  # Exploration probabilities

    features = state_to_features(game_state)
    Q = self.network(features)
    action_prob = np.array(torch.softmax(Q, dim=1).detach().squeeze())
    best_action = ACTIONS[np.argmax(action_prob)]

    self.logger.debug("Action selected: " + best_action)
    return best_action



def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # Implement your state representation here
    # Example: channels = []
    #          channels.append(...)
    #          stacked_channels = np.stack(channels)
    #          return stacked_channels.reshape(-1)
    # You may need to create multiple channels to represent different aspects of the game state.

    return None  # Replace None with your state representation
