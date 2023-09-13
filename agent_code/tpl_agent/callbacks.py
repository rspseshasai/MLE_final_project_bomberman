import os
import pickle
import random

import numpy as np

from model import QLearning

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    Initialize your Q-learning model here and assign it to self.q_learning.

    :param self: This object is passed to all callbacks, and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up Q-learning model from scratch.")
        self.q_learning = QLearning()  # Initialize your Q-learning model
    else:
        self.logger.info("Loading Q-learning model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_learning = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Check if there are revealed coins in the current game state
    if 'coins' in game_state and game_state['coins']:
        # If there are revealed coins, calculate the distance to each coin
        coin_positions = np.array(game_state['coins'])
        agent_position = np.array(game_state['self'][3])
        distances_to_coins = np.linalg.norm(coin_positions - agent_position, axis=1)

        # Choose the action that moves the agent closer to the nearest coin using Q-learning
        nearest_coin_index = np.argmin(distances_to_coins)
        nearest_coin_position = coin_positions[nearest_coin_index]

        # Determine the direction to move towards the nearest coin
        x_diff = nearest_coin_position[0] - agent_position[0]
        y_diff = nearest_coin_position[1] - agent_position[1]

        if abs(x_diff) > abs(y_diff):
            # Move horizontally
            if x_diff > 0:
                action = 'RIGHT'
            else:
                action = 'LEFT'
        else:
            # Move vertically
            if y_diff > 0:
                action = 'DOWN'
            else:
                action = 'UP'
    else:
        # If there are no revealed coins, use Q-learning to choose the action
        state = state_to_features(game_state)
        action = self.q_learning.get_action(state)

        # Random exploration during training
        random_prob = 0.1
        if self.train and random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

    return action


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

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
