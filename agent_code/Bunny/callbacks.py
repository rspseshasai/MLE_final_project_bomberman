import os
import random

import numpy as np

from .features import state_to_features
from .model import QLearningAgent  # Import the QLearningAgent class with PyTorch

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
        # Initialize the QLearningAgent instance with PyTorch in training mode
        self.q_learning_agent = QLearningAgent()
    else:
        self.logger.info("Loading model from saved state.")
        # Create a QLearningAgent instance with PyTorch
        self.q_learning_agent = QLearningAgent()
        # Load the trained model
        self.q_learning_agent.load_model("saved_parameters/my-saved-model.pt")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if self.train:
        # Training mode: Perform exploration and exploitation
        random_prob = .1
        if random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

        self.logger.debug("Querying model for action.")
        # Use the Q-learning agent to choose the action
    else:
        # Testing mode: Already loaded the model during setup
        self.logger.debug("Using model for action.")

    return get_best_move(self, game_state)


def get_best_move(self, game_state: dict) -> str:
    if self.train and random.random() < self.q_learning_agent.EXPLORATION_PROB:
        return np.random.choice(ACTIONS)

    state = state_to_features(game_state)
    q_values = self.q_learning_agent.q_network(state)
    q_values_array = q_values.detach().numpy()
    best_action_index = np.argmax(q_values_array)
    best_action = ACTIONS[best_action_index]
    return best_action
