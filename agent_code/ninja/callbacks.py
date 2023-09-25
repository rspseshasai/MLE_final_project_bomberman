import os
import random
from itertools import product
from queue import Queue

import numpy as np
import torch
from torch import optim

from agent_code.ninja.model import RNNModel
from agent_code.ninja.rule_based_for_random import wait_if_bomb_or_explosion, random_clever_move

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# Define the maximum number of features per channel
MAX_FEATURES_PER_CHANNEL = 10


def setup(self):
    self.logger.info("Setting up agent...")

    # Initialize your RNN model here
    self.input_dim = 50  # Define your input dimension
    self.output_dim = len(ACTIONS)
    self.hidden_dim = 128  # Adjust as needed

    # Check if the agent is in training mode
    if self.train:
        self.logger.info("Agent is in training mode. Initializing a new model...")
        self.rnn_model = RNNModel(self.input_dim, self.hidden_dim, self.output_dim)
    else:
        # Load a saved model when not in training mode
        self.logger.info("Agent is not in training mode. Loading a saved model...")
        model_filename = "my-saved-model.pt"
        if os.path.isfile(model_filename):
            self.rnn_model = RNNModel(self.input_dim, self.hidden_dim, self.output_dim)
            self.rnn_model.load_state_dict(torch.load(model_filename))
            self.rnn_model.eval()
        else:
            self.logger.warning(f"No saved model found at {model_filename}. Initializing a new model...")
            self.rnn_model = RNNModel(self.input_dim, self.hidden_dim, self.output_dim)
            self.optimizer = optim.Adam(self.rnn_model.parameters(), lr=0.001)


def act(self, game_state: dict) -> str:
    if self.train:
        # Calculate epsilon based on decay
        epsilon = max(self.EPSILON_END, self.EPSILON_START * self.EPSILON_DECAY ** self.EPISODES)

        if random.random() < epsilon:
            action = random_clever_move(self, game_state)
            self.logger.info(f"Select action {action} after the rule-based agent.")

            action_wait = wait_if_bomb_or_explosion(self, game_state, action)
            if action_wait is not None:
                return action_wait
            return action

    # Exploitation: Query the model for the best action
    self.logger.debug("Querying model for action.")
    state_features = state_to_features(game_state)
    state_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    q_values = self.rnn_model(state_tensor)
    action_index = q_values.argmax(1).item()
    return ACTIONS[action_index]


def bfs_shortest_distance(agent_pos, targets, game_state):
    if not targets:
        return [np.inf] * len(game_state['self'][0])

    width, height = game_state['field'].shape
    visited = np.zeros((width, height), dtype=bool)
    queue = Queue()

    for target in targets:
        queue.put((target, 0))

    while not queue.empty():
        (x, y), distance = queue.get()
        visited[x, y] = True

        if (x, y) == agent_pos:
            return [distance] * len(game_state['self'][0])

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_x, new_y = x + dx, y + dy

            if 0 <= new_x < width and 0 <= new_y < height and not visited[new_x, new_y]:
                queue.put(((new_x, new_y), distance + 1))

    return [np.inf] * len(game_state['self'][0])


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e., a feature vector.

    :param game_state: A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        return np.zeros(MAX_FEATURES_PER_CHANNEL * NUM_CHANNELS)  # Return zeros for None state

    # Get the game board and agent's position
    board = game_state['field']
    agent_pos = game_state['self'][3]

    # Define possible actions
    actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

    # Initialize feature channels
    channels = []

    # Initialize feature counters
    feature_count = 0

    # Feature 1: Distance to the nearest coin for each action (max 10 features)
    coins = game_state['coins']
    coin_distances = bfs_shortest_distance(agent_pos, coins, game_state)
    coin_distance_channel = np.array(coin_distances * len(actions))
    channels.append(coin_distance_channel[:MAX_FEATURES_PER_CHANNEL])
    feature_count += min(MAX_FEATURES_PER_CHANNEL, len(coin_distance_channel))

    # Feature 2: Distance to the nearest crate for each action (max 10 features)
    crates = [(x, y) for x, y in product(range(board.shape[0]), range(board.shape[1])) if board[x, y] == 1]
    crate_distances = bfs_shortest_distance(agent_pos, crates, game_state)
    crate_distance_channel = np.array(crate_distances * len(actions))
    channels.append(crate_distance_channel[:MAX_FEATURES_PER_CHANNEL])
    feature_count += min(MAX_FEATURES_PER_CHANNEL, len(crate_distance_channel))

    # Feature 3: Can agent place a bomb for each action (max 10 features)
    can_place_bomb_channel = np.array([int(game_state['self'][2])] * len(actions))
    channels.append(can_place_bomb_channel[:MAX_FEATURES_PER_CHANNEL])
    feature_count += min(MAX_FEATURES_PER_CHANNEL, len(can_place_bomb_channel))

    # Feature 4: Distance to the nearest opponent for each action (max 10 features)
    opponents = game_state['others']
    opponent_distances = bfs_shortest_distance(agent_pos, [opp[3] for opp in opponents], game_state)
    opponent_distance_channel = np.array(opponent_distances * len(actions))
    channels.append(opponent_distance_channel[:MAX_FEATURES_PER_CHANNEL])
    feature_count += min(MAX_FEATURES_PER_CHANNEL, len(opponent_distance_channel))

    # Feature 5: Is the agent in the explosion range for each action (max 10 features)
    explosion_map = game_state['explosion_map']
    in_explosion_range_channel = np.array([int(explosion_map[agent_pos[0], agent_pos[1]] > 0)] * len(actions))
    channels.append(in_explosion_range_channel[:MAX_FEATURES_PER_CHANNEL])
    feature_count += min(MAX_FEATURES_PER_CHANNEL, len(in_explosion_range_channel))

    # Create the feature vector by stacking all channels
    feature_vector = np.concatenate(channels)

    # Pad the feature vector to reach the desired feature count
    if feature_count < MAX_FEATURES_PER_CHANNEL * NUM_CHANNELS:
        padding = np.zeros(MAX_FEATURES_PER_CHANNEL * NUM_CHANNELS - feature_count)
        feature_vector = np.concatenate([feature_vector, padding])

    return feature_vector


# Define the total number of feature channels
NUM_CHANNELS = 5  # You can adjust this based on the number of feature types used
