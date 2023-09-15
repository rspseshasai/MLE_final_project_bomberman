import os
import pickle
import random
from collections import deque

import numpy as np
import torch

from agent_code.Phoenix.model import Phoenix

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
STEP = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])


def setup(self):
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        # Initialize your model here if training or create a default model
        self.model = Phoenix()
    else:
        self.logger.info("Loading model from saved state.")
        with open("saved/my-phoenix-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    if self.train:  # Exploration vs exploitation
        eps = self.epsilon_arr[self.episode_counter]
        if random.random() <= eps:  # choose random action
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])  # EXPLORATION

    features = state_to_features(self, game_state)
    Q = self.model(features)
    action_prob = np.array(torch.softmax(Q, dim=1).detach().squeeze())
    best_action = ACTIONS[np.argmax(action_prob)]
    self.logger.debug("action returned by callbacks#act: " + best_action)
    return best_action


def state_to_features(self, game_state: dict) -> np.array:
    max_len_wanted_fields = 9  # only the coins

    # at the beginning and the end:

    if game_state is None:
        return None

    def possible_neighbors(pos):
        result = []
        for new_pos in (pos + STEP):
            if game_state["field"][new_pos[0], new_pos[1]] == 0:
                result.append(new_pos.tolist())

        return result

    player_pos = np.array(game_state["self"][3])
    wanted_fields = np.array(game_state["coins"])
    number_of_coins = len(wanted_fields)
    if len(wanted_fields) == 0:
        return torch.tensor([1, 1, 1, 1]).float().unsqueeze(0)
    # if the len of wanted fields changes, we receive an error
    # => fill it with not reachable entries (e.g. [16,16]) and shuffle afterward to prevent a bias.
    fake_entries = []
    for _ in range(max_len_wanted_fields - len(wanted_fields)):
        fake_entries.append([16, 16])
    if len(fake_entries) != 0:
        wanted_fields = np.append(wanted_fields, fake_entries, axis=0)
        np.random.shuffle(wanted_fields)  # prevent a bias by having the fake entries always on the end.
        # all of the coin fields should have the same influence since the order in game_state is arbitrary

    possible_next_pos = possible_neighbors(player_pos)
    features = []

    way_to_nearest = []

    for pos in (player_pos + STEP):
        new_distances = np.empty(len(wanted_fields))
        pos = pos.tolist()

        if pos not in possible_next_pos:
            features = np.append(features, -1)
            way_to_nearest.append(np.inf)
            continue

        new_distances.fill(np.inf)  # if no way can be found we consider the distance to be infinite

        # analyse the change of the distances of the shortest paths to all coins if we do a STEP
        visited = [player_pos.tolist()]
        q = deque()
        q.append([pos, 1])

        while len(q) != 0:

            pos, distance = q.popleft()
            if pos in visited:
                continue
            visited.append(pos)
            index = np.argwhere((wanted_fields == pos).all(axis=1))  # check if pos is in wanted_fields
            new_distances[index] = distance
            assert sum((wanted_fields == pos).all(axis=1)) <= 1
            neighbors = possible_neighbors(pos)
            for node in neighbors:
                q.append([node, distance + 1])

        features = np.append(features, np.sum(1 / new_distances))

    # encode the movement to the coins in one hot manner
    hot_one = np.argmax(features)
    features[features >= 0] = 0
    features[hot_one] = number_of_coins

    # print("\n\n")
    # print(f"oben:{features[3]}")
    # print(f"unten:{features[2]}")
    # print(f"links:{features[1]}")
    # print(f"rechts:{features[0]}")
    # a = ["rechts", "links", "unten", "oben"]
    # print(f"--> {a[np.argmax(features)]}")

    features = torch.from_numpy(features).float()

    return features.unsqueeze(0)

# def state_to_features(game_state: dict) -> np.array:
#
#     # Extract relevant information from game_state
#     field = game_state['field']
#     bombs = game_state['bombs']
#     explosion_map = game_state['explosion_map']
#     coins = game_state['coins']
#     self_info = game_state['self']
#     self_x, self_y = self_info[3]  # Get the coordinates of your agent
#
#     # Define the size of the game board
#     width, height = field.shape[0], field.shape[1]
#
#     # Create an empty feature vector (you should adjust its size)
#     feature_vector = []
#
#     # Iterate over the game board and add relevant information to the feature vector
#     for x in range(width):
#         for y in range(height):
#             tile = field[x][y]
#             if tile == 1:
#                 # Add a feature indicating a crate
#                 feature_vector.append(1)
#             elif tile == -1:
#                 # Add a feature indicating a stone wall
#                 feature_vector.append(-1)
#             else:
#                 # Add a feature indicating a free tile
#                 feature_vector.append(0)
#
#             # Add features based on the bomb countdown and explosion map
#             for bomb_coords, bomb_countdown in bombs:
#                 if bomb_coords == (x, y):
#                     # Add a feature indicating a bomb with countdown
#                     feature_vector.append(bomb_countdown)
#                     break
#             else:
#                 # Add a feature indicating no bomb at this location
#                 feature_vector.append(0)
#
#             # Add features based on the explosion map
#             feature_vector.append(explosion_map[x][y])
#
#             # Add a feature indicating the presence of a coin
#             if (x, y) in coins:
#                 feature_vector.append(1)
#             else:
#                 feature_vector.append(0)
#
#             # Add features indicating the relative position of your agent
#             feature_vector.append(x - self_x)
#             feature_vector.append(y - self_y)
#
#     # Convert the feature vector to a numpy array
#     return np.array(feature_vector)
