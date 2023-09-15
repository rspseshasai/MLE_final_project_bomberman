import os
import pickle
import random
import numpy as np
import torch

from agent_code.Phoenix_old.model import QNetwork

# Define your list of actions here
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are independent of the game state.

    :param self: This object is passed to all callbacks, and you can set arbitrary values.
    """

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")

        # Initialize your Q-network here
        input_size = 1734  # Define the input size of your Q-network
        hidden_size = 256  # Define the hidden layer size of your Q-network
        output_size = len(ACTIONS)  # The output size should match the number of actions

        self.model = QNetwork(input_size, hidden_size, output_size)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


from queue import Queue


def calculate_direction_to_nearest_coin_bfs(game_state):
    # Get relevant information from the game state
    field = game_state["field"]
    coins = game_state["coins"]
    self_x, self_y = game_state["self"][3]  # Agent's current position

    # Define possible movements (UP, RIGHT, DOWN, LEFT)
    movements = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    directions = ["UP", "RIGHT", "DOWN", "LEFT"]

    # Create a visited matrix to keep track of visited positions
    visited = [[False for _ in range(len(field[0]))] for _ in range(len(field))]
    visited[self_x][self_y] = True

    # Initialize a queue for BFS
    queue = Queue()
    queue.put((self_x, self_y, []))  # Queue stores position and path to that position

    while not queue.empty():
        x, y, path = queue.get()

        # Check if the current position contains a coin
        if (x, y) in coins:
            # If a coin is found, return the first direction from the path
            if len(path) > 0:
                return path[0]

        # Explore neighboring positions
        for dx, dy in movements:
            new_x, new_y = x + dx, y + dy

            # Check if the new position is within the game board
            if 0 <= new_x < len(field) and 0 <= new_y < len(field[0]):

                # Check if the new position is a free tile and has not been visited
                if field[new_x][new_y] == 0 and not visited[new_x][new_y]:
                    visited[new_x][new_y] = True

                    # Add the new position and path to the queue
                    new_path = path + [directions[movements.index((dx, dy))]]
                    queue.put((new_x, new_y, new_path))

    # If no coin is found, return None
    return None


def act(self, game_state: dict) -> str:
    self.logger.debug("Querying model for action.")

    if self.train:
        epsilon = 0.4  # Initial exploration rate, adjust as needed
        if random.random() < epsilon:
            self.logger.debug("Choosing action with exploration.")
            # Combine exploration with moving towards the nearest coin
            direction_to_coin = calculate_direction_to_nearest_coin_bfs(game_state)
            # direction_to_coin = astar_search(game_state, manhattan_distance_heuristic)
            if direction_to_coin:
                return direction_to_coin
            else:
                return np.random.choice(ACTIONS)  # If no coin found, explore randomly
        else:
            # Choose the action using the Q-network
            self.logger.debug("Choosing action using the Q-network.")
            state_features = state_to_features(game_state)
            state_tensor = torch.tensor(state_features, dtype=torch.float32)
            q_values = self.model(state_tensor)
            action_probabilities = torch.softmax(q_values, dim=0)
            action_index = torch.multinomial(action_probabilities, 1).item()
            chosen_action = ACTIONS[action_index]
            return chosen_action
    else:
        # In testing mode, choose the action with the highest Q-value
        state_features = state_to_features(game_state)
        state_tensor = torch.tensor(state_features, dtype=torch.float32)
        q_values = self.model(state_tensor)
        best_action_index = torch.argmax(q_values).item()
        self.logger.info("[TEST] Chosen Action: " + str(best_action_index))
        return ACTIONS[best_action_index]


import numpy as np


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e., a feature vector.

    :param game_state: A dictionary describing the current game board.
    :return: np.array
    """
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
