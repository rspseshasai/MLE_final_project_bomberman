import os
import random
import numpy as np
from .model import DQNAgent  # Import the DQNAgent class with PyTorch
from .features import state_to_features
from queue import PriorityQueue

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):

    if self.train or not os.path.isfile("logs/saved-model.pt"):
        self.logger.info("Setting up model from start.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
        # Initialize the DQNAgent instance with PyTorch in training mode
        self.dqn_agent = DQNAgent()
    else:
        self.logger.info("Loading model from saved state.")
        # Create a DQNAgent instance with PyTorch
        self.dqn_agent = DQNAgent()
        # Load the trained model
        self.dqn_agent.load_model("required_parameters/saved-model.pt")

def wait_if_bomb_or_explosion(self, game_state, action):
    # Check if the selected action leads to an explosion
    if will_run_into_explosion(self, game_state, action):
        self.logger.info("Selected action leads to an explosion. Waiting instead.")
        return "WAIT"

    position = game_state["self"][3]
    # Calculate the next position based on the selected action
    next_position = get_next_position(position, action)
    bomb_positions = get_bomb_positions(game_state)
    if not is_position_in_bomb_range(position, bomb_positions):
        if is_position_in_bomb_range(next_position, bomb_positions):
            return "WAIT"

"""
Used Greedy epsilon due to better result; abandoned softmax action selection.
def softmax_action_selection(q_values, temperature):
    exp_values = np.exp(q_values / temperature)
    action_probs = exp_values / exp_values.sum()
    return action_probs

def act(self, game_state: dict) -> str:
    self.temperature = torch.tensor(1.0, dtype=torch.float32)
    self.features = state_to_features(game_state)
    if self.train:
        # Training mode: Perform exploration and exploitation
        random_prob = self.epsilon_arr[self.episode_counter]
        if random.random() <= random_prob:
            if random_prob > 0.1:
                # Use Softmax Action Selection for exploration
                q_values = self.dqn_agent.q_network(self.features)
                exp_values = np.exp((q_values / self.temperature).detach().numpy())
                action_probs = exp_values / exp_values.sum()
                action_probs = action_probs.flatten()
                # Normalize the probabilities to sum to 1
                action_probs /= np.sum(action_probs)
                action = np.random.choice(ACTIONS, p=action_probs)
                self.logger.info(f"Choose action {action} using Softmax Action Selection")
            else:
                action = random_clever_move(self, game_state)
                self.logger.info(f"Select action {action} after the rule-based agent.")

            action_wait = wait_if_bomb_or_explosion(self, game_state, action)
            if action_wait is not None:
                return action_wait
            return action

        self.logger.debug("Querying model for action.")
    else:
        # Testing mode: Already loaded the model during setup
        self.logger.debug("Using model for action.")

    return get_best_move(self, game_state)
"""

def act(self, game_state: dict) -> str:

    self.features = state_to_features(game_state)
    if self.train:
        # Training mode: Perform exploration and exploitation
        random_prob = self.epsilon_arr[self.episode_counter]
        if random.random() <= random_prob:
            if random_prob > 0.1:
                if np.random.randint(10) == 0:  # old: 10 / 100 now: 3/4
                    action = np.random.choice(ACTIONS, p=[.167, .167, .167, .167, .166, .166])
                    self.logger.info(f"Choose action {action} completely at random")

                else:
                    action = random_clever_move(self, game_state)
                    self.logger.info(f"Select action {action} after the rule-based agent.")

            action_wait = wait_if_bomb_or_explosion(self, game_state, action)
            if action_wait is not None:
                return action_wait
            return action

        self.logger.debug("Querying model for action.")
        # Use the DQN agent to choose the action
    else:
        # Testing mode: Already loaded the model during setup
        self.logger.debug("Using model for action.")

    return get_best_move(self, game_state)



def will_run_into_explosion(self, game_state: dict, action: str) -> bool:

    # Get the agent's current position
    position = game_state["self"][3]

    # Calculate the next position based on the selected action
    next_position = get_next_position(position, action)

    # Check if the next position has a positive value in the explosion map
    explosion_map = game_state["explosion_map"]
    if explosion_map[next_position[1]][next_position[0]] > 0:
        return True

    return False


def get_best_move(self, game_state: dict) -> str:
    if self.train and random.random() < self.dqn_agent.EXPLORATION_PROB:
        return np.random.choice(ACTIONS)

    state = state_to_features(game_state)
    q_values = self.dqn_agent.q_network(state)
    q_values_array = q_values.detach().numpy()
    best_action_index = np.argmax(q_values_array)
    best_action = ACTIONS[best_action_index]
    self.logger.info(f"Predicted action {best_action} by our QNet model.")

    return best_action


def random_clever_move(self, game_state: dict) -> str:
    should_collect_coins = collect_coins(self, game_state)

    # Check if there is a bomb nearby based on the explosion map
    if should_run_away_from_explosion(self, game_state):
        return run_away_from_explosion(self, game_state)

    # Check if the agent should avoid bombs
    position = game_state["self"][3]
    bomb_positions = get_bomb_positions(game_state)
    if is_position_in_bomb_range(position, bomb_positions):
        return avoid_bombs(self, game_state)

    # Check if there is an opponent nearby
    if is_opponent_nearby(self, game_state):
        return "BOMB"

    # Check if there are coins to collect
    elif should_collect_coins != 'WAIT':
        return should_collect_coins  # Collect coins if the path is clear
    # Check if there is a crate nearby
    elif is_crate_nearby(self, game_state):
        return "BOMB"

    else:
        return move_to_nearest_crate(self, game_state)  # Move towards the nearest crate


def should_run_away_from_explosion(self, game_state: dict) -> bool:

    position = game_state["self"][3]
    explosion_map = game_state["explosion_map"]

    # Check if the agent's current position is within the explosion range
    if explosion_map[position[0]][position[1]] > 0:
        return True

    return False


def run_away_from_explosion(self, game_state: dict) -> str:

    position = game_state["self"][3]
    explosion_map = game_state["explosion_map"]

    # Determine safe positions to move to
    safe_positions = []
    for action in ACTIONS:
        next_position = get_next_position(position, action)
        if (
                next_position[0] > 0
                and next_position[0] < 17
                and next_position[1] > 0
                and next_position[1] < 17
                and explosion_map[next_position[0]][next_position[1]] == 0
        ):
            safe_positions.append(next_position)

    if not safe_positions:
        return "WAIT"  # No safe positions to move to

    # Choose a random safe position to move to
    return random.choice(safe_positions)

def move_to_nearest_crate(self, game_state: dict) -> str:
    # Get the agent's current position
    current_position = game_state["self"][3]

    # Get a list of all reachable crates and their positions
    reachable_crates = [(x, y) for x in range(17) for y in range(17) if game_state["field"][x, y] == 1]

    if not reachable_crates:
        return "WAIT"  # No crates to reach

    # Define a heuristic function to estimate the cost to reach a crate
    def heuristic(position, crate_position):
        return abs(position[0] - crate_position[0]) + abs(position[1] - crate_position[1])

    # Initialize the A* algorithm with a priority queue
    open_set = PriorityQueue()
    open_set.put((0, current_position))
    came_from = {}

    # Initialize costs
    g_score = {position: float("inf") for position in reachable_crates}
    g_score[current_position] = 0

    while not open_set.empty():
        _, current = open_set.get()

        if current in reachable_crates:
            # Reached a crate, so return the path to it
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()

            # Calculate the action to take to reach the crate
            next_position = path[1]
            if next_position[0] < current_position[0]:
                return "LEFT"
            elif next_position[0] > current_position[0]:
                return "RIGHT"
            elif next_position[1] < current_position[1]:
                return "UP"
            elif next_position[1] > current_position[1]:
                return "DOWN"

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < 17 and 0 <= neighbor[1] < 17 and game_state["field"][neighbor] != -1:
                tentative_g_score = g_score[current] + 1

                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, reachable_crates[0])
                    open_set.put((f_score, neighbor))

    return "WAIT"  # Default to waiting if no path is found


def is_crate_blocking_path(self, game_state: dict) -> bool:

    coins = game_state["coins"]
    position = game_state["self"][3]

    # Check if there are coins to collect
    if not coins:
        return False

    # Find the nearest coin
    nearest_coin = min(coins, key=lambda c: abs(position[0] - c[0]) + abs(position[1] - c[1]))

    # Check if there is a crate blocking the path to the nearest coin
    if nearest_coin:
        dx, dy = nearest_coin[0] - position[0], nearest_coin[1] - position[1]
        next_position = get_next_position(position, "")
        if dx < 0:
            next_position[0] -= 1
        elif dx > 0:
            next_position[0] += 1
        elif dy < 0:
            next_position[1] -= 1
        elif dy > 0:
            next_position[1] += 1

        # Check if the next position contains a crate
        if game_state["field"][next_position] == 1:
            return True

    return False


def is_crate_nearby(self, game_state: dict) -> bool:

    position = game_state["self"][3]

    # Get a list of all reachable crates and their positions
    crates = [(x, y) for x in range(17) for y in range(17) if game_state["field"][x, y] == 1]

    # Calculate distances to crates
    crate_distances = [abs(position[0] - crate[0]) + abs(position[1] - crate[1]) for crate in crates]

    # Check if there are crates within a certain range (e.g., 2 tiles)
    return any(distance <= 1 for distance in crate_distances)


def is_opponent_nearby(self, game_state: dict) -> bool:

    position = game_state["self"][3]
    opponents = [agent[3] for agent in game_state["others"]]

    # Calculate distances to opponents
    opponent_distances = [abs(position[0] - opponent[0]) + abs(position[1] - opponent[1]) for opponent in opponents]

    # Check if there are opponents within a certain range (e.g., 2 tiles)
    return any(distance <= 2 for distance in opponent_distances)


def get_bomb_positions(game_state: dict) -> list:

    bomb_positions = [(x, y) for x, y in game_state["bombs"]]
    return bomb_positions


def avoid_bombs(self, game_state: dict) -> str:

    position = game_state["self"][3]
    bomb_positions = get_bomb_positions(game_state)
    explosion_map = game_state["explosion_map"]

    # Check if the agent is already in a safe position
    if not is_position_in_bomb_range(position, bomb_positions):
        return "WAIT"  # Wait for the active bombs to explode

    # Calculate the maximum safe distance from a bomb blast
    max_safe_distance = 5

    # Check if there are safe positions to move to
    safe_positions = []

    # Iterate over all possible positions on the board
    for x in range(17):
        for y in range(17):
            if (
                    explosion_map[x][y] <= max_safe_distance
                    and not is_position_in_bomb_range((x, y), bomb_positions)
                    and game_state["field"][x, y] == 0
            ):
                # This position is safe, add it to the list of safe positions
                safe_positions.append((x, y))

    if not safe_positions:
        return "WAIT"  # No safe positions to move to

    # Find the nearest safe position and move towards it
    while safe_positions:
        nearest_safe_position = min(safe_positions, key=lambda pos: minkowski_distance(position, pos))
        action = move_towards_target(self, position, nearest_safe_position, game_state)
        if action != "WAIT":
            return action  # Return the action if a path is found
        safe_positions.remove(nearest_safe_position)

    return "WAIT"  # Default to waiting if no path is found


def get_next_position(position, action):

    x, y = position
    if action == "UP":
        return (x, y - 1)
    elif action == "DOWN":
        return (x, y + 1)
    elif action == "LEFT":
        return (x - 1, y)
    elif action == "RIGHT":
        return (x + 1, y)
    else:
        return position


def is_position_in_bomb_range(position, bomb_positions):

    for bomb_position, countdown in bomb_positions:
        x, y = position
        bomb_x, bomb_y = bomb_position

        if x == bomb_x:
            if abs(y - bomb_y) <= 5:
                return True
        elif y == bomb_y:
            if abs(x - bomb_x) <= 5:
                return True

    return False


def collect_coins(self, game_state: dict) -> str:

    # Get the current position of the agent
    current_position = game_state["self"][3]

    # Get a list of all revealed coins
    coins = game_state["coins"]
    revealed_coins = [coin for coin in coins if game_state["field"][coin] == 0]

    if not revealed_coins:
        return "WAIT"  # No revealed coins to collect

    # Find the nearest coin's position
    nearest_coin = min(revealed_coins, key=lambda x: minkowski_distance(current_position, x))

    # Determine the action that moves the agent towards the nearest coin
    return move_towards_target(self, current_position, nearest_coin, game_state)


def minkowski_distance(position1, position2, p=1):
    # Calculate the Minkowski distance between two positions.
    # p: The order of the distance (1 for Manhattan, 2 for Euclidean).

    return (abs(position1[0] - position2[0])**p + abs(position1[1] - position2[1])**p)**(1/p)


def move_towards_target(self, current_position, target_position, game_state):

    def heuristic(position, target):
        return abs(position[0] - target[0]) + abs(position[1] - target[1])

    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    open_set = PriorityQueue()
    open_set.put((0, current_position))
    came_from = {}
    g_score = {position: float("inf") for position in game_state["field"].shape}
    g_score[current_position] = 0

    while not open_set.empty():
        _, current = open_set.get()

        if current == target_position:
            path = reconstruct_path(came_from, current)
            if len(path) > 1:
                next_position = path[1]
                if next_position[0] < current_position[0]:
                    return "LEFT"
                elif next_position[0] > current_position[0]:
                    return "RIGHT"
                elif next_position[1] < current_position[1]:
                    return "UP"
                elif next_position[1] > current_position[1]:
                    return "DOWN"

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if (
                    0 <= neighbor[0] < game_state["field"].shape[0]
                    and 0 <= neighbor[1] < game_state["field"].shape[1]
                    and game_state["field"][neighbor] != 1  # Check for crates
                    and game_state["field"][neighbor] != -1
            ):
                tentative_g_score = g_score[current] + 1

                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, target_position)
                    open_set.put((f_score, neighbor))

    return "WAIT"  # Default to waiting if no path is found