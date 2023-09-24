import heapq

import numpy as np

# Define constants for the game elements
FREE, CRATE, WALL, BOMB, COIN, OPPONENT = 0, 1, -1, 2, 3, 4

# Define valid actions
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]


def is_valid_move(new_position, field):
    x, y = new_position
    return 0 <= x < len(field) and 0 <= y < len(field[0]) and field[x][y] != WALL


def is_in_bomb_range(position, bombs):
    for (x, y), countdown in bombs:
        if x == position[0] and abs(y - position[1]) <= 3:
            return True
        if y == position[1] and abs(x - position[0]) <= 3:
            return True
    return False


def find_safe_tiles(current_position, bomb_range, field, bombs):
    safe_tiles = set()

    for dx in range(-bomb_range, bomb_range + 1):
        for dy in range(-bomb_range, bomb_range + 1):
            x, y = current_position[0] + dx, current_position[1] + dy
            new_position = (x, y)

            if is_valid_move(new_position, field) and not any(
                    (x, y) == new_position
                    for (x, y), countdown in bombs
            ):
                if field[new_position] != CRATE and field[new_position] != WALL and not is_in_bomb_range(
                        new_position, bombs):
                    safe_tiles.add(new_position)

    return safe_tiles


def shortest_path_to_safety(current_position, bomb_range, field, bombs):
    if not is_valid_move(current_position, field):
        return float('inf')  # Current position is a wall, no safe path

    safe_tiles = find_safe_tiles(current_position, bomb_range, field, bombs)
    if not safe_tiles:
        return float('inf')  # No safe tiles nearby

    shortest_distance = float('inf')

    for safe_tile in safe_tiles:
        visited = set()
        queue = [(0, current_position)]
        path = []  # Store the path

        while queue:
            steps, current = heapq.heappop(queue)

            if current in visited:
                continue

            visited.add(current)
            if field[current] == CRATE or field[current] == WALL:
                continue
            path.append(current)  # Add the current position to the path

            if current == safe_tile:
                if steps < shortest_distance:
                    shortest_distance = steps
                break

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                x, y = current[0] + dx, current[1] + dy
                new_position = (x, y)

                if is_valid_move(new_position, field) and new_position not in visited:
                    heapq.heappush(queue, (steps + 1, new_position))

    # print(f"Shortest path to safety: {shortest_path}")  # Print the shortest path
    return shortest_distance


# Define function to check if a position is valid (within the game board)
def is_valid_position(position, field):
    x, y = position
    return 0 <= x < len(field) and 0 <= y < len(field[0])


# Define function to calculate distances to the nearest coins in all four directions
def coin_features(game_state):
    position = game_state["self"][3]
    field = game_state["field"]
    coins = game_state["coins"]

    directions = ["RIGHT", "LEFT", "DOWN", "UP"]

    def dijkstra(start, is_free):
        queue = [(0, start)]
        visited = set()

        while queue:
            distance, current = heapq.heappop(queue)

            if current in visited:
                continue

            visited.add(current)

            if current in coins:
                return distance

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                x, y = current[0] + dx, current[1] + dy
                new_position = (x, y)

                if is_free(new_position) and is_valid_position(new_position, field):
                    heapq.heappush(queue, (distance + 1, new_position))

        return np.inf

    coin_features = []

    for direction in directions:
        dx, dy = 0, 0

        if direction == "RIGHT":
            dx = 1
        elif direction == "LEFT":
            dx = -1
        elif direction == "DOWN":
            dy = 1
        elif direction == "UP":
            dy = -1

        new_position = (position[0] + dx, position[1] + dy)

        def is_free(position):
            x, y = position
            if not is_valid_position(position, field) or field[x][y] == CRATE or field[x][y] == WALL:
                return False
            if direction == "RIGHT" and x + dx > 15:
                return False
            elif direction == "LEFT" and x + dx < 1:
                return False
            elif direction == "DOWN" and y + dy > 15:
                return False
            elif direction == "UP" and y + dy < 1:
                return False
            return True

        distance = dijkstra(new_position, is_free)

        # Calculate the inverse distance (1 divided by the distance)
        inverse_distance = 1.0 / distance if distance != np.inf and distance != 0 else 0.0
        coin_features.append(inverse_distance)
    return coin_features


# Define function to calculate distances to the nearest crates in all four directions
def crate_features(game_state):
    position = game_state["self"][3]
    field = game_state["field"]
    crates = [(x, y) for x in range(17) for y in range(17) if field[x][y] == CRATE]

    directions = ["RIGHT", "LEFT", "DOWN", "UP"]

    crate_features = []

    for direction in directions:
        dx, dy = 0, 0

        if direction == "RIGHT":
            dx = 1
        elif direction == "LEFT":
            dx = -1
        elif direction == "DOWN":
            dy = 1
        elif direction == "UP":
            dy = -1

        new_position = (position[0] + dx, position[1] + dy)

        def dijkstra(start, is_free):
            queue = [(0, start)]
            visited = set()

            while queue:
                distance, current = heapq.heappop(queue)

                if current in visited:
                    continue

                visited.add(current)

                if current in crates:
                    return distance

                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    x, y = current[0] + dx, current[1] + dy
                    new_position = (x, y)

                    if is_free(new_position) and is_valid_position(new_position, field):
                        heapq.heappush(queue, (distance + 1, new_position))

            return np.inf

        def is_free(position):
            x, y = position
            if not is_valid_position(position, field) or field[x][y] == WALL:
                return False
            if direction == "RIGHT" and x + dx > 15:
                return False
            elif direction == "LEFT" and x + dx < 1:
                return False
            elif direction == "DOWN" and y + dy > 15:
                return False
            elif direction == "UP" and y + dy < 1:
                return False
            return True

        distance = dijkstra(new_position, is_free)

        # Calculate the inverse distance (1 divided by the distance)
        inverse_distance = 1.0 / distance if distance != np.inf and distance != 0 else 0.0

        # Calculate the expected crate destruction points
        destruction_points = 0
        if inverse_distance > 0:
            for crate_position in crates:
                if crate_position != position:
                    crate_distance = np.linalg.norm(
                        np.array(crate_position) - np.array(new_position)
                    )
                    if crate_distance <= inverse_distance:
                        destruction_points += 1

        crate_features.append(inverse_distance * destruction_points)

    return crate_features


# Define function to calculate opponents feature
def opponent_features(game_state):
    position = game_state["self"][3]
    field = game_state["field"]
    opponents = game_state["others"]

    directions = ["RIGHT", "LEFT", "DOWN", "UP"]

    def dijkstra(start, end):
        queue = [(0, start)]
        visited = set()

        while queue:
            distance, current = heapq.heappop(queue)

            if current in visited:
                continue

            visited.add(current)

            if current == end:
                return distance

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                x, y = current[0] + dx, current[1] + dy
                new_position = (x, y)

                if is_valid_position(new_position, field) and field[x][y] != WALL:
                    heapq.heappush(queue, (distance + 1, new_position))

        return float('inf')

    opponent_distances = []

    for direction in directions:
        dx, dy = 0, 0

        if direction == "RIGHT":
            dx = 1
        elif direction == "LEFT":
            dx = -1
        elif direction == "DOWN":
            dy = 1
        elif direction == "UP":
            dy = -1

        new_position = (position[0] + dx, position[1] + dy)

        if (
                not is_valid_position(new_position, field)
                or field[new_position[0]][new_position[1]] == WALL
        ):
            # If moving in this direction would hit a wall or go out of bounds,
            opponent_distances.append(-1)
        else:
            nearest_opponent_distance = 9999
            for opponent_position in opponents:
                distance = dijkstra(new_position, opponent_position[3])
                nearest_opponent_distance = min(nearest_opponent_distance, distance)
            opponent_distances.append(
                1.0 / nearest_opponent_distance if nearest_opponent_distance != np.inf and nearest_opponent_distance != 0 else -1.0)

    return opponent_distances


# Define function to calculate bomb features if placed at the current position
def bomb_features(game_state):
    position = game_state["self"][3]
    field = game_state["field"]
    opponents = game_state["others"]
    bomb_range = 3  # Constant bomb range

    if not game_state['self'][2]:
        return [-1]

    def count_destroyed_crates(position):
        destroyed_crates = 0

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for i in range(1, bomb_range + 1):
                x, y = position[0] + dx * i, position[1] + dy * i
                if not is_valid_position((x, y), field) or field[x][y] == WALL:
                    break  # Stop if we encounter a wall
                if field[x][y] == CRATE:
                    destroyed_crates += 1

        return destroyed_crates

    def count_killed_opponents(position):
        killed_opponents = 0

        for opponent in opponents:
            if opponent[0] == position:
                killed_opponents += 1

        return killed_opponents

    destroyed_crates = count_destroyed_crates(position)
    killed_opponents = count_killed_opponents(position)

    bomb_effect = destroyed_crates + 10 * killed_opponents

    # Check if the bomb can't destroy anything or if there is no safe path to escape
    if bomb_effect == 0 or shortest_path_to_safety(position, bomb_range, field, [(position, 3)]) == float('inf'):
        return [-1]
    else:
        return [bomb_effect]


# Define function to calculate the nearest distances to safe tiles in all direction if there is a bomb
def safety_features(game_state):
    position = game_state["self"][3]
    field = game_state["field"]
    bombs = game_state["bombs"]
    bomb_range = 3  # Constant bomb range

    directions = ["RIGHT", "LEFT", "DOWN", "UP"]

    if is_in_bomb_range(position, bombs):
        min_steps_to_safety = float('inf')
        best_direction = -1

        for i, direction in enumerate(directions):
            dx, dy = 0, 0

            if direction == "RIGHT":
                dx = 1
            elif direction == "LEFT":
                dx = -1
            elif direction == "DOWN":
                dy = 1
            elif direction == "UP":
                dy = -1

            new_position = (position[0] + dx, position[1] + dy)
            steps = shortest_path_to_safety(new_position, bomb_range, field, bombs)

            if steps < min_steps_to_safety:
                min_steps_to_safety = steps
                best_direction = i

        safety_features = [-1 if i != best_direction else 5 for i in range(4)]
    else:
        safety_features = [0, 0, 0, 0]

    return safety_features


# Define function to calculate danger feature
def danger_features(game_state):
    position = game_state["self"][3]

    danger_features = []
    bombs = game_state['bombs']
    directions = ["RIGHT", "LEFT", "DOWN", "UP"]

    for direction in directions:
        dx, dy = 0, 0

        if direction == "RIGHT":
            dx = 1
        elif direction == "LEFT":
            dx = -1
        elif direction == "DOWN":
            dy = 1
        elif direction == "UP":
            dy = -1

        next_position = position

        next_position = (next_position[0] + dx, next_position[1] + dy)

        def is_in_bomb_range(position, bombs):
            for (x, y), countdown in bombs:
                if x == position[0] and abs(y - position[1]) <= 3:
                    return True
                if y == position[1] and abs(x - position[0]) <= 3:
                    return True
            return False

        # Check if the current position is within the game board
        if not is_valid_position(next_position, game_state["field"]):
            danger_features.append(0)  # No danger in this direction
            break

        if len(bombs) > 0 and not is_in_bomb_range(position, bombs):
            if is_in_bomb_range(next_position, bombs):
                danger_features.append(-1)
            else:
                danger_features.append(0)  # No danger in this direction
        else:
            # Check if there is a bomb or explosion at the current position

            if game_state["explosion_map"][next_position[0]][next_position[1]] > 0:
                danger_features.append(-1)
            else:
                danger_features.append(0)  # No danger in this direction

    return danger_features


# Define function to convert game state to features
def state_to_features(self, game_state):
    features = []

    # Calculate coin distances
    coin_distances = coin_features(game_state)
    features.extend(coin_distances)

    # Calculate crate distances
    crate_distances = crate_features(game_state)
    features.extend(crate_distances)

    # Calculate opponents distances
    opponent_distances = opponent_features(game_state)
    features.extend(opponent_distances)

    # Calculate bomb scope feature
    bomb_feature = bomb_features(game_state)
    features.extend(bomb_feature)

    # Calculate feature to indicate safe path
    safety_feature = safety_features(game_state)
    features.extend(safety_feature)

    # Calculate danger feature
    danger_feature = danger_features(game_state)
    features.extend(danger_feature)

    # Convert features to a numpy array for easy computation
    feature_array = np.array(features)

    # Compute mean and standard deviation for each feature
    mean = np.mean(feature_array, axis=0)
    std = np.std(feature_array, axis=0)

    # Normalize the features
    normalized_features = (feature_array - mean) / (std + 1e-8)  # Adding a small constant to avoid division by zero

    return normalized_features.reshape(1, -1)
