import heapq

import numpy as np

# Define constants for the game elements
FREE, CRATE, WALL, BOMB, COIN, OPPONENT = 0, 1, -1, 2, 3, 4

# Define valid actions
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]


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


# Define function to calculate bomb features if placed at the current position
def bomb_features(game_state):
    position = game_state["self"][3]
    field = game_state["field"]
    opponents = game_state["others"]
    bombs = game_state["bombs"]
    bomb_range = 3  # Constant bomb range

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

    def is_safe_path_to_escape(position, bomb_range):
        visited = np.zeros_like(field, dtype=bool)
        queue = [position]

        while queue:
            x, y = queue.pop(0)

            if visited[x][y]:
                continue

            visited[x][y] = True

            if field[x][y] == WALL:
                continue  # Skip walls

            if any((x, y) == bomb[0] for bomb in bombs):
                continue  # Skip bomb positions

            if bomb_range == 0:
                return True  # Reached a safe tile

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_x, new_y = x + dx, y + dy

                if is_valid_position((new_x, new_y), field) and not visited[new_x][new_y]:
                    queue.append((new_x, new_y))

        return False  # No safe path to escape

    destroyed_crates = count_destroyed_crates(position)
    killed_opponents = count_killed_opponents(position)

    bomb_effect = destroyed_crates + 10 * killed_opponents

    # Check if the bomb can't destroy anything or if there is no safe path to escape
    if bomb_effect == 0 or not is_safe_path_to_escape(position, bomb_range):
        return [-1]
    else:
        return [bomb_effect]


# Define function to calculate safety features
def safety_features(game_state):
    position = game_state["self"][3]
    field = game_state["field"]
    explosion_map = game_state["explosion_map"]

    safety_features = []

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

        new_position = (position[0] + dx, position[1] + dy)

        if not is_valid_position(new_position, field) or field[new_position[0]][new_position[1]] == WALL:
            safety_features.append(-1)  # No path due to crates or walls
        else:
            # Check if the path is safe using the explosion map
            if explosion_map[new_position[0]][new_position[1]] == 0:
                safety_features.append(0)  # No explosion, but no safe path either
            else:
                # Calculate the steps required to go to the nearest safe tile
                steps = 0
                current_position = new_position
                while explosion_map[current_position[0]][current_position[1]] > 0:
                    steps += 1
                    current_position = (current_position[0] + dx, current_position[1] + dy)
                if steps != 0:
                    safety_features.append(1 / steps)
                else:
                    safety_features.append(steps)

    return safety_features


# Define function to calculate danger feature
def danger_features(game_state):
    position = game_state["self"][3]
    bombs = game_state["bombs"]

    danger_features = []

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

        current_position = position

        for _ in range(3):  # Check the next 3 tiles in the direction
            current_position = (current_position[0] + dx, current_position[1] + dy)

            # Check if the current position is within the game board
            if not is_valid_position(current_position, game_state["field"]):
                danger_features.append(0)  # No danger in this direction
                break

            # Check if there is a bomb or explosion at the current position

            if any(current_position == bomb[0] for bomb in bombs) or game_state["explosion_map"][current_position[0]][
                current_position[1]] > 0:
                danger_features.append(-1)
        else:
            danger_features.append(0)  # No danger in this direction

    return danger_features[:4]


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
            # set a large value to represent that it's not possible to reach an opponent
            opponent_distances.append(-1)
        else:
            nearest_opponent_distance = 9999
            for opponent_position in opponents:
                distance = dijkstra(new_position, opponent_position[3])
                nearest_opponent_distance = min(nearest_opponent_distance, distance)
            opponent_distances.append(
                1.0 / nearest_opponent_distance if nearest_opponent_distance != np.inf and nearest_opponent_distance != 0 else -1.0)

    return opponent_distances


# Define function to convert game state to features
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
    opponents_feature = opponent_features(game_state)
    features.extend(opponents_feature)

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