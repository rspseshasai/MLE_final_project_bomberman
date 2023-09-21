import numpy as np


import numpy as np

import numpy as np


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e., a feature tensor.

    :param game_state: A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        return None

    # Extract relevant information from the game state dictionary
    field = game_state["field"]
    player_position = game_state["self"][3]
    coins = game_state["coins"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    opponents = [opponent[3] for opponent in game_state["others"]]

    # Initialize empty feature channels
    channels = []

    # Feature 1: Coin Distances (right, left, down, up)
    coin_distances = [1 / (player_position[0] - coin[0]) if player_position[0] > coin[0] else
                      1 / (coin[0] - player_position[0]) if player_position[0] < coin[0] else 0 for coin in coins]
    channels.extend(coin_distances)

    # Feature 2: Crate Distances (right, left, down, up)
    crate_distances = [1 / (player_position[0] - crate[0]) if player_position[0] > crate[0] else
                       1 / (crate[0] - player_position[0]) if player_position[0] < crate[0] else 0 for crate in field]
    channels.extend(crate_distances)

    # Feature 3: Bomb Here Feature
    bomb_here = 0
    for bomb in bombs:
        if bomb[0] == player_position:
            bomb_here = 1
            break
    channels.append(bomb_here)

    # Feature 4: Negative Explosion Timer Feature
    explosion_timer = explosion_map[player_position[0]][player_position[1]]
    negative_explosion_timer = -explosion_timer if explosion_timer > 0 else 0
    channels.append(negative_explosion_timer)

    # Feature 5: Running Away Feature (right, left, down, up)
    running_away = [0, 0, 0, 0]
    for i, direction in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
        next_position = (player_position[0] + direction[0], player_position[1] + direction[1])
        if not any(np.array_equal(next_position, opponent) for opponent in opponents) and \
                (next_position[0] >= 0 and next_position[0] < len(field) and
                 next_position[1] >= 0 and next_position[1] < len(field[0])):
            running_away[i] = 1
    channels.extend(running_away)

    # Feature 6: Danger Feature (right, left, down, up)
    danger = [0, 0, 0, 0]
    for i, direction in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
        next_position = (player_position[0] + direction[0], player_position[1] + direction[1])
        if explosion_map[next_position[0]][next_position[1]] > 0:
            danger[i] = 1
    channels.extend(danger)

    # Feature 7: Opponent Distances (right, left, down, up)
    opponent_distances = []
    for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        closest_opponent_distance = float("inf")
        for opponent_position in opponents:
            distance = np.linalg.norm(np.array(player_position) - np.array(opponent_position))
            if distance < closest_opponent_distance:
                closest_opponent_distance = distance
        opponent_distances.extend([1 / closest_opponent_distance if closest_opponent_distance < float("inf") else 0])

    channels.extend(opponent_distances)

    # Feature 8: Remaining Positive Total Reward
    remaining_coins = len(coins)
    remaining_crates = sum(crate == 1 for crate in field.tolist())
    remaining_positive_total_reward = remaining_coins + remaining_crates * 0.1
    channels.append(remaining_positive_total_reward)

    # Pad or truncate the channels to a fixed size of 23
    if len(channels) < 23:
        channels.extend([0] * (23 - len(channels)))
    elif len(channels) > 23:
        channels = channels[:23]

    # Convert the feature channels to a numpy array
    feature_tensor = np.array(channels)

    # Return the feature tensor as a flattened vector
    return feature_tensor.reshape(1, -1)
