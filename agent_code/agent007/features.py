import numpy as np
def state_to_features(game_state: dict) -> np.array:
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
    coin_space = [1 / (player_position[0] - coin[0]) if player_position[0] > coin[0] else
                      1 / (coin[0] - player_position[0]) if player_position[0] < coin[0] else 0 for coin in coins]
    channels.extend(coin_space)

    # Feature 2: Crate Distances (right, left, down, up)
    crate_space = [1 / (player_position[0] - crate[0]) if player_position[0] > crate[0] else
                       1 / (crate[0] - player_position[0]) if player_position[0] < crate[0] else 0 for crate in field]
    channels.extend(crate_space)

    # Feature 3: Bomb Here Feature
    bomb_here = 0
    for bomb in bombs:
        if bomb[0] == player_position:
            bomb_here = 1
            break
    channels.append(bomb_here)

    # Feature 4: Running Away Feature (right, left, down, up)
    escape_bomb = [0, 0, 0, 0]
    for i, direction in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
        next_position = (player_position[0] + direction[0], player_position[1] + direction[1])
        if not any(np.array_equal(next_position, opponent) for opponent in opponents) and \
                (next_position[0] >= 0 and next_position[0] < len(field) and
                 next_position[1] >= 0 and next_position[1] < len(field[0])):
            escape_bomb[i] = 1
    channels.extend(escape_bomb)

    # Feature 5: Opponent Distances (right, left, down, up)
    enemy_space = []
    for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        closest_opponent_distance = float("inf")
        for opponent_position in opponents:
            distance = np.linalg.norm(np.array(player_position) - np.array(opponent_position))
            if distance < closest_opponent_distance:
                closest_opponent_distance = distance
        enemy_space.extend([1 / closest_opponent_distance if closest_opponent_distance < float("inf") else 0])

    channels.extend(enemy_space)

    # Pad or truncate the channels to a fixed size of 17
    if len(channels) < 17:
        channels.extend([0] * (17 - len(channels)))
    elif len(channels) > 17:
        channels = channels[:17]

    # Convert the feature channels to a numpy array
    feature_tensor = np.array(channels)

    # Return the feature tensor as a flattened vector
    return feature_tensor.reshape(1, -1)
