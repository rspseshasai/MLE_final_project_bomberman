from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from scipy.ndimage import uniform_filter1d

import events as e
from .features import state_to_features
from .model import QLearningAgent

# Hyperparameters
# Initialize variables for tracking training progress
scores_per_round = []
game_score_arr = []
TOTAL_ROUNDS = 100000
LINEAR_CONSTANT_QUOTIENT = 0.85
EPSILON = (0.6, 0.2)


def setup_training(self):
    self.q_learning_agent = QLearningAgent()
    self.episode_counter = 0
    # Hyperparameters
    self.q_learning_agent.LEARNING_RATE = 0.001
    self.q_learning_agent.DISCOUNT_FACTOR = 0.85
    self.q_learning_agent.optimizer = optim.Adam(self.q_learning_agent.q_network.parameters(),
                                                 lr=self.q_learning_agent.LEARNING_RATE)
    self.total_rounds = TOTAL_ROUNDS
    self.epsilon_range = epsilon_greedy_policy(self, LINEAR_CONSTANT_QUOTIENT)


def epsilon_greedy_policy(self, q):
    N = self.total_rounds
    N_1 = int(N * q)
    N_2 = N - N_1
    eps1 = np.linspace(EPSILON[0], EPSILON[1], N_1)
    if N_1 == N:
        return eps1
    eps2 = np.ones(N_2) * EPSILON[1]
    return np.append(eps1, eps2)


def calculate_round_score(events: List[str]) -> int:
    """
    Calculate the score for a round based on game events.

    :param events: The events that occurred during a round.
    :return: The round score.
    """
    coin_reward = events.count("COIN_COLLECTED")  # 1 point per coin

    opponent_kills = events.count("KILLED_OPPONENT")
    opponent_reward = 5 * opponent_kills  # 5 points per opponent killed

    return coin_reward + opponent_reward


def plot_scores(self, smooth=False):
    global scores_per_round
    global game_score_arr

    total_score = sum(scores_per_round)
    game_score_arr.append(total_score)
    scores_per_round = []
    y = game_score_arr
    if smooth:
        window_size = self.episode_counter // 25
        if window_size < 1:
            window_size = 1
        y = uniform_filter1d(y, window_size, mode="nearest", output="float")
    x = range(len(y))

    fig, ax = plt.subplots()
    ax.set_title('Score per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Points')
    ax.plot(x, y, marker='o', markersize=3, linewidth=1)

    plt.savefig('saved_parameters/training_progress.png')
    plt.close()


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    reward = reward_from_events(self, events)
    self.q_learning_agent.update_q_table(state_to_features(self, last_game_state), last_action, reward, None)
    self.q_learning_agent.save_model("saved_parameters/model.pt")

    round_score = calculate_round_score(events)
    scores_per_round.append(round_score)  # Store the round score
    self.episode_counter += 1
    plot_scores(self)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Calculate the standard reward from events
    reward = reward_from_events(self, events)

    # Calculate the immediate reward for crate destruction after bomb placement
    crate_destruction_reward = reward_for_crate_destruction(self, events)

    # Sum both rewards
    total_reward = reward + crate_destruction_reward

    # Update the Q-table once with the total reward
    self.q_learning_agent.update_q_table(state_to_features(self, old_game_state), self_action, total_reward,
                                         state_to_features(self, new_game_state))

    round_score = calculate_round_score(events)
    scores_per_round.append(round_score)  # Store the round score


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.KILLED_SELF: 0,
        e.GOT_KILLED: -500,
        e.COIN_COLLECTED: 100,
        e.BOMB_DROPPED: -1,
        e.WAITED: -1,
        e.KILLED_OPPONENT: 500,
        e.INVALID_ACTION: -10,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def reward_for_crate_destruction(self, events: List[str]) -> int:
    crate_destruction_reward = 0

    # Check if any crates were destroyed
    if e.BOMB_DROPPED in events:
        # Count the number of crates destroyed by the bomb
        crate_destruction_reward = events.count(e.CRATE_DESTROYED) * 30

    self.logger.info(f"Awarded {crate_destruction_reward} for crate destruction")
    return crate_destruction_reward
