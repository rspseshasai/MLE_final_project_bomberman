from collections import deque, namedtuple
import pickle
from typing import List
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

from .callbacks import state_to_features, ACTIONS
from .model import QNetwork  # Import your QNetwork model
import events as e

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
TRANSITION_HISTORY_SIZE = 1000000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
BATCH_SIZE = 50
REPLAY_BUFFER_SIZE = 1000000
LEARNING_RATE = 0.0001  # Adjust as needed

# Initialize Q-network and optimizer
input_size = 1734  # Define the input size based on your state representation
hidden_size = 256  # Define the hidden layer size
output_size = len(ACTIONS)  # The number of possible actions
q_network = QNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

# Initialize experience replay buffer
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

# Initialize lists to store training statistics
rounds = []
training_accuracy = []
losses = []
scores = []


def setup_training(self):
    self.score = 0
    self.game_score_arr = []
    self.episode_counter = 0
    self.total_episodes = 1000
    self.pos_saver = []
    self.gamma = 0.8
    self.learning_rate = 0.001
    self.batch_size = 50
    self.buffer_size = 200
    self.epsilon_begin = 1.0
    self.epsilon_end = 0.0001
    self.epsilon_arr = generate_eps_greedy_policy(self)

    self.loss_function = nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    self.loss_function = nn.MSELoss()
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.discount_factor = 0.8  # Define your discount factor


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 5,
        e.MOVED_RIGHT: -0.01,
        e.MOVED_LEFT: -0.01,
        e.MOVED_UP: -0.01,
        e.MOVED_DOWN: -0.01,
        e.WAITED: -0.5,
        e.INVALID_ACTION: -0.5,
        e.BOMB_DROPPED: -20,
        e.KILLED_SELF: -40,
        e.GOT_KILLED: -50,
        e.CRATE_DESTROYED: -0.5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


# Update game_events_occurred
# Define a variable to keep track of the total score
total_score = 0


# Update game_events_occurred
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    self.score += calculate_score(events)

    # Calculate the step score based on rewards obtained from events
    step_score = reward_from_events(self, events)
    step_score += rewards_from_own_events(self, old_game_state, self_action, new_game_state, events)

    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), step_score))

    # Update experience replay buffer
    if old_game_state is not None:
        state = state_to_features(old_game_state)
        next_state = state_to_features(new_game_state)
        action = ACTIONS.index(self_action)
        transition = Transition(state, action, next_state, step_score)
        replay_buffer.append(transition)

        # Perform Q-learning with experience replay after each step
        if len(replay_buffer) >= BATCH_SIZE:
            batch = np.random.choice(len(replay_buffer), BATCH_SIZE, replace=False)
            batch_transitions = [replay_buffer[i] for i in batch]
            batch_state_array = np.array([t[0] for t in batch_transitions], dtype=np.float32)
            batch_state = torch.tensor(batch_state_array)
            batch_action = torch.tensor([t[1] for t in batch_transitions], dtype=torch.int64)
            batch_reward = torch.tensor([t[3] for t in batch_transitions], dtype=torch.float32)
            batch_next_state_array = np.array([t[2] for t in batch_transitions], dtype=np.float32)
            batch_next_state = torch.tensor(batch_next_state_array)

            q_values = q_network(batch_state)
            q_values_next = q_network(batch_next_state)

            target_q_values = q_values.clone()
            target_q_values[range(BATCH_SIZE), batch_action] = batch_reward + self.discount_factor * \
                                                               torch.max(q_values_next, dim=1)[0]

            loss = F.smooth_l1_loss(q_values, target_q_values.detach())
            self.logger.debug(f'loss: {loss} at the end of this step')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save the model after training if needed
            with open("saved_models/my-saved-model.pt", "wb") as file:
                pickle.dump(q_network, file)


def calculate_score(events):
    true_game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
    }
    score = 0
    for event in events:
        if event in true_game_rewards:
            score += true_game_rewards[event]
    return score


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.episode_counter += 1
    self.score += calculate_score(events)
    plot_training_graph(self)

    step_score = reward_from_events(self, events)
    step_score += rewards_from_own_events(self, last_game_state, last_action, last_game_state, events)

    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, reward_from_events(self, events), step_score))

    # Perform Q-learning with experience replay
    if len(replay_buffer) >= BATCH_SIZE:
        batch = np.random.choice(len(replay_buffer), BATCH_SIZE, replace=False)
        batch_transitions = [replay_buffer[i] for i in batch]
        # Assuming batch_transitions is a list of numpy arrays
        batch_state_array = np.array([t[0] for t in batch_transitions], dtype=np.float32)
        batch_state = torch.tensor(batch_state_array)
        batch_action = torch.tensor([t[1] for t in batch_transitions], dtype=torch.int64)
        batch_reward = torch.tensor([t[3] for t in batch_transitions], dtype=torch.float32)
        batch_next_state_array = np.array([t[2] for t in batch_transitions], dtype=np.float32)
        batch_next_state = torch.tensor(batch_next_state_array)

        q_values = q_network(batch_state)
        q_values_next = q_network(batch_next_state)

        target_q_values = q_values.clone()
        target_q_values[range(BATCH_SIZE), batch_action] = batch_reward + self.discount_factor * \
                                                           torch.max(q_values_next, dim=1)[0]

        loss = F.smooth_l1_loss(q_values, target_q_values.detach())
        self.logger.debug(f'loss: {loss} at the end of this round')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save the model after training if needed
        with open("saved_models/my-saved-model.pt", "wb") as file:
            pickle.dump(q_network, file)


def plot_training_graph(self, smooth=False):
    self.game_score_arr.append(self.score)
    self.score = 0

    # plot scores
    y = self.game_score_arr
    if smooth:
        window_size = self.total_episodes // 25
        if window_size < 1:
            window_size = 1
        y = uniform_filter1d(y, window_size, mode="nearest", output="float")
    x = range(len(y))

    fig, ax = plt.subplots()
    ax.set_title('score')
    ax.set_xlabel('episode')
    ax.set_ylabel('total points')
    ax.plot(x, y, marker='o', markersize=3, linewidth=1)

    plt.savefig('saved_models/training_progress.png')

    plt.close()


def rewards_from_own_events(self, old_game_state, action, new_game_state, events):
    reward_sum = 0

    # check if agent moved closer to next coin
    reward_sum += moved_closer_to_next_coin(old_game_state, action, events)
    reward_sum += loop_killer(self, new_game_state)

    self.logger.info(f"Awarded {reward_sum} for own transition events")
    return reward_sum


def moved_closer_to_next_coin(old_game_state, action, events):
    if old_game_state is None:
        return 0

    if e.INVALID_ACTION in events:
        return 0

    good, bad = 0.05, -0.06

    agent_x, agent_y = agent_x, agent_y = old_game_state['self'][3]
    coin = closest_coin(agent_x, agent_y, old_game_state['coins'])
    if (coin[0] == 1) and (action == 'RIGHT'):
        return good
    elif (coin[1] == 1) and (action == 'LEFT'):
        return good
    elif (coin[2] == 1) and (action == 'DOWN'):
        return good
    elif (coin[3] == 1) and (action == 'UP'):
        return good
    else:
        return bad


def loop_killer(self, new_game_state):
    if new_game_state is None:
        return 0
    loop = False
    if self.pos_saver.count(new_game_state["self"][3]) > 3:
        loop = True
    self.pos_saver.append(new_game_state["self"][3])
    if len(self.pos_saver) > 10:
        self.pos_saver.pop(0)
    if loop:
        return -0.5
    else:
        return 0


def closest_coin(agent_x, agent_y, game_state_coins):
    coins = torch.zeros(4)
    closest_coin = None
    closest_dist = 100
    for coin_x, coin_y in game_state_coins:
        dist = np.linalg.norm([coin_x - agent_x, coin_y - agent_y])
        if dist < closest_dist:
            closest_dist = dist
            closest_coin = [coin_x, coin_y]

    if closest_coin is not None:
        x, y = closest_coin
        if x - agent_x > 0:
            coins[0] = 1
        elif x - agent_x < 0:
            coins[1] = 1

        if y - agent_y > 0:
            coins[2] = 1
        elif y - agent_y < 0:
            coins[3] = 1

    return coins


def generate_eps_greedy_policy(self):
    # __ANSATZ 1: Linear__#
    # return np.linspace(network.epsilon_begin, network.epsilon_end, network.training_episodes)

    # __ANSATZ 2: Linear + Const__#
    N = self.total_episodes
    N_1 = int(N * 0.7)
    N_2 = N - N_1
    eps1 = np.linspace(self.epsilon_begin, self.epsilon_end, N_1)
    eps2 = np.ones(N_2) * self.epsilon_end
    return np.append(eps1, eps2)
