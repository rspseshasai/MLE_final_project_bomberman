import pickle
import random
from collections import namedtuple, deque
from typing import List

import matplotlib
import numpy as np
import torch
from torch import nn

from .model import Phoenix

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1000000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    self.score = 0
    self.game_score_arr = []
    self.episode_counter = 0
    self.total_episodes = 1000
    self.pos_saver = []
    self.batch_size = 32
    self.gamma = 0.5
    self.learning_rate = 0.001
    self.batch_size = 64
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    self.loss_function = nn.MSELoss()

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def add_experience(self, old_game_state, self_action, new_game_state, events):
    # Calculate rewards from events
    reward = reward_from_events(self, events)

    # Calculate rewards from own events
    reward += rewards_from_own_events(self, old_game_state, self_action, new_game_state, events)

    # Convert game states to features
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    # Create a transition tuple (s, a, r, s') and add it to the experience buffer
    transition = (old_features, action_to_index(self, self_action), reward, new_features)
    self.transitions.append(transition)


def action_to_index(self, action):
    # Define a method to convert actions to indices
    if action == 'UP':
        return 0
    elif action == 'RIGHT':
        return 1
    elif action == 'DOWN':
        return 2
    elif action == 'LEFT':
        return 3
    elif action == 'WAIT':
        return 4
    elif action == 'BOMB':
        return 5
    else:
        raise ValueError(f"Invalid action: {action}")


def update_model(self):
    if len(self.transitions) >= self.batch_size:
        # Sample a batch of transitions
        batch = random.sample(self.transitions, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        # Convert to tensors
        states = torch.stack([torch.tensor(s) for s in states])  # Convert NumPy arrays to PyTorch tensors
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.stack([torch.tensor(s) for s in next_states])  # Convert NumPy arrays to PyTorch tensors

        # Compute Q-values for the current and next states
        current_q_values = self.model(states)
        next_q_values = self.model(next_states)

        # Compute target Q-values using the Bellman equation
        target_q_values = rewards + self.gamma * torch.max(next_q_values, dim=1).values

        # Get the Q-values for the selected actions
        selected_q_values = torch.gather(current_q_values, 1, actions.unsqueeze(1))

        # Calculate the loss using the mean squared error loss function
        loss = torch.mean((selected_q_values - target_q_values.unsqueeze(1)) ** 2)

        # Zero the gradients, perform backpropagation, and update the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    # 1. Add experience
    add_experience(self, old_game_state, self_action, new_game_state, events)

    # 2. Update the model
    update_model(self)

    self.logger.info('####################')
    self.logger.info(events)
    # TODO: Update PLS
    self.score += calculate_score(events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.score += calculate_score(events)
    plot_training_graph(self)

    # 1. Add experience
    add_experience(self, last_game_state, last_action, last_game_state, events)

    # 2. Update the model
    update_model(self)

    # Store the model
    with open("saved/my-phoenix-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 5,
        e.MOVED_RIGHT: -0.01,
        e.MOVED_LEFT: -0.01,
        e.MOVED_UP: -0.01,
        e.MOVED_DOWN: -0.01,
        e.WAITED: -0.5,
        e.INVALID_ACTION: -0.5,
        e.BOMB_DROPPED: -0.01,
        e.KILLED_SELF: -20,
        e.GOT_KILLED: -10,
        e.CRATE_DESTROYED: 0.5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


# TODO: Below all functions update PLS
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

    plt.savefig('saved/training_progress.png')

    plt.close()
