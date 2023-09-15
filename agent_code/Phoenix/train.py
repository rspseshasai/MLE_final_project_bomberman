import pickle
import random
from collections import namedtuple, deque
from typing import List

import matplotlib
import numpy as np
import torch
from torch import nn, optim

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
ACTIONS_IDX = {'LEFT': 0, 'RIGHT': 1, 'UP': 2, 'DOWN': 3, 'WAIT': 4, 'BOMB': 5}

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    self.score = 0
    self.game_score_arr = []
    self.episode_counter = 0
    self.total_episodes = 200
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


def add_experience(self, old_game_state, self_action, new_game_state, events):
    old_features = state_to_features(self, old_game_state)
    if old_features is not None:
        if new_game_state is None:
            new_features = old_features
        else:
            new_features = state_to_features(self, new_game_state)
        reward = reward_from_events(self, events)
        reward += rewards_from_own_events(self, old_game_state, self_action, new_game_state, events)

        action_idx = ACTIONS_IDX[self_action]
        action = torch.zeros(6)
        action[action_idx] = 1

        self.transitions.append((old_features, action, reward, new_features))

        # TODO
        number_of_elements_in_buffer = len(self.transitions)
        if number_of_elements_in_buffer > self.buffer_size:
            self.transitions.popleft()


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
    '''
    model: the model that gets updated
    experience_buffer: the collected experiences, list of game_episodes
    '''
    model = self.model
    experience_buffer = self.transitions

    # randomly choose batch out of the experience buffer
    number_of_elements_in_buffer = len(experience_buffer)
    batch_size = min(number_of_elements_in_buffer, self.batch_size)

    random_i = [random.randrange(number_of_elements_in_buffer) for _ in range(batch_size)]

    # compute for each experience in the batch
    # - the Ys using n-step TD Q-learning
    # - the current guess for the Q function
    sub_batch = []
    Y = []
    for i in random_i:
        random_experience = experience_buffer[i]
        sub_batch.append(random_experience)

    for b in sub_batch:
        old_state = b[0]
        action = b[1]
        reward = b[2]
        new_state = b[3]

        y = reward
        if new_state is not None:
            y += self.gamma * torch.max(model(new_state))

        Y.append(y)

    Y = torch.tensor(Y)

    # Qs
    states = torch.cat(tuple(b[0] for b in sub_batch))  # put all states of the sub_batch in one batch
    q_values = model(states)
    actions = torch.cat([b[1].unsqueeze(0) for b in sub_batch])
    Q = torch.sum(q_values * actions, dim=1)

    loss = self.loss_function(Q, Y)
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

    # add_experience(self, last_game_state, last_action, None, events)
    # if len(self.experience_buffer) > 0:
    #     update_network(self)

    self.score += calculate_score(events)

    plot_training_graph(self)

    add_experience(self, last_game_state, last_action, None, events)
    if len(self.transitions) > 0:
        update_model(self)

    self.episode_counter += 1
    if self.episode_counter % (self.total_episodes // 10) == 0:  # save parameters 2 times
        print("\nsaving to phoenix.pt " + "my-phoenix-model")
        torch.save(self.model.state_dict(), f"saved/my-phoenix-model.pt")


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
