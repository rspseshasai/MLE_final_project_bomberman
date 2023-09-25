from .features import state_to_features
from .model import DQNAgent
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from scipy.ndimage import uniform_filter1d
import events as e

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Initialize variables for tracking training progress
scores_per_round = []
game_score_arr = []
TOTAL_EPISODES = 100000
LINEAR_CONSTANT_QUOTIENT = 0.9
EPSILON = (0.9, 0.4)
TRANSITION_HISTORY_SIZE = 3200
RECORD_ENEMY_TRANSITIONS = 1.0
PLACEHOLDER_EVENT = "PLACEHOLDER"

def setup_training(self):
    # Initialize a DQN agent instance
    self.dqn_agent = DQNAgent()
    self.episode_counter = 0
    # Hyperparameters and setup for DQN training
    self.dqn_agent.LEARNING_RATE = 0.001
    self.dqn_agent.EXPLORATION_PROB = 0.1
    self.dqn_agent.DISCOUNT_FACTOR = 0.9
    self.dqn_agent.optimizer = optim.Adam(self.dqn_agent.q_network.parameters(),
                                          lr=self.dqn_agent.LEARNING_RATE)
    self.epsilon_begin = EPSILON[0]
    self.epsilon_end = EPSILON[1]
    self.training_episodes = TOTAL_EPISODES
    self.epsilon_arr = generate_eps(self, LINEAR_CONSTANT_QUOTIENT)



def generate_eps(self, q):

    N = self.training_episodes
    N_1 = int(N * q)
    N_2 = N - N_1
    eps1 = np.linspace(self.epsilon_begin, self.epsilon_end, N_1)
    if N_1 == N:
        return eps1
    eps2 = np.ones(N_2) * self.epsilon_end
    return np.append(eps1, eps2)


def calculate_round_score(events: List[str]) -> int:

    coin_reward = events.count("COIN_COLLECTED")  # 1 point per coin

    opponent_kills = events.count("KILLED_OPPONENT")
    opponent_reward = opponent_kills * 20 # 20 points per opponent killed

    return coin_reward + opponent_reward

def track_game_score(self, smooth=False):
    #   Track the score for game round.

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

    plt.savefig('required_parameters/training_progress.png')
    plt.close()

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    reward = reward_from_events(self, events)
    self.dqn_agent.update_q_network(state_to_features(last_game_state), last_action, reward, None)
    self.dqn_agent.save_model("required_parameters/saved-model.pt")

    round_score = calculate_round_score(events)
    scores_per_round.append(round_score)  # Store the round score
    self.episode_counter += 1
    track_game_score(self)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Calculate the standard reward from events
    reward = reward_from_events(self, events)

    # Calculate the immediate reward for crate destruction after bomb placement
    crate_destruction_reward = reward_for_crate_destruction(self, events)

    # Sum both rewards
    total_reward = reward + crate_destruction_reward

    # Update the Q-table once with the total reward
    self.dqn_agent.update_q_network(state_to_features(old_game_state), self_action, total_reward,
                                         state_to_features(new_game_state))

    round_score = calculate_round_score(events)
    scores_per_round.append(round_score)  # Store the round score

def reward_from_events(self, events: List[str]) -> int:
    #Modify the rewards agent gets so as to en/discourage certain behavior.
    game_rewards = {
        e.COIN_COLLECTED: 20,
        e.KILLED_OPPONENT: 50,
        e.MOVED_RIGHT: 2,
        e.MOVED_LEFT: 2,
        e.MOVED_UP: 2,
        e.MOVED_DOWN: 2,
        e.WAITED: -1,
        e.INVALID_ACTION: -10,
        e.BOMB_DROPPED: 2,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -100
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
        crate_destruction_reward = events.count(e.CRATE_DESTROYED) * 15

    self.logger.info(f"Awarded {crate_destruction_reward} for crate destruction")
    return crate_destruction_reward
