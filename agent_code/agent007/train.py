from agent_code.agent007_remake.features import state_to_features
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
# TODO: Remove hard coded value
TOTAL_EPISODES = 5000
LINEAR_CONSTANT_QUOTIENT = 0.9
EPSILON = (0.9, 0.4)
TRANSITION_HISTORY_SIZE = 3
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
    self.epsilon_arr = generate_eps_greedy_policy(self, LINEAR_CONSTANT_QUOTIENT)

# Rest of the code remains similar to your original `train.py` but adapted for DQN
def generate_eps_greedy_policy(self, q):
    '''
    :param self: the network that is used for training
            (contains eps-threshold for start and end of training
            and the number of total episodes)
    :param q: the fraction of the training where the eps-threshold is linearly diminished

    returns: array containing the eps-thresholds for training
    '''
    N = self.training_episodes
    N_1 = int(N * q)
    N_2 = N - N_1
    eps1 = np.linspace(self.epsilon_begin, self.epsilon_end, N_1)
    if N_1 == N:
        return eps1
    eps2 = np.ones(N_2) * self.epsilon_end
    return np.append(eps1, eps2)


def calculate_round_score(events: List[str]) -> int:
    """
    Calculate the score for a round based on game events.

    :param events: The events that occurred during a round.
    :return: The round score.
    """
    coin_reward = events.count("COIN_COLLECTED")  # 1 point per coin

    opponent_kills = events.count("KILLED_OPPONENT")
    opponent_reward = opponent_kills * 20 # 20 points per opponent killed

    return coin_reward + opponent_reward


def track_game_score(self, smooth=False):
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
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    reward = reward_from_events(self, events)
    self.dqn_agent.update_q_network(state_to_features(last_game_state), last_action, reward, None)
    self.dqn_agent.save_model("required_parameters/saved-model.pt")

    round_score = calculate_round_score(events)
    scores_per_round.append(round_score)  # Store the round score
    self.episode_counter += 1
    track_game_score(self)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
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
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
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
    """
    Give rewards immediately after placing bombs for crate destruction.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param events: Predefined events (events.py) that occurred in the game step.

    :return: Reward based on how many crates will be destroyed by a dropped bomb.
    """
    crate_destruction_reward = 0

    # Check if any crates were destroyed
    if e.BOMB_DROPPED in events:
        # Count the number of crates destroyed by the bomb
        crate_destruction_reward = events.count(e.CRATE_DESTROYED) * 15

    self.logger.info(f"Awarded {crate_destruction_reward} for crate destruction")
    return crate_destruction_reward