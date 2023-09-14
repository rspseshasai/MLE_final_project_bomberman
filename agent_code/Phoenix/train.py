from collections import deque, namedtuple
from typing import List
import torch
import events as e
from .callbacks import state_to_features, ACTIONS
from .model import QLearning

# This is only an example!
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters -- Modify as needed
TRANSITION_HISTORY_SIZE = 3  # Keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # Record enemy transitions with probability ...

# Events
COIN_COLLECTED_EVENT = e.COIN_COLLECTED
PLACEHOLDER_EVENT = "PLACEHOLDER"

# Define a variable to specify the number of episodes before saving parameters
TRAINING_EPISODES = 10  # Adjust this value as needed


def setup_training(self):
    """
    Initialize self for training purposes.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks, and you can set arbitrary values.
    """
    # Initialize game score, episode counter, and experience buffer
    self.game_score = 0
    self.episode_counter = 0
    self.experience_buffer = []

    # Create an instance of QLearning
    self.q_learning = QLearning(input_size=578, hidden_size=10, num_actions=len(ACTIONS), learning_rate=0.1, discount_factor=0.9)

    # Initialize the transitions deque
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks, and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from `old_game_state` to `new_game_state`.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Get the state representation (you should implement state_to_features function)
    state = state_to_features(old_game_state)

    # Get the action chosen by Q-learning
    action = self.q_learning.get_action(state)

    # Provide rewards for coin collection
    if COIN_COLLECTED_EVENT in events:
        self.logger.info(f"Collected a coin in step {new_game_state['step']}")
        events.append(PLACEHOLDER_EVENT)  # You can add a custom event for coin collection if needed

    # Update Q-values based on the observed rewards and state transitions
    reward = reward_from_events(self, events)
    next_state = state_to_features(new_game_state)
    self.q_learning.update_q_value(state, action, reward, next_state)

    # Store the chosen action for the next step
    self.next_action = ACTIONS[action]


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    :param last_game_state: The last game state received.
    :param last_action: The last action taken.
    :param events: The events that occurred in the final step.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in the final step')

    # Save the model using torch.save
    torch.save(self.q_learning.state_dict(), 'my-saved-model.pt')


def reward_from_events(self, events: List[str]) -> int:
    """
    Calculate the cumulative reward from a list of events.

    :param self: This object is passed to all callbacks, and you can set arbitrary values.
    :param events: The events for which to calculate the reward.
    :return: The cumulative reward.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,       # Positive reward for collecting a coin
        e.KILLED_OPPONENT: -5,     # Negative reward for killing an opponent (optional)
        e.INVALID_ACTION: -1,      # Negative reward for invalid actions (optional)
        PLACEHOLDER_EVENT: -0.1    # Negative penalty for each time step
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    # Apply a penalty for each time step
    reward_sum -= len(events) * game_rewards[PLACEHOLDER_EVENT]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
