from collections import namedtuple

from torch import optim

import events as e
from .callbacks import state_to_features
from .model import RNNLearningAgent


def setup_training(self):
    """
    Initialize training-specific variables and hyperparameters.
    """
    # Hyperparameters
    self.DISCOUNT_FACTOR = 0.99
    self.LEARNING_RATE = 0.001
    self.BATCH_SIZE = 32
    self.EPISODES = 10000
    self.EPSILON_START = 0.7
    self.EPSILON_END = 0.2
    self.EPSILON_DECAY = 0.995

    # Initialize other training variables
    self.training_data = []
    self.rnn_agent = RNNLearningAgent(self.input_dim, 10000, self.BATCH_SIZE)
    # Define an optimizer for the model
    self.optimizer = optim.Adam(self.rnn_model.parameters(), lr=0.001)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    """
    Collect training data and update the agent's model here.

    :param old_game_state: The state before the agent's action.
    :param self_action: The agent's chosen action.
    :param new_game_state: The state after the agent's action and all events.
    :param events: A list of events that occurred due to agent's and opponents' actions.
    """
    if old_game_state is not None:
        # Calculate reward from game events
        reward = reward_from_events(self, events)

        # Convert game states to feature tensors
        state = state_to_features(old_game_state)
        next_state = state_to_features(new_game_state) if new_game_state is not None else None

        # Append data to training data
        self.training_data.append((state, self_action, reward, next_state))


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """
    Called once per agent after the last step of a round.

    :param last_game_state: The final game state of the round.
    :param last_action: The last action the agent took in the round.
    :param events: A list of events that occurred during the round.
    """
    # Calculate reward from game events
    reward = reward_from_events(self, events)

    # Convert game state to feature tensor
    state = state_to_features(last_game_state)

    # Append data to training data
    self.training_data.append((state, last_action, reward, None))

    # Perform model training using the collected training data
    self.rnn_agent.train_model()
    # After training is complete, save the model to a file
    self.rnn_agent.save_rnn_model("my_model.pt")


def reward_from_events(self, events: list) -> float:
    """
    Calculate the agent's reward based on game events.

    :param events: A list of events that occurred during an agent's action.
    :return: The calculated reward.
    """
    # Define rewards for specific events
    game_rewards = {
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.KILLED_SELF: -400,
        e.GOT_KILLED: -500,
        e.COIN_COLLECTED: 100,
        e.BOMB_DROPPED: -1,
        e.WAITED: -1,
        e.KILLED_OPPONENT: 500,
        e.INVALID_ACTION: -10,
    }

    # Calculate the reward as the sum of event-specific rewards
    reward = sum(game_rewards.get(event, 0.0) for event in events)

    return reward
