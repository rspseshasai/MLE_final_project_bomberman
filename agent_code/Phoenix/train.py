from collections import namedtuple, deque
import pickle
from typing import List
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from .callbacks import state_to_features, ACTIONS
from .model import QNetwork  # Import your QNetwork model
import events as e

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
TRANSITION_HISTORY_SIZE = 10  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000

# Initialize Q-network and optimizer
input_size = 1734  # Define the input size based on your state representation
hidden_size = 256  # Define the hidden layer size
output_size = len(ACTIONS)  # The number of possible actions
q_network = QNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(q_network.parameters(), lr=0.001)

# Initialize experience replay buffer
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.discount_factor = 0.95  # Define your discount factor


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage certain behavior.
    """
    game_rewards = {
        e.MOVED_RIGHT: 2,
        e.MOVED_LEFT: 2,
        e.MOVED_UP: 2,
        e.MOVED_DOWN: 2,
        e.COIN_COLLECTED: 10,
        e.INVALID_ACTION: -5,
        PLACEHOLDER_EVENT: -0.1,  # Custom event is bad
        e.WAITED: -5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


# Update game_events_occurred
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    # if ...:  # Implement your own event-based rewards/penalties
    #     events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                   reward_from_events(self, events)))

    # Update experience replay buffer
    if old_game_state is not None:
        state = state_to_features(old_game_state)
        next_state = state_to_features(new_game_state)
        reward = reward_from_events(self, events)
        action = ACTIONS.index(self_action)
        transition = Transition(state, action, next_state, reward)
        replay_buffer.append(transition)


# Update end_of_round
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Perform Q-learning with experience replay
    if len(replay_buffer) >= BATCH_SIZE:
        batch = np.random.choice(len(replay_buffer), BATCH_SIZE, replace=False)
        batch_transitions = [replay_buffer[i] for i in batch]
        # Assuming batch_transitions is a list of numpy arrays
        batch_state_array = np.array([transition.state for transition in batch_transitions], dtype=np.float32)
        batch_state = torch.tensor(batch_state_array)
        batch_action = torch.tensor([transition.action for transition in batch_transitions], dtype=torch.int64)
        batch_reward = torch.tensor([transition.reward for transition in batch_transitions], dtype=torch.float32)

        batch_next_state_array = np.array([transition.next_state for transition in batch_transitions],
                                          dtype=np.float32)
        batch_next_state = torch.tensor(batch_next_state_array)

        # Compute Q-values for current and next states using your Q-network
        q_values = q_network(batch_state)
        q_values_next = q_network(batch_next_state)

        # Compute target Q-values
        target_q_values = q_values.clone()
        target_q_values[range(BATCH_SIZE), batch_action] = batch_reward + self.discount_factor * \
                                                           torch.max(q_values_next, dim=1)[0]

        # Compute the loss
        loss = F.smooth_l1_loss(q_values, target_q_values.detach())

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save the model after training if needed
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(q_network, file)
