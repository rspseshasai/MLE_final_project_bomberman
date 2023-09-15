from collections import deque, namedtuple
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
TRANSITION_HISTORY_SIZE = 1000000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
BATCH_SIZE = 256
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
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
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

    # Calculate the step score based on rewards obtained from events
    step_score = reward_from_events(self, events)

    global total_score
    # Add the step score to the total score
    total_score += step_score

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
            with open("my-saved-model.pt", "wb") as file:
                pickle.dump(q_network, file)




def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Calculate the total score for this round
    round_score = total_score

    # Log and plot the total score for this round
    self.logger.info(f"Total score till now: {round_score}")

    # Reinitialize the total score for the next round

    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

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
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(q_network, file)
