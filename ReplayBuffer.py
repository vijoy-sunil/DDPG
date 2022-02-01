import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.buffer_capacity = capacity
        # sampling batch size
        self.batch_size = batch_size
        # buffer must store (state, action, reward, next_state)
        self.buffer = deque(maxlen=self.buffer_capacity)

    def get_size(self):
        return len(self.buffer)

    # left most element gets popped when size exceeds maxlen
    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample_batch(self):
        mini_batch = np.random.choice(self.buffer, self.batch_size)
        # clear lists
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []

        for index, sample in enumerate(mini_batch):
            state, action, reward, next_state = sample
            states_batch.append(state)
            actions_batch.append(action)
            rewards_batch.append(reward)
            next_states_batch.append(next_state)

        # convert python lists to numpy array
        states_batch = np.array(states_batch)
        actions_batch = np.array(actions_batch)
        rewards_batch = np.array(rewards_batch)
        next_states_batch = np.array(next_states_batch)

        return states_batch, actions_batch, rewards_batch, next_states_batch



