import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.buffer_capacity = capacity
        # sampling batch size
        self.batch_size = batch_size
        # buffer
        self.buffer = deque(maxlen=self.buffer_capacity)

    def get_size(self):
        return len(self.buffer)

    # left most element gets popped when size exceeds maxlen
    def add_experience(self, state, action, reward, next_state, done):
        # add to deque
        self.buffer.append([state,
                            action,
                            np.expand_dims(reward, -1),
                            next_state,
                            np.expand_dims(done, -1)])

    def sample_batch(self):
        indices = np.random.choice(len(self.buffer),
                                   size=min(self.batch_size, len(self.buffer)),
                                   replace=False)
        batch = [self.buffer[index] for index in indices]
        # state, action, reward, next_state, done
        return batch
