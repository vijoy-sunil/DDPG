import numpy as np
import tensorflow as tf

class ReplayBuffer:
    def __init__(self, capacity, batch_size, state_space, action_space):
        self.buffer_capacity = capacity
        self.buffer_counter = 0
        # sampling batch size
        self.batch_size = batch_size
        # buffer must store (state, action, reward, next_state)
        self.state_buffer = np.zeros((self.buffer_capacity, state_space))
        self.action_buffer = np.zeros((self.buffer_capacity, action_space))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_space))

    def get_size(self):
        return self.buffer_counter

    # left most element gets popped when size exceeds maxlen
    def add_experience(self, experience):
        state, action, reward, next_state = experience
        # Set index to zero if buffer_capacity is exceeded, replacing
        # old records
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state
        # increment counter
        self.buffer_counter += 1

    def sample_batch(self):
        batch_indices = np.random.choice(self.buffer_counter, self.batch_size)
        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        return state_batch, action_batch, reward_batch, next_state_batch



