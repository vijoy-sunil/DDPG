import ReplayBuffer
import tensorflow as tf
from tensorflow.keras import layers
from keras.optimizers import adam_v2
from keras.models import load_model

class Model:
    def __init__(self, state_space, action_space, upper_bound_action):
        # parameters
        self.action_bound = upper_bound_action
        self.state_space = state_space
        self.action_space = action_space
        print("state space {}, action space {}, action upper bound {}"
              .format(state_space, action_space, upper_bound_action))
        # learning rates
        self.actor_lr = 0.00005
        self.critic_lr = 0.0005
        # discount factor for future rewards
        self.gamma = 0.99
        # rate of update for target_ networks
        self.tau = 0.001
        # networks
        self.actor = self.get_actor()
        self.critic = self.get_critic()
        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()
        # optimizers
        self.actor_optimizer = adam_v2.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = adam_v2.Adam(learning_rate=self.critic_lr)
        # Making the weights equal initially
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        # replay buffer
        self.replay_buffer_capacity = 100000
        self.batch_size = 100
        self.replay_buffer = ReplayBuffer.ReplayBuffer(self.replay_buffer_capacity,
                                                       self.batch_size,
                                                       self.state_space,
                                                       self.action_space)
        # paths
        self.weights_dir = 'Weights/'

    # network architecture
    def get_actor(self):
        input_0 = layers.Input(shape=(self.state_space,))
        hidden_0 = layers.Dense(600)(input_0)
        bn_0 = layers.BatchNormalization()(hidden_0)
        ac_0 = layers.Activation('relu')(bn_0)

        w_init_0 = tf.random_uniform_initializer(minval=-0.0015, maxval=0.0015)
        hidden_1 = layers.Dense(300, kernel_initializer=w_init_0)(ac_0)
        bn_1 = layers.BatchNormalization()(hidden_1)
        ac_1 = layers.Activation('relu')(bn_1)

        w_init_1 = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        output_0 = layers.Dense(self.action_space, activation="tanh",
                                kernel_initializer=w_init_1)(ac_1)
        # tanh activation function
        # The function takes any real value as input and outputs values
        # in the range -1 to 1. The larger the input (more positive), the
        # closer the output value will be to 1.0, whereas the smaller the
        # input (more negative), the closer the output will be to -1.0.
        # Scale output to -action_bound to action_bound
        output_0 = output_0 * self.action_bound
        model = tf.keras.Model(input_0, output_0)
        return model

    def get_critic(self):
        # State as input
        input_0 = layers.Input(shape=(self.state_space,))
        hidden_0 = layers.Dense(600)(input_0)
        bn_0 = layers.BatchNormalization()(hidden_0)
        ac_0 = layers.Activation('relu')(bn_0)
        w_init_0 = tf.random_uniform_initializer(minval=-0.0015, maxval=0.0015)
        hidden_1 = layers.Dense(300, kernel_initializer=w_init_0)(ac_0)
        # Action as input
        input_1 = layers.Input(shape=(self.action_space,))
        hidden_2 = layers.Dense(300,)(input_1)
        # Concat
        concat = layers.Concatenate()([hidden_1, hidden_2])
        ac_1 = layers.Activation('relu')(concat)
        w_init_1 = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        output_0 = layers.Dense(1, kernel_initializer=w_init_1)(ac_1)
        # Outputs single value for give state-action
        model = tf.keras.Model([input_0, input_1], output_0)
        return model

    @tf.function
    def train(self):
        # get batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample_batch()
        # first, train critic model
        ones = tf.ones((self.batch_size, 1))
        with tf.GradientTape() as tape:
            # Some neural network layers behave differently during training
            # and inference, for example Dropout and BatchNormalization layers.
            # For example during training, dropout will randomly drop out units
            # and correspondingly scale up activations of the remaining units.
            # During inference, it does nothing (since you usually don't want the
            # randomness of dropping out units here). The training argument lets the
            # layer know which of the two "paths" it should take. If you set this
            # incorrectly, your network might not behave as expected.
            target_actions = self.target_actor(next_state_batch, training=True)
            q_values = self.target_critic([next_state_batch, target_actions], training=True)
            # bellman equation
            y = reward_batch + (self.gamma * q_values * (ones - done_batch))

            critic_value = self.critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables))

        # second, train actor model
        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value = self.critic([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given by the
            # critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables))

    # Instead of updating the target network periodically and all at once,
    # we will be updating it frequently, but slowly.
    def update_target_weights(self):
        # actor -> target_actor
        source_w, target_w = self.actor.get_weights(), self.target_actor.get_weights()
        for i in range(len(source_w)):
            target_w[i] = self.tau * source_w[i] + (1 - self.tau) * target_w[i]
        self.target_actor.set_weights(target_w)

        # critic -> target_critic
        source_w, target_w = self.critic.get_weights(), self.target_critic.get_weights()
        for i in range(len(source_w)):
            target_w[i] = self.tau * source_w[i] + (1 - self.tau) * target_w[i]
        self.target_critic.set_weights(target_w)

    # load and save model weights, NOTE: we are not saving model
    # architecture here
    def load_model_weights(self, train_id, ep_id):
        weights_file = self.weights_dir + 'target_actor_' + str(train_id) + '_' + str(ep_id) + '.h5'
        weights = load_model(weights_file)
        return weights

    def continue_training(self, train_id, ep_id):
        # load actor
        weights_file = self.weights_dir + 'actor_' + str(train_id) + '_' + str(ep_id) + '.h5'
        self.actor = load_model(weights_file)
        # load critic
        weights_file = self.weights_dir + 'critic_' + str(train_id) + '_' + str(ep_id) + '.h5'
        self.critic = load_model(weights_file)
        # load target_actor
        weights_file = self.weights_dir + 'target_actor_' + str(train_id) + '_' + str(ep_id) + '.h5'
        self.target_actor = load_model(weights_file)
        # load target_critic
        weights_file = self.weights_dir + 'target_critic_' + str(train_id) + '_' + str(ep_id) + '.h5'
        self.target_critic = load_model(weights_file)

    def save_model_weights(self, train_id, ep_id):
        weights_file = self.weights_dir + 'actor_' + str(train_id) + '_' + str(ep_id) + '.h5'
        self.actor.save(weights_file)
        weights_file = self.weights_dir + 'target_actor_' + str(train_id) + '_' + str(ep_id) + '.h5'
        self.target_actor.save(weights_file)
        weights_file = self.weights_dir + 'critic_' + str(train_id) + '_' + str(ep_id) + '.h5'
        self.critic.save(weights_file)
        weights_file = self.weights_dir + 'target_critic_' + str(train_id) + '_' + str(ep_id) + '.h5'
        self.target_critic.save(weights_file)
