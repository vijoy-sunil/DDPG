import ReplayBuffer
import Noise
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import glorot_normal
from keras.models import load_model


class Model:
    def __init__(self, state_space, action_space, action_lower_bound, action_upper_bound):
        # parameters
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound = action_upper_bound
        self.state_space = state_space
        self.action_space = action_space
        print("state space {}, action space {}, action lower bound {}, action upper bound {}"
              .format(state_space, action_space, action_lower_bound, action_upper_bound))
        # learning rates
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        # discount factor for future rewards
        self.gamma = 0.99
        # rate of update for target_ networks
        self.tau = 0.001
        # exploration
        self.epsilon = 0.2
        # networks
        self.actor = self.get_actor()
        self.critic = self.get_critic()
        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()
        # optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        # Making the weights equal initially
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        # replay buffer
        self.replay_buffer_capacity = 100000
        self.batch_size = 200
        self.replay_buffer = ReplayBuffer.ReplayBuffer(self.replay_buffer_capacity,
                                                       self.batch_size)
        # actor noise
        self.noise = Noise.OUActionNoise(mean=np.zeros(1),
                                         std_deviation=float(0.2) * np.ones(1))
        # paths
        self.weights_dir = 'Weights/'

    # network architecture
    def get_actor(self):
        input_0 = layers.Input(shape=(self.state_space,), dtype=tf.float32)
        hidden_0 = layers.Dense(600, activation=tf.nn.leaky_relu,
                                kernel_initializer=glorot_normal())(input_0)
        hidden_1 = layers.Dense(300, activation=tf.nn.leaky_relu,
                                kernel_initializer=glorot_normal())(hidden_0)
        # tanh activation function
        # The function takes any real value as input and outputs values
        # in the range -1 to 1. The larger the input (more positive), the
        # closer the output value will be to 1.0, whereas the smaller the
        # input (more negative), the closer the output will be to -1.0.
        # Scale output to -action_bound to action_bound

        # Initialize weights
        last_init = tf.random_normal_initializer(stddev=0.0005)
        output_0 = layers.Dense(self.action_space, activation='tanh',
                                kernel_initializer=last_init)(hidden_1)
        output_0 = output_0 * self.action_upper_bound
        model = tf.keras.Model(input_0, output_0)
        return model

    def get_critic(self):
        # State as input
        input_0 = layers.Input(shape=(self.state_space,), dtype=tf.float32)

        hidden_0 = layers.Dense(600, activation=tf.nn.leaky_relu,
                                kernel_initializer=glorot_normal())(input_0)
        hidden_0 = layers.BatchNormalization()(hidden_0)

        hidden_1 = layers.Dense(300, activation=tf.nn.leaky_relu,
                                kernel_initializer=glorot_normal())(hidden_0)

        # Action as input
        input_1 = layers.Input(shape=(self.action_space,), dtype=tf.float32)

        hidden_2 = layers.Dense(300, activation=tf.nn.leaky_relu,
                                kernel_initializer=glorot_normal())(input_1)

        # concat
        added = layers.Add()([hidden_1, hidden_2])
        added = layers.BatchNormalization()(added)

        hidden_3 = layers.Dense(150, activation=tf.nn.leaky_relu,
                                kernel_initializer=glorot_normal())(added)
        hidden_3 = layers.BatchNormalization()(hidden_3)

        last_init = tf.random_normal_initializer(stddev=0.00005)
        output_0 = layers.Dense(1, kernel_initializer=last_init)(hidden_3)
        # Outputs single value for give state-action
        model = tf.keras.Model([input_0, input_1], output_0)
        return model

    # get current action from current state
    def act(self, state):
        tf_state = tf.expand_dims(state, 0)
        # exploration vs exploitation
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.uniform(self.action_lower_bound,
                                       self.action_upper_bound,
                                       self.action_space) + self.noise()
        else:
            action = self.actor(tf_state)[0].numpy()

        # clip to range
        action = np.clip(action, self.action_lower_bound, self.action_upper_bound)
        return action

    @tf.function
    def train(self, s, a, r, ns, dn):
        # first, train critic model
        with tf.GradientTape() as tape:
            y = r + self.gamma * (1 - dn) * self.target_critic([ns, self.target_actor(ns)])
            critic_value = self.critic([s, a])
            critic_loss = tf.math.reduce_mean(tf.math.abs(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables))

        # second, train actor model
        with tf.GradientTape() as tape:
            actor_loss = -tf.math.reduce_mean(self.critic([s, self.actor(s)]))

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables))

    def learn(self, entry):
        s, a, r, ns, dn = zip(*entry)
        self.train(tf.convert_to_tensor(s, dtype=tf.float32),
                   tf.convert_to_tensor(a, dtype=tf.float32),
                   tf.convert_to_tensor(r, dtype=tf.float32),
                   tf.convert_to_tensor(ns, dtype=tf.float32),
                   tf.convert_to_tensor(dn, dtype=tf.float32))
        # update target weights
        self.update_target_weights(self.target_actor, self.actor)
        self.update_target_weights(self.target_critic, self.critic)

    # Instead of updating the target network periodically and all at once,
    # we will be updating it frequently, but slowly.
    def update_target_weights(self, model_target, model_ref):
        model_target.set_weights([self.tau * ref_weight + (1 - self.tau) * target_weight
                                  for (target_weight, ref_weight) in
                                  list(zip(model_target.get_weights(),
                                           model_ref.get_weights()))])

    # load and save model weights, NOTE: we are not saving model
    # architecture here
    def load_model_weights(self, train_id, ep_id):
        weights_file = self.weights_dir + 'actor_' + str(train_id) + '_' + str(ep_id) + '.h5'
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
