import numpy as np
import matplotlib.pyplot as plt
import Model
import Noise
import gym
import tensorflow as tf

problem = "Pendulum-v1"
env = gym.make(problem)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]

model = Model.Model(num_states, num_actions, upper_bound)
noise = Noise.OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))

# parameters
episodes = 100
max_epochs_per_episode = 150
avg_reward_lookup_episodes = 40
ep_save_checkpoint = 100

def train(train_id):
    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    for e in range(episodes):
        # get current state, the process gets started by calling
        # reset(), which returns an initial observation
        state = env.reset()
        # clear total reward accumulated per episode
        ep_reward = 0
        # clear done flag
        done = False
        # clear epoch count
        epoch = 0
        # start episode
        while done is not True:
            env.render()
            # reshape input state
            # same as state = state.reshape(1, num_states)
            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            # get action (+ noise)
            action = model.actor(tf_state) + noise()
            # play step
            next_state, reward, done, _ = env.step(action[0])
            ep_reward += reward

            # save to replay buffer
            model.replay_buffer.add_experience((state, action, reward, next_state))
            # train
            if model.replay_buffer.get_size() > model.batch_size:
                model.train()
                model.update_target_weights()

            # update state
            state = next_state
            epoch += 1

        # episode is complete
        ep_reward_list.append(ep_reward)
        avg_reward = np.mean(ep_reward_list[-avg_reward_lookup_episodes:])
        avg_reward_list.append(avg_reward)
        print("episode {} complete, epochs {}, reward {}, avg reward in last {} episodes {}"\
              .format(e, epoch, ep_reward, avg_reward_lookup_episodes, avg_reward))
        # plot result
        plot_avg_reward(train_id, avg_reward_list)
        # save model weights every x episodes
        if e % ep_save_checkpoint == 0:
            model.save_model_weights(train_id, e)

def plot_avg_reward(train_id, avg_reward_list):
    plt.plot(avg_reward_list)
    plt.xlabel("episode")
    plt.ylabel("avg reward")
    # save fig
    fig_name = str(train_id) + '.png'
    fig_path = 'Log/' + fig_name
    plt.savefig(fig_path)
    plt.show()


if __name__ == "__main__":
    train(0)
