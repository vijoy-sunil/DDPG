import numpy as np
import matplotlib.pyplot as plt
import Model
import Noise
import gym
import tensorflow as tf
import os
import shutil

# environment description in readme
# state space: 24
# State consists of hull angle speed, angular velocity, horizontal speed,
# vertical speed, position of joints and joints angular speed, legs contact
# with ground, and 10 lidar rangefinder measurements. There's no coordinates
# in the state vector.
# action space: 4 | upper bound 1, lower bound -1
problem = "BipedalWalker-v3"
env = gym.make(problem)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]

model = Model.Model(num_states, num_actions, upper_bound)
noise = Noise.OUActionNoise(mean=np.zeros(num_actions))

# parameters
episodes = 500
avg_reward_lookup_episodes = 40
ep_save_checkpoint = 100
max_epoch_per_ep = 2000

def train(train_id):
    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    # continue training - loads last saved weights
    # model.continue_training(2, 499)
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

            # save to replay buffer
            model.replay_buffer.add_experience((state, action, reward, next_state, done))
            # train
            if model.replay_buffer.get_size() > model.batch_size:
                model.train()
                model.update_target_weights()

            # update
            state = next_state
            ep_reward += reward
            epoch += 1

            # end episode when epochs exceed limit
            if epoch > max_epoch_per_ep:
                print('epoch limit exceeded')
                break

        # episode is complete
        ep_reward_list.append(ep_reward)
        avg_reward = np.mean(ep_reward_list[-avg_reward_lookup_episodes:])
        avg_reward_list.append(avg_reward)
        print("episode {} complete, epochs {}, reward {}, "
              "avg reward in last {} episodes {}, "
              "replay buffer size {}".format(e, epoch, ep_reward,
                                             avg_reward_lookup_episodes, avg_reward,
                                             model.replay_buffer.buffer_counter))

        # save model weights every x episodes
        if e % ep_save_checkpoint == 0 and e != 0:
            print("saving checkpoint weights, episode {}".format(e))
            model.save_model_weights(train_id, e)

    print("training complete")
    # save final episode weights
    model.save_model_weights(train_id, episodes-1)
    # plot result
    plot_avg_reward(train_id, avg_reward_list)
    # close env
    env.close()

def plot_avg_reward(train_id, avg_reward_list):
    plt.plot(avg_reward_list)
    plt.xlabel("episode")
    plt.ylabel("avg reward")
    # save fig
    fig_name = str(train_id) + '.png'
    fig_path = 'Log/' + fig_name
    plt.savefig(fig_path)
    plt.show()

# clear previous saved outputs
def clear_history(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == "__main__":
    clear_history('Weights/')
    # clear_history('Log/')
    train(0)
