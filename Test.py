import numpy as np
import Model
import Utils
import gym
import tensorflow as tf

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
action_lower_bound = env.action_space.low[0]
action_upper_bound = env.action_space.high[0]

model = Model.Model(num_states, num_actions, action_lower_bound, action_upper_bound)

# parameters
episodes = 10
avg_reward_lookup_episodes = 40
max_epoch_per_ep = 2000

def test(train_id, ep_id):
    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    # load weights
    model.actor = model.load_model_weights(train_id, ep_id)
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
            # get action
            tf_state = tf.expand_dims(state, 0)
            action = model.actor(tf_state)[0].numpy()
            action = np.clip(action, action_lower_bound, action_upper_bound)
            # play step
            next_state, reward, done, _ = env.step(action)

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
                                             model.replay_buffer.get_size()))

    print("testing complete")
    # plot result
    test_id = train_id + 0.1
    Utils.plot_avg_reward(test_id, avg_reward_list)
    # close env
    env.close()


if __name__ == "__main__":
    test(0, 499)
