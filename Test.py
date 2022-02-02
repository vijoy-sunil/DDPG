import numpy as np
import Train
import Model
import gym
import tensorflow as tf

problem = "LunarLanderContinuous-v2"
env = gym.make(problem)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]

model = Model.Model(num_states, num_actions, upper_bound)

# parameters
episodes = 25
avg_reward_lookup_episodes = 40

def test(train_id):
    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    # load model weights for target actor
    model.target_actor = model.load_model_weights(train_id, 500)
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
            # get action
            action = model.target_actor(tf_state)
            # play step
            next_state, reward, done, _ = env.step(action[0])
            # update
            state = next_state
            ep_reward += reward
            epoch += 1

        # episode is complete
        ep_reward_list.append(ep_reward)
        avg_reward = np.mean(ep_reward_list[-avg_reward_lookup_episodes:])
        avg_reward_list.append(avg_reward)
        print("episode {} complete, epochs {}, reward {}, avg reward in last {} episodes {}"\
              .format(e, epoch, ep_reward, avg_reward_lookup_episodes, avg_reward))

    print("testing complete")
    # plot result, with different name
    test_id = train_id + 0.1
    Train.plot_avg_reward(test_id, avg_reward_list)


if __name__ == "__main__":
    # test with train_id
    test(0)
