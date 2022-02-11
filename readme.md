**Environment**\
BipedalWalker [Link](https://gym.openai.com/envs/BipedalWalker-v2/)

*Action Space*
1. BipedalWalker has 2 legs. 
2. Each leg has 2 joints. 
3. You have to teach the Bipedal-walker to walk by applying the torque on these joints. 
4. The size of our action space is 4 which is torque applied on 4 joints. 
5. You can apply the torque in the range of (-1, 1)

*Reward*
1. The agent gets a positive reward proportional to the distance walked on the terrain. 
2. It can get a total of 300+ reward all the way up to the end. 
3. If agent tumbles, it gets a reward of -100. 
4. There is some negative reward proportional to the torque applied on the joint. 
5. So that agent learns to walk smoothly with minimal torque.


**Reference**
1. Deep Deterministic Policy Gradient (DDPG) [Link](https://keras.io/examples/rl/ddpg_pendulum/)