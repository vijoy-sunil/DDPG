**Environment**
LunarLanderContinuous-v2 [Link](https://gym.openai.com/envs/LunarLanderContinuous-v2/) 
1. Landing pad is always at coordinates (0,0). 
2. Coordinates are the first two numbers in state vector. 
3. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. 
4. If lander moves away from landing pad it loses rewards.
5. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. 
6. Each leg ground contact is +10. 
7. Firing main engine is -0.3 points each frame. 
8. Solving is 200 points. 
9. Landing outside landing pad is possible. 
10. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. 
11. Action is two real values vector from -1 to +1
12. First controls main engine, -1..0 off \
    0..+1 controls throttle from 50% to 100% power. 
13. Engine can't work with less than 50% power. 
14. Second value -1.0..-0.5 fires left engine, \
    +0.5..+1.0 fire right engine, \
    -0.5..0.5 off.

**Reference**
1. Deep Deterministic Policy Gradient (DDPG) [Link](https://keras.io/examples/rl/ddpg_pendulum/)