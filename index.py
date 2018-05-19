import gym
import time
import custom_envs

env = gym.make('GridEnv-v0')
# env = gym.make('MazeEnv-v0')

env.reset()
env.render()
for i in range(9):
    # env.setState(i)
    env.render()
    time.sleep(0.5)
time.sleep(3)