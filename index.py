import gym
import time
import custom_envs

env = gym.make('MazeEnv-v0')
env.reset()
env.render()
for i in range(9):
    env.render()
    time.sleep(0.5)
time.sleep(60)