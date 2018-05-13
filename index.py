import gym
import time
import custom_envs

# env = gym.make('GridEnv-v0')
env = gym.make('MazeEnv-v0')

for i in range(1, 25):
    t = 0.2
    state = env.set_state(i)
    env.render()
    time.sleep(t)
    env.step('north')
    env.render()
    time.sleep(t)

    env.set_state(state)
    env.render()
    time.sleep(t)
    env.step('east')
    env.render()
    time.sleep(t)

    env.set_state(state)
    env.render()
    time.sleep(t)
    env.step('south')
    env.render()
    time.sleep(t)

    env.set_state(state)
    env.render()
    time.sleep(t)
    env.step('west')
    env.render()
    time.sleep(t)
    env.set_state(state)
    env.render()
    time.sleep(t)

time.sleep(60)