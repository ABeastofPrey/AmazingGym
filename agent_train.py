import gym
import time
from agent_brain import Brain

# env definition
env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

# brain definition
Brain = Brain(
    actions=env.action_space.n,
    features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
)

# print("*********************")
# print(env.action_space)
# print(env.action_space.n)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.observation_space.shape[0])
# print("*********************")

def traing(count=85):
    for episode in range(count):
        observation = env.reset()
        while True:
            # exploration of environment
            action = Brain.random_action(observation)
            next_observation, reward, done, info = env.step(action)
            #将观测，动作和回报存储起来
            Brain.store_transition(observation, action, reward)
            if done: 
                print(episode)
                #每个episode学习一次
                Brain.learn()
                break
            observation = next_observation

def testing():
    observation = env.reset()
    while True:
        env.render()
        action = Brain.greedy_action(observation)
        next_observation, reward, done, info = env.step(action)
        if done: 
            # observation = env.reset()
            # Brain.close()
            break
        observation = next_observation

def main():
    traing()
    testing()

if __name__ == "__main__":
    main()