import gym
import time
import random
import pprint
import custom_envs

class ValueIterate(object):
    def __init__(self, grid_mdp):
        self.env = grid_mdp

        # 初始化值函数。
        self.v = {k:0 for k in range(1, 9)}

        # 初始化随机策略,为每一个状态产生随机行为。
        self.pi = {i:self.env.actions[random.randint(0, 3)] for i in range(1, 6)}

        # 打印初始化值函数和随机策略。
        print('初始值：')
        pprint.pprint(self.v)
        pprint.pprint(self.pi)

    def value_iterate(self):
        """值迭代需要三重循环。
        第一重循环保证值函数收敛；第二重循环用来遍历整个状态空间，对应着一次策略评估；第三重循环遍历动作空间，用来选取最优动作。
        
        Returns:
            策略和值函数 -- 最终策略和最终值函数。
        """

        env = self.env
        for _ in range(1000):
            delta = 0.0
            for state in env.states:
                if state in env.getTerminateStates(): continue
                action1 = env.actions[0]
                next_state1, reward1 = env.transform(state, action1)
                value1 = reward1 + env.gamma * self.v[next_state1]
                for action2 in env.actions:
                    next_state, reward = env.transform(state, action2)
                    value2 = reward + env.gamma * self.v[next_state]
                    if value1 < value2:
                        action1, value1 = action2, value2
                delta += abs(value1 - self.v[state])
                self.pi[state], self.v[state] = action1, value1
            if delta < 1e-6: break

        return self.pi, self.v

def main():
    env = gym.make('GridEnv-v0')
    ### 探索策略
    policy_obj = ValueIterate(env)
    final_policy, final_value = policy_obj.value_iterate()
    # 打印最终值函数和最终策略。
    print('最终值：')
    pprint.pprint(final_value)
    pprint.pprint(final_policy)

    ### 使用最终策略让机器人找金币。
    # 尝试找金币10次.
    for _ in range(10):
        env.reset()
        env.render()
        time.sleep(0.3)
        state = env.getState()
        # 判断是否为最终状态
        if state in env.getTerminateStates():
            time.sleep(1)
            continue
        # 根据最终策略采取行为
        is_not_terminal = True
        while is_not_terminal:
            action = final_policy[state]
            next_state, _, is_terminal, _ = env.step(action)
            state = next_state
            env.render()
            is_not_terminal = not is_terminal
            if is_not_terminal:
                time.sleep(0.3)
            else:
                time.sleep(1)

if __name__ == "__main__":
    main()