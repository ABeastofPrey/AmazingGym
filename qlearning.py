import gym
import time
import random
import pprint
import custom_envs
from utils import greedy, epsilon_greedy

class TemporalDifference(object):
    def __init__(self, env):
        self.env = env
        self.states = env.getStates()
        self.actions = env.getActions()
        self.gamma = env.gamma

    def sarsa(self):
        pass

    def q_learning(self, num_iter, alpha, epsilon):
        # 定义行为值函数为字典，并初始化为0
        qfunc = dict()
        for s in self.states:
            for a in self.actions:
               qfunc['%d_%s'%(s, a)] = 0.0
        # 迭代探索环境
        for _ in range(num_iter):
            # 随机初始化初始状态
            state = self.states[int(random.random() * len(self.states))]
            action = self.actions[int(random.random()*len(self.actions))]

            is_terminal, count = False, 0
            while False == is_terminal and count < 100:
                policy = "%d_%s"%(state, action)
                is_terminal, next_state, reward = self.env.transform1(state, action)
                # next_state处的最大动作
                next_action = greedy(qfunc, next_state, self.actions)
                next_policy = "%d_%s"%(next_state, next_action)
                # 利用qlearning算法更新值函数
                qfunc[policy] = qfunc[policy] + alpha*(reward + self.gamma * qfunc[next_policy] - qfunc[policy])
                # 转到下一个状态
                state, action = next_state, epsilon_greedy(qfunc, next_state, self.actions, epsilon)
                count += 1
        return qfunc

    def ql_final_policy(self, qfunc):
        pi = dict()
        for state in self.states:
            if state in self.env.terminate_states: continue
            action = greedy(qfunc, state, self.actions)
            pi[state] = action
        return pi

def main():
    env = gym.make('GridEnv-v0')
    ### 探索策略
    policy_obj = TemporalDifference(env)
    qfunc = policy_obj.q_learning(num_iter=500, alpha=0.2, epsilon=0.2)
    final_policy = policy_obj.ql_final_policy(qfunc)
    # 打印最终值函数和最终策略。
    print('最终值：')
    pprint.pprint(qfunc)
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
