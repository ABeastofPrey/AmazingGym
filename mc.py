import gym
import time
import random
import custom_envs
import pprint

class MCOBJ(object):
    def __init__(self, grid_mdp):
        self.env = grid_mdp
        # Just for pass the error.
        self.states = self.env.getStates()
        self.actions = self.env.getActions()

    def gen_random_pi_sample(self, num):
        """蒙特卡罗样本采集
        
        Arguments:
            num {number} -- 次数
        
        Returns:
            array -- 状态集，行为集，回报集
        """

        state_sample = []
        action_sample = []
        reward_sample = []
        for _ in range(num):
            s_tmp = []
            a_tmp = []
            r_tmp = []
            s = self.states[int(random.random() * len(self.states))]
            t = False
            while t == False:
                a = self.actions[int(random.random() * len(self.actions))]
                t, s1, r = self.env.transform1(s, a)
                s_tmp.append(s)
                r_tmp.append(r)
                a_tmp.append(a)
                s = s1
            # 样本包含多个状态序列。
            state_sample.append(s_tmp)
            action_sample.append(a_tmp)
            reward_sample.append(r_tmp)
        return state_sample, action_sample, reward_sample

    def mc_evaluation(self, state_sample, action_sample, reward_sample, gamma=0.8):
        vfunc, nfunc = dict(), dict()
        for s in self.states:
            vfunc[s] = 0.0
            nfunc[s] = 0.0
        for iter1 in range(len(state_sample)):
            G = 0.0
            # 逆向计算初始状态的累积回报: s1 -> s2 -> s3 -> s7
            for step in range(len(state_sample[iter1])-1, -1, -1): 
                G *= gamma
                G += reward_sample[iter1][step]
            # 正向计算每个状态处的累计回报
            for step in range(len(state_sample[iter1])):
                s = state_sample[iter1][step]
                vfunc[s] += G
                nfunc[s] += 1.0
                G -= reward_sample[iter1][step]
                G /= gamma
        # 在每个状态处求经验平均
        for s in self.states:
            if nfunc[s] > 0.000001:
                vfunc[s] /= nfunc[s]
        return vfunc

    def mc(self, num_iter1, epsilon, gamma):
        # x, y = [], []
        qfunc, n = dict(), dict()
        for s in self.states:
            for a in self.actions:
                qfunc["%d_%s"%(s, a)] = 0.0
                n["%d_%s"%(s, a)] = 0.001
        for _ in range(num_iter1):
            # x.append(iter1)
            # y.append(compute_error(qfunc))
            s_sample, a_sample, r_sample = [], [], []
            s = self.states[int(random.random() * len(self.states))]
            t = False
            count = 0
            while False == t and count < 100:
                a = self.epsilon_greedy(qfunc, s, epsilon)
                t, s1, r = self.env.transform1(s, a)
                s_sample.append(s)
                a_sample.append(a)
                r_sample.append(r)
                s = s1
                count += 1
            g = 0.0
            for i in range(len(s_sample)-1, -1, -1):
                g *= gamma
                g += r_sample[i]
            for i in range(len(s_sample)):
                key = "%d_%s"%(s_sample[i], a_sample[i])
                n[key] += 1.0
                qfunc[key] = (qfunc[key] * (n[key] - 1) + g) / n[key]
                g -= r_sample[i]
                g /= gamma
        return qfunc

    def epsilon_greedy(self, qfunc, state, epsilon):
        if random.random() > epsilon:
            return self.__max_action(qfunc, state)
        else:
            return self.actions[int(random.random() * len(self.actions))]

    def __max_action(self, qfunc, state):
        state_rewards = { key: value for key, value in qfunc.items() if int(key.split("_")[0]) == state }
        max_reward = max(zip(state_rewards.values(), state_rewards.keys()))
        return max_reward[1].split("_")[1]


def main():
    env = gym.make('GridEnv-v0')
    gamma = env.gamma
    mc_obj = MCOBJ(env)
    ### 探索策略

    ### method1
    # state_sample, action_sample, reward_sample = mc_obj.gen_random_pi_sample(10)
    # print(state_sample)
    # print(action_sample)
    # print(reward_sample)
    # vfunc = mc_obj.mc_evaluation(state_sample, action_sample, reward_sample, gamma)
    # print('mc evaluation: ')
    # print(vfunc)

    ### method2
    qfunc = mc_obj.mc(10, 0.5, gamma)
    # 打印最终行为值函数。
    pprint.pprint(qfunc)
    ### 使用最终行为值函数让机器人找金币。
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
        # 根据最终行为值函数采取行为
        is_not_terminal = True
        while is_not_terminal:
            action = qfunc[state]
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