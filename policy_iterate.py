import gym
import time
import random
import pprint
import custom_envs

class PolicyIterate(object):
    def __init__(self, env):
        self.grid_mdp = env

        # 初始化值函数。
        self.v = {k:0 for k in range(1, 9)}

        # 初始化随机策略,为每一个状态产生随机行为。
        self.pi = {i:env.actions[random.randint(0, 3)] for i in range(1, 6)}

        # # 初始化值函数。
        # self.v = dict()
        # for i in range(1, 9):
        #     self.v[i] = 0

        # # 初始化随机策略。
        # self.pi = dict()
        # actions = env.actions
        # for s in range(1, 6):
        #     # 为每一个状态产生随机行为。
        #     direction = random.randint(0, 3)
        #     action = actions[direction]
        #     self.pi[s] = action

        # 打印初始化值函数和随机策略。
        print('初始值：')
        pprint.pprint(self.v)
        pprint.pprint(self.pi)

    def policy_iterate(self):
        """策略迭代分两部分，策略策略评估和策略改进。
        
        Arguments:
            grid_mdp {环境} -- 机器人找金币
        """

        for _ in range(100):
            self.policy_evaluate(self.grid_mdp)
            self.policy_improve(self.grid_mdp)
        
        # 返回最终策略
        return self.pi, self.v

    def policy_evaluate(self, grid_mdp):
        """第一个循环是为了保证值函数收敛到该初始随机策略所对应的真实值函数。
        第二个循环为整个状态空间的扫描，这样保证状态空间美一点的值函数都得到估计。
        
        Arguments:
            grid_mdp {环境} -- 机器人找金币
        """

        for _ in range(1000):
            delta = 0.0
            for state in grid_mdp.states:
                if state in grid_mdp.terminate_states: continue
                action = self.pi[state]
                next_s, r = grid_mdp.transform(state, action)
                new_v = r + grid_mdp.gamma * self.v[next_s]
                delta += abs(self.v[state] - new_v)
                self.v[state] = new_v
            if delta < 1e-6: break

    def policy_improve(self, grid_mdp):
        """策略改善基于单前随机策略的值函数并使用贪婪策略作为改善策略。
        外循环对整个状态空间进行遍历，内循环怼每个状态空间所对应的动作空间进行遍历，通过动作值函数得到贪婪策略。
        
        Arguments:
            grid_mdp {环境} -- 机器人找金币
        """

        for state in grid_mdp.states:
            if state in grid_mdp.terminate_states: continue
            a1 = grid_mdp.actions[0]
            next_s, r = grid_mdp.transform(state, a1)
            v1 = r + grid_mdp.gamma * self.v[next_s]
            # 贪婪策略。
            for action in grid_mdp.actions:
                s, r = grid_mdp.transform(state, action)
                v2 = r + grid_mdp.gamma * self.v[s]
                if v1 < v2:
                    a1 = action
                    v1 = v2
            self.pi[state] = a1

def main():
    env = gym.make('GridEnv-v0')
    ### 探索策略
    policy_obj = PolicyIterate(env)
    final_policy, final_value = policy_obj.policy_iterate()
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