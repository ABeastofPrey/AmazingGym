import random

def greedy(qfunc, state, actions):
    # amax = 0
    # key = "%d_%s" % (state, actions[0])
    # qmax = qfunc[key]
    # for i in range(len(actions)):  # 扫描动作空间得到最大动作值函数
    #     key = "%d_%s" % (state, actions[i])
    #     q = qfunc[key]
    #     if qmax < q:
    #         qmax, amax = q, i
    # return actions[amax]
    state_rewards = { key: value for key, value in qfunc.items() if int(key.split("_")[0]) == state }
    max_reward = max(zip(state_rewards.values(), state_rewards.keys()))
    return max_reward[1].split("_")[1]

def epsilon_greedy(qfunc, state, actions, epsilon):
    amax = 0
    key = "%d_%s"%(state, actions[0])
    qmax = qfunc[key]
    for i in range(len(actions)):    #扫描动作空间得到最大动作值函数
        key = "%d_%s"%(state, actions[i])
        q = qfunc[key]
        if qmax < q:
            qmax, amax = q, i

    #概率部分
    pro = [0.0 for i in range(len(actions))]
    pro[amax] += 1-epsilon
    for i in range(len(actions)):
        pro[i] += epsilon/len(actions)

    ##选择动作
    r = random.random()
    s = 0.0
    for i in range(len(actions)):
        s += pro[i]
        if s>= r: return actions[i]
    return actions[len(actions)-1]
