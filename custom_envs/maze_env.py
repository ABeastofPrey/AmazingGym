import gym
import random
import numpy as np

class MazeEnv(gym.Env):
    # 定义窗口的宽和高。
    __window_width, __window_height = 550, 600

    def __init__(self):
        self.viewer = None  # 窗口。
        self.state = None   # 初始状态。

        # 描述马尔可夫决策过程, 元祖(S, P, A, R, gamma)
        self.__MDP()

    def __MDP(self):
        # 创建状态空间States
        self.__create_states()
        # 创建行动集合Actions
        self.__create_actions()
        # 创建状态转移概论矩阵P
        self.__create_state_transition()
        # 创建回报数据Rewards
        self.__create_rewards()
        # 折扣因子gamma
        self.gamma = 0.8

    def __create_states(self):
        # 状态集合    
        '''    
        [[ 1  2  3  4  5]
         [ 6  7  8  9 10]
         [11 12 13 14 15]
         [16 17 18 19 20]
         [21 22 23 24 25]]
        '''
        self.states = np.asarray(np.arange(1, 26).reshape(5, 5))

        # 终止状态集合
        self.terminal_states = dict()
        self.terminal_states[4] = 1
        self.terminal_states[9] = 1
        self.terminal_states[11] = 1
        self.terminal_states[12] = 1
        self.terminal_states[15] = 1    # 出口
        self.terminal_states[23] = 1
        self.terminal_states[24] = 1
        self.terminal_states[25] = 1

    def __create_actions(self):
        self.actions = ['north', 'east', 'south', 'west']   # 上，右，下，左

    def __create_state_transition(self):
        self.transition = dict()
        # state 1
        self.transition['1_east'] = 2
        self.transition['1_south'] = 6
        # state 2
        self.transition['2_east'] = 3
        self.transition['2_south'] = 7
        self.transition['2_west'] = 1
        # state 3
        self.transition['3_south'] = 8
        self.transition['3_west'] = 2
        # state 5
        self.transition['5_south'] = 10
        # state 6
        self.transition['6_north'] = 1
        self.transition['6_east'] = 7
        # state 7
        self.transition['7_north'] = 2
        self.transition['7_east'] = 8
        self.transition['7_west'] = 6
        # state 8
        self.transition['8_north'] = 3
        self.transition['8_south'] = 13
        self.transition['8_west'] = 7
        # state 10
        self.transition['10_north'] = 5
        self.transition['10_south'] = 15
        # state 13
        self.transition['13_north'] = 8
        self.transition['13_east'] = 14
        self.transition['13_south'] = 18
        # state 14
        self.transition['14_east'] = 15
        self.transition['14_south'] = 19
        self.transition['14_west'] = 13
        # state 16
        self.transition['16_east'] = 17
        self.transition['16_south'] = 21
        # state 17
        self.transition['17_east'] = 18
        self.transition['17_south'] = 22
        self.transition['17_west'] = 16
        # state 18
        self.transition['18_north'] = 13
        self.transition['18_east'] = 19
        self.transition['18_west'] = 17
        # state 19
        self.transition['19_north'] = 14
        self.transition['19_east'] = 20
        self.transition['19_west'] = 18
        # state 20
        self.transition['20_north'] = 15
        self.transition['20_west'] = 19
        # state 21
        self.transition['21_north'] = 16
        self.transition['21_east'] = 22
        # state 22
        self.transition['22_north'] = 17
        self.transition['22_west'] = 21

    def __create_rewards(self):
        self.rewards = dict()
        # state 3
        self.rewards['3_east'] = -1
        # state 5
        self.rewards['5_west'] = -1
        # state 6
        self.rewards['6_south'] = -1
        # state 7
        self.rewards['7_south'] = -1
        # state 8
        self.rewards['8_east'] = -1
        # state 10
        self.rewards['10_south'] = 1
        self.rewards['10_west'] = -1
        # state 13
        self.rewards['13_west'] = -1
        # state 14
        self.rewards['14_north'] = -1
        self.rewards['14_east'] = 1
        # state 16
        self.rewards['16_north'] = -1
        # state 17
        self.rewards['17_north'] = -1
        # state 18
        self.rewards['18_south'] = -1
        # state 19
        self.rewards['19_south'] = -1
        # state 20
        self.rewards['20_north'] = 1
        self.rewards['20_south'] = -1
        # state 22
        self.rewards['22_east'] = -1


    def __create_window(self):
        from gym.envs.classic_control import rendering
        self.viewer = rendering.Viewer(self.__window_width, self.__window_height)
        
        # 创建网格世界Environment
        # 创建网格X轴
        for i in range(200, 500, 50):
            line_x = rendering.Line((150,i),(400,i))
            self.viewer.add_geom(line_x)
        # 创建网格Y轴
        for j in range(150, 450, 50):
            line_y = rendering.Line((j, 200), (j, 450))
            self.viewer.add_geom(line_y)

        # 创建陷阱
        # 创建第一组陷阱（状态11，12）
        l, r, t, b = 150, 250, 350, 300
        trap = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        trap.set_color(1, 0, 0)
        self.viewer.add_geom(trap)
        # 创建第二组陷阱（状态4，9）
        l, r, t, b = 300, 350, 450, 350
        trap = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        trap.set_color(1, 0, 0)
        self.viewer.add_geom(trap)
        # 创建第三组陷阱（状态23，24，25）
        l, r, t, b = 250, 400, 200, 250
        trap = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        trap.set_color(1, 0, 0)
        self.viewer.add_geom(trap)
        
        # 创建出口（状态15）
        l, r, t, b = 350, 400, 350, 300
        exit = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        exit.set_color(1, 0.9, 0)
        self.viewer.add_geom(exit)

        # 创建机器人Agent
        self.robot= rendering.make_circle(25)
        self.robotranslator = rendering.Transform()
        self.robot.add_attr(self.robotranslator)
        self.robot.set_color(0, 0, 1)
        self.viewer.add_geom(self.robot)

        # 保存网格信息
        self.cell_x = [175, 225, 275, 325, 375]
        self.cell_y = [425, 375, 325, 275, 225]
    
    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.__create_window()

        if self.state is not None:
            cell_axis = np.argwhere(self.states == self.state)
            x, y = cell_axis[0, 0], cell_axis[0, 1]
            self.robotranslator.set_translation(self.cell_x[y], self.cell_y[x])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def reset(self):
        random_x = random.randint(0, 4)
        random_y = random.randint(0, 4)
        self.state = self.states[random_x, random_y]
        return self.state

    def step(self, action):
        if self.state is None: return None
        if action not in self.actions: return None
        # 获取当前状态
        state = self.state
        is_terminal = False
        reward = 0

        # 当前状态是否为终止状态
        if state in self.terminal_states:
            is_terminal = True
            return state, reward, is_terminal, {}   # 返回状态，回报，是否终止和调试信息
        
        # 采取action实现状态转移
        transition_key = "%d_%s"%(state, action)
        if transition_key in self.transition:
            next_state = self.transition[transition_key]
        else:
            next_state = state
        # 判断下一个状态是否为终止状态
        if next_state in self.terminal_states:
            is_terminal = True
        # 判断是否有奖励
        if transition_key in self.rewards:
            reward = self.rewards[transition_key]

        # 保存下一个状态为当前状态
        self.state = next_state

        return next_state, reward, is_terminal, {}   # 返回状态，回报，是否终止和调试信息
    
    def set_state(self, state):
        self.state = state
        return self.state

    def close(self):
        if self.viewer is None: return
        self.viewer.close()
        self.viewer = None


if __name__ == '__main__':
    import time

    env = MazeEnv()
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
    '''
    Just run python3 maze_env.py command to test the MazeEnv environment.
    '''