import gym

class MazeEnv(gym.Env):
    # 定义窗口的宽和高。
    __window_width, __window_height = 550, 600

    def __init__(self):
        self.viewer = None  # 窗口。
        
        self.x=[140,220,300,380,460,140,300,460]
        self.y=[250,250,250,250,250,150,150,150]

        self.state = None

    def __create_window(self):
        from gym.envs.classic_control import rendering
        self.viewer = rendering.Viewer(self.__window_width, self.__window_height)
        
        # 创建网格世界Environment
        # 创建X轴
        for i in range(200, 500, 50):
            line_x = rendering.Line((150,i),(400,i))
            self.viewer.add_geom(line_x)
        # 创建Y轴
        for j in range(150, 450, 50):
            line_y = rendering.Line((j, 200), (j, 450))
            self.viewer.add_geom(line_y)

        # 创建陷阱
        # 创建第一组陷阱
        l, r, t, b = 150, 250, 350, 300
        trap = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        trap.set_color(1, 0, 0)
        self.viewer.add_geom(trap)
        # 创建第二组陷阱
        l, r, t, b = 300, 350, 450, 350
        trap = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        trap.set_color(1, 0, 0)
        self.viewer.add_geom(trap)
        # 创建第三组陷阱
        l, r, t, b = 250, 400, 200, 250
        trap = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        trap.set_color(1, 0, 0)
        self.viewer.add_geom(trap)
        
        # 创建出口
        l, r, t, b = 350, 400, 350, 300
        exit = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        exit.set_color(1, 0.9, 0)
        self.viewer.add_geom(exit)

        # 创建机器人Agent
        self.robot= rendering.make_circle(25)
        self.robotranslator = rendering.Transform()
        self.robot.add_attr(self.robotranslator)
        self.robot.set_color(0.8, 0.6, 0.4)
        self.viewer.add_geom(self.robot)
    
    def reset(self):
        pass

    def setState(self, state):
        self.state = state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.__create_window()

        # if self.state is None: return None
        # self.robotranslator.set_translation(self.x[self.state-1], self.y[self.state- 1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def step(self):
        pass
    
    def close(self):
        pass