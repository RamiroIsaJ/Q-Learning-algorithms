import numpy as np
import time
import tkinter as tk


UNIT = 60
MAZE_H = 4
MAZE_W = 4


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Maze_RL')
        self.geometry('{0}x{1}'.format(MAZE_H*UNIT, MAZE_W*UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H*UNIT, width=MAZE_W*UNIT)

        # grids
        for c in range(0, MAZE_W*UNIT, UNIT):
            x0, x1, y0, y1 = c, 0, c, MAZE_H*UNIT
            self.canvas.create_line(x0, x1, y0, y1)
        for r in range(0, MAZE_H*UNIT, UNIT):
            x0, x1, y0, y1 = 0, r, MAZE_H*UNIT, r
            self.canvas.create_line(x0, x1, y0, y1)

        # origin
        origin = np.array([20, 20])

        # wall 1
        wall1_center = origin + np.array([UNIT*3, UNIT])
        self.wall1 = self.canvas.create_rectangle(wall1_center[0]-15, wall1_center[1]-15,
                                                  wall1_center[0]+35, wall1_center[1]+35,
                                                  fill='black')
        # wall 2
        wall2_center = origin + np.array([UNIT, UNIT*3])
        self.wall2 = self.canvas.create_rectangle(wall2_center[0]-15, wall2_center[1]-15,
                                                  wall2_center[0]+35, wall2_center[1]+35,
                                                  fill='black')

        # goal
        oval_center = origin + np.array([UNIT*3, UNIT*3])
        self.oval = self.canvas.create_oval(oval_center[0]-15, oval_center[1]-15,
                                            oval_center[0]+35, oval_center[1]+35,
                                            fill='red')

        # agent
        self.rect = self.canvas.create_rectangle(origin[0]-15, origin[1]-15,
                                                 origin[0]+35, origin[1]+35,
                                                 fill='green')

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(origin[0]-15, origin[1]-15,
                                                 origin[0]+35, origin[1]+35,
                                                 fill='green')
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # UP
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # DOWN
            if s[1] < (MAZE_H - 1)*UNIT:
                base_action[1] += UNIT
        elif action == 2:  # RIGHT
            if s[0] < (MAZE_W - 1)*UNIT:
                base_action[0] += UNIT
        elif action == 3:  # LEFT
            if s[0] > UNIT:
                base_action[0] -= UNIT
        # move agent
        self.canvas.move(self.rect, base_action[0], base_action[1])
        # new state
        s_ = self.canvas.coords(self.rect)
        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 10
            done = True
        elif s_ in [self.canvas.coords(self.wall1), self.canvas.coords(self.wall2)]:
            reward = -5
            done = True
        else:
            reward = 1
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


UNIT_ = 60
MAZE_H_ = 5
MAZE_W_ = 5


class MazeRL(tk.Tk, object):
    def __init__(self):
        super(MazeRL, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('MAZE - Reinforcement Learning')
        self.geometry('{0}x{1}'.format(MAZE_H_*UNIT_, MAZE_W_*UNIT_))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H_*UNIT_, width=MAZE_W_*UNIT_)

        # grids
        for c in range(0, MAZE_W_*UNIT_, UNIT_):
            x0, x1, y0, y1 = c, 0, c, MAZE_H_*UNIT_
            self.canvas.create_line(x0, x1, y0, y1)
        for r in range(0, MAZE_H_*UNIT_, UNIT_):
            x0, x1, y0, y1 = 0, r, MAZE_H_*UNIT_, r
            self.canvas.create_line(x0, x1, y0, y1)

        # origin
        origin = np.array([20, 20])

        # wall 1
        wall1_center = origin + np.array([UNIT_, UNIT_])
        self.wall1 = self.canvas.create_rectangle(wall1_center[0]-15, wall1_center[1]-15,
                                                  wall1_center[0]+35, wall1_center[1]+35,
                                                  fill='black')
        # wall 2
        wall2_center = origin + np.array([UNIT_, UNIT_*2])
        self.wall2 = self.canvas.create_rectangle(wall2_center[0]-15, wall2_center[1]-15,
                                                  wall2_center[0]+35, wall2_center[1]+35,
                                                  fill='black')
        # wall 3
        wall3_center = origin + np.array([UNIT_, UNIT_*3])
        self.wall3 = self.canvas.create_rectangle(wall3_center[0] - 15, wall3_center[1] - 15,
                                                  wall3_center[0] + 35, wall3_center[1] + 35,
                                                  fill='gray')
        # wall 4
        wall4_center = origin + np.array([0, UNIT_*3])
        self.wall4 = self.canvas.create_rectangle(wall4_center[0] - 15, wall4_center[1] - 15,
                                                  wall4_center[0] + 35, wall4_center[1] + 35,
                                                  fill='black')

        # wall 5
        wall5_center = origin + np.array([UNIT_*3, UNIT_*3])
        self.wall5 = self.canvas.create_rectangle(wall5_center[0] - 15, wall5_center[1] - 15,
                                                  wall5_center[0] + 35, wall5_center[1] + 35,
                                                  fill='gray')
        # wall 6
        wall6_center = origin + np.array([UNIT_ * 4, UNIT_ * 3])
        self.wall6 = self.canvas.create_rectangle(wall6_center[0] - 15, wall6_center[1] - 15,
                                                  wall6_center[0] + 35, wall6_center[1] + 35,
                                                  fill='black')
        # wall 7
        wall7_center = origin + np.array([UNIT_*3, 0])
        self.wall7 = self.canvas.create_rectangle(wall7_center[0] - 15, wall7_center[1] - 15,
                                                  wall7_center[0] + 35, wall7_center[1] + 35,
                                                  fill='black')
        # wall 8
        wall8_center = origin + np.array([UNIT_ * 3, UNIT_])
        self.wall8 = self.canvas.create_rectangle(wall8_center[0] - 15, wall8_center[1] - 15,
                                                  wall8_center[0] + 35, wall8_center[1] + 35,
                                                  fill='black')

        # exit
        exit_center = origin + np.array([UNIT_*4, UNIT_*4])
        self.exit = self.canvas.create_oval(exit_center[0] - 15, exit_center[1] - 15,
                                            exit_center[0] + 35, exit_center[1] + 35,
                                            fill='red')

        # agent
        self.rect = self.canvas.create_rectangle(origin[0]-15, origin[1]-15,
                                                 origin[0]+35, origin[1]+35,
                                                 fill='green')

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(origin[0]-15, origin[1]-15,
                                                 origin[0]+35, origin[1]+35,
                                                 fill='green')
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # UP
            if s[1] > UNIT_:
                base_action[1] -= UNIT_
        elif action == 1:  # DOWN
            if s[1] < (MAZE_H_ - 1)*UNIT_:
                base_action[1] += UNIT_
        elif action == 2:  # RIGHT
            if s[0] < (MAZE_W_ - 1)*UNIT_:
                base_action[0] += UNIT_
        elif action == 3:  # LEFT
            if s[0] > UNIT_:
                base_action[0] -= UNIT_
        # move agent
        self.canvas.move(self.rect, base_action[0], base_action[1])
        # new state
        s_ = self.canvas.coords(self.rect)
        # reward function
        if s_ == self.canvas.coords(self.exit):
            reward = 10
            done = True
        elif s_ in [self.canvas.coords(self.wall1), self.canvas.coords(self.wall2), self.canvas.coords(self.wall4),
                    self.canvas.coords(self.wall6), self.canvas.coords(self.wall7), self.canvas.coords(self.wall8)]:
            reward = -5
            done = True
        elif s_ in [self.canvas.coords(self.wall3), self.canvas.coords(self.wall5)]:
            reward = -2
            done = True
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = Maze()
    env.after(100, update())
    env.mainloop()





