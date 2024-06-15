# 导入相关包
import os
import random
import numpy as np
from Maze import Maze
from Runner import Runner
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本
import matplotlib.pyplot as plt


import numpy as np

# 机器人移动方向
move_map = {
    'u': (-1, 0), # up
    'r': (0, +1), # right
    'd': (+1, 0), # down
    'l': (0, -1), # left
}


# 迷宫路径搜索树
class SearchTree(object):


    def __init__(self, loc=(), action='', parent=None):
        """
        初始化搜索树节点对象
        :param loc: 新节点的机器人所处位置
        :param action: 新节点的对应的移动方向
        :param parent: 新节点的父辈节点
        """

        self.loc = loc  # 当前节点位置
        self.to_this_action = action  # 到达当前节点的动作
        self.parent = parent  # 当前节点的父节点
        self.children = []  # 当前节点的子节点

    def add_child(self, child):
        """
        添加子节点
        :param child:待添加的子节点
        """
        self.children.append(child)

    def is_leaf(self):
        """
        判断当前节点是否是叶子节点
        """
        return len(self.children) == 0


def expand(maze, is_visit_m, node):
    """
    拓展叶子节点，即为当前的叶子节点添加执行合法动作后到达的子节点
    :param maze: 迷宫对象
    :param is_visit_m: 记录迷宫每个位置是否访问的矩阵
    :param node: 待拓展的叶子节点
    """
    can_move = maze.can_move_actions(node.loc)
    for a in can_move:
        new_loc = tuple(node.loc[i] + move_map[a][i] for i in range(2))
        if not is_visit_m[new_loc]:
            child = SearchTree(loc=new_loc, action=a, parent=node)
            node.add_child(child)


def back_propagation(node):
    """
    回溯并记录节点路径
    :param node: 待回溯节点
    :return: 回溯路径
    """
    path = []
    while node.parent is not None:
        path.insert(0, node.to_this_action)
        node = node.parent
    return path


def breadth_first_search(maze):
    """
    对迷宫进行广度优先搜索
    :param maze: 待搜索的maze对象
    """
    start = maze.sense_robot()
    root = SearchTree(loc=start)
    queue = [root]  # 节点队列，用于层次遍历
    h, w, _ = maze.maze_data.shape
    is_visit_m = np.zeros((h, w), dtype=np.int)  # 标记迷宫的各个位置是否被访问过
    path = []  # 记录路径
    while True:
        current_node = queue[0]
        is_visit_m[current_node.loc] = 1  # 标记当前节点位置已访问

        if current_node.loc == maze.destination:  # 到达目标点
            path = back_propagation(current_node)
            break

        if current_node.is_leaf():
            expand(maze, is_visit_m, current_node)

        # 入队
        for child in current_node.children:
            queue.append(child)

        # 出队
        queue.pop(0)

    return path


def my_search(maze):
    """
    任选深度优先搜索算法、最佳优先搜索（A*)算法实现其中一种
    :param maze: 迷宫对象
    :return :到达目标点的路径 如：["u","u","r",...]
    """

    path = []

    # -----------------请实现你的算法代码--------------------------------------
    start = maze.sense_robot()
    root = SearchTree(loc=start)
    stack = [root]
    h, w, _ = maze.maze_data.shape
    is_visit_m = np.zeros((h, w), dtype=int)

    while stack:
        current_node = stack[-1]
        is_visit_m[current_node.loc] = 1

        if current_node.loc == maze.destination:
            path = back_propagation(current_node)
            break
        
        if current_node.is_leaf():
            expand(maze, is_visit_m, current_node)
            
        has_unvisited_child = False
        for child in reversed(current_node.children):  # 反转子节点顺序以确保深度优先遍历
            if not is_visit_m[child.loc]:
                stack.append(child)
                has_unvisited_child = True

        if not has_unvisited_child:  # 如果没有未访问的子节点，回溯
            stack.pop()

    # -----------------------------------------------------------------------
    return path


import random
import numpy as np
import torch
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot
from Maze import Maze

class Robot(TorchRobot):
    def __init__(self, maze):
        super(Robot, self).__init__(maze)
        maze.set_reward({
            "hit_wall": 5.0,
            "destination": -maze.maze_size ** 2,
            "default": 1.0,
        })
        self.maze = maze
        self.epsilon = 0
        self.memory.build_full_view(maze=maze)
        self.loss_list = self._train()

    def _train(self):
        loss_list = []
        while True:
            loss_list.append(self._learn(batch=len(self.memory)))
            self.reset()
            for _ in range(self.maze.maze_size ** 2 - 1):
                _, reward = self.test_update()
                if reward == self.maze.reward["destination"]:
                    return loss_list

    def train_update(self):
        state = self.sense_state()
        action = self._choose_action(state)
        reward = self.maze.move_robot(action)
        return action, reward

    def test_update(self):
        state = torch.from_numpy(np.array(self.sense_state(), dtype=np.int16)).float().to(self.device)
        self.eval_model.eval()
        with torch.no_grad():
            q_value = self.eval_model(state).cpu().data.numpy()
        action = self.valid_action[np.argmin(q_value).item()]
        reward = self.maze.move_robot(action)
        return action, reward

# 示例用法
if __name__ == "__main__":
    maze = Maze(maze_size=5)
    robot = Robot(maze=maze)
    print(robot.maze.reward)
    runner = Runner(robot=robot)
    runner.run_training(training_epoch=10, training_per_epoch=75)
    runner.generate_gif('pytorch.gif')
    robot.reset()
    for _ in range(25):
        a, r = robot.test_update()
        print("action:", a, "reward:", r)
        if r == maze.reward["destination"]:
            print("success")
            break
    print(maze)