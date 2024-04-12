import random
from copy import deepcopy
import numpy as np
import time

reverse_color = {'X': 'O', 'O': 'X'}
num2char = {0: 'X', 1: 'O', 2: '-'}

class Node:

    def __init__(self, board, color, root_color, parent=None, action=None):
        self.board = board
        self.color = color
        self.root_color = root_color
        self.parent = parent
        self.children = []
        self.best_child = None
        self.get_best_child()
        self.action = action
        self.actions = list(self.board.get_legal_actions(color=color))
        self.isOver = self.game_over(self.board)
        self.N = 0
        self.Q = {'X': 0, 'O': 0}
        self.value = {'X': 1e5, 'O': 1e5}
        self.isLeaf = True
        self.best_reward_child = None
        self.get_best_reward_child()

    def game_over(self, board):
        b_list = list(board.get_legal_actions('X'))
        w_list = list(board.get_legal_actions('O'))
        is_over = len(b_list) == 0 and len(w_list) == 0 
        return is_over

    def UCB1_value(self, c=2):
        if self.N == 0:
            return
        for color in ['X', 'O']:
            self.value[color] = self.Q[color] / self.N + c * np.sqrt(2*np.log(self.parent.N) / self.N) 
        
    def add_child(self, child):
        self.children.append(child)
        self.get_best_child()
        self.get_best_reward_child()
        self.isLeaf = False

    # 搜索的时候用, 找UCB值最大的子节点，
    def get_best_child(self):
        if len(self.children) == 0:
            self.best_child = None
        else:
            sorted_children = sorted(self.children, key=lambda child: child.value[self.color], reverse=True)
            self.best_child = sorted_children[0]
        return self.best_child
    
    # 搜索过程结束后使用, 用于选择具有最高累积回报的节点
    def get_best_reward_child(self):
        if len(self.children) == 0:
            best_reward_child = None
        else:
            sorted_children = sorted(self.children, key=lambda child: child.Q[self.color] / child.N if child.N > 0 else -1e5, reverse=True)
            best_reward_child = sorted_children[0]
        self.best_reward_child=best_reward_child
        return self.best_reward_child


class MCTS:

    def __init__(self, board, color):
        self.root = Node(board=deepcopy(board), color=color, root_color=color)
        self.color = color
        # self.epsilon的概率随机选择，每次选择后会以self.gamma的概率衰减
        self.eps = 0.3
        self.gamma = 0.999

    def UCT_Search(self, time_limit=30):
        """
        输入: 时间限制 time_limit
        输出: 玩家MAX行动下，当前最优动作a
        """
        if len(self.root.actions) == 1:
            return self.root.actions[0]
        
        time_start = time.time()

        while time.time() - time_start < time_limit:
            current_node = self.SelectPolicy()  
            if current_node.isOver:  
                winner, score = current_node.board.get_winner()
            else:
                if current_node.N: 
                    current_node = self.Expand(current_node)  
                winner, score = self.SimulatePolicy(current_node)  
            self.BackPropagate(current_node, winner, score)  

        return self.root.get_best_reward_child().action
    

    def SelectPolicy(self):
        """
        输出: 选择步骤的结束节点 v
        """
        v = self.root
        while not v.isLeaf:
            if random.random() > self.eps:
                v = v.get_best_child()
            else:
                v = random.choice(v.children)
            self.eps *= self.gamma # 一开始倾向于随机探索，之后倾向于UCB值
        return v


    def Expand(self, v: Node):
        """
        输入: 节点 v
        输出: 未被扩展的后继节点 v.best_child
        """
        
        if len(v.actions) == 0:
            board = deepcopy(v.board)
            child = Node(board=board, color=reverse_color[v.color], parent=v, action="none", root_color=self.color)
            v.add_child(child)
            return v.best_child
        
        for action in v.actions:
            board = deepcopy(v.board) # 这里绝对不能省事，每个动作都需要从该结点 deepcopy() 出新的棋盘进行模拟
            board._move(action, v.color)
            child = Node(board=board, color=reverse_color[v.color], parent=v, action=action, root_color=self.color)
            v.add_child(child)
        return v.best_child

    def SimulatePolicy(self, node: Node):
        """
        输出：模拟终止时的胜者 winner, 分差 score
        """
        board = deepcopy(node.board)
        color = node.color
        while not node.game_over(board=board):
            actions = list(board.get_legal_actions(color))
            if len(actions) != 0:
                board._move(random.choice(actions), color)
            color = reverse_color[color]
        winner, diff = board.get_winner()
        return winner, diff
    
    def BackPropagate(self, v: Node, winner, score):
        """
        输入: 反向传播更新的起始节点 v, 终局状态胜者 winner, 分差 score
        """
        winner = num2char[winner]
        while v is not None:
            v.N += 1
            if winner != '-': # 平局得分不变，不考虑
                v.Q[winner] += score # reward中胜者的颜色的加分
                v.Q[reverse_color[winner]] -= score # reward中败者颜色的扣大分
                
            if v is not self.root:
                for child in v.parent.children:
                    child.UCB1_value()
            v = v.parent

class AIPlayer:
    """
    AI 玩家
    """
    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        self.color = color

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        # -----------------请实现你的算法代码--------------------------------------
        action = MCTS(board, self.color).UCT_Search()
        # ------------------------------------------------------------------------
        
        return action