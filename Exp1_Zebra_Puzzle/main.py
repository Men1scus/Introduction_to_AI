from kanren import run, eq, membero, var, conde        # kanren一个描述性Python逻辑编程系统
from kanren.core import lall                           # lall包用于定义规则
import time

###############################################################################
####             可在此处定义自己所需要用到的自定义函数(可选)                  #### 
####   提示：定义左邻近规则left(), 定义右邻近规则right(),定义邻近规则next()    ####
###############################################################################
#               

# Reference：https://blog.csdn.net/weixin_46291251/article/details/122246347
def left(a, b, list):
    return membero((a, b), zip(list, list[1:]))
# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
# 如果 list = [1, 2, 3, 4]，那么 zip(list, list[1:]) 将产生 [(1, 2), (2, 3), (3, 4)]
# 相当于 a 元素从 list 里面取，b 元素从 list 的第二位开始取，二者构成一个元组

def right(a, b, list):
    return membero((b, a), zip(list, list[1:]))
# right 就是 left 反过来

def next(a, b, list):
         return conde([left(a, b, list)], [right(a, b, list)])



#                                                                             #
###############################################################################
#################                非必要性工作                 ################## 
###############################################################################

class Agent:
    """
    推理智能体.
    """
    
    def __init__(self):
        """
        智能体初始化.
        """
        
        self.units = var()              # 单个unit变量指代一座房子的信息(国家，工作，饮料，宠物，颜色) 
                                        # 例如('英国人', '油漆工', '茶', '狗', '红色')即为正确格式，但不是本题答案
                                        # 请基于给定的逻辑提示求解五条正确的答案
        self.rules_zebraproblem = None  # 用lall包定义逻辑规则
        self.solutions = None           # 存储结果
        
    def define_rules(self):
        """
        定义逻辑规则.
        """

        self.rules_zebraproblem = lall(
            (eq, (var(), var(), var(), var(), var()), self.units),         # self.units共包含五个unit成员，即每一个unit对应的var都指代一座房子(国家，工作，饮料，宠物，颜色) 
                                                                           # 各个unit房子又包含五个成员属性: (国家，工作，饮料，宠物，颜色)
            
            ##############################################################################
            ####               请在以下区域中添加逻辑规则，感受逻辑约束问题               #### 
            ####     输出：五条房子匹配信息('英国人', '油漆工', '茶', '狗', '红色')       ####
            ##############################################################################
            #                                                                            #
            
            # 示例：基于问题信息可以提炼出，有人养斑马，有人喜欢喝矿泉水等信息
            (membero, (var(), var(), var(), '斑马', var()), self.units),   
            (membero, (var(), var(), '矿泉水', var(), var()), self.units),
            
            (membero, ('英国人', var(), var(), var(), '红色'), self.units),
            (membero, ('西班牙人', var(), var(), '狗', var()), self.units),
            (membero, ('日本人', '油漆工', var(), var(), var()), self.units),
            (membero, ('意大利人', var(), '茶', var(), var()), self.units),
            (membero, (var(), '小提琴家', '橘子汁', var(), var()), self.units),

            (eq, 
                (('挪威人', var(), var(), var(), var()), var(), var(), var(), var()), 
                self.units),  # 挪威人住在左边的第一个房子里
            
            (eq,
                (var(), var(), (var(), var(), '牛奶', var(), var()), var(), var()),
                self.units),
            
            (membero, (var(), '摄影师', var(), '蜗牛', var()), self.units),
            (membero, (var(), '外交官', var(), var(), '黄色'), self.units),
            (membero, (var(), var(), '咖啡', var(), '绿色'), self.units),
            
            (right, 
                (var(), var(), var(), var(), '绿色'),
                (var(), var(), var(), var(), '白色'),
                self.units),
        
            (next,
                ('挪威人', var(), var(), var(), var()),
                (var(), var(), var(), var(), '蓝色'),
                self.units),
            (next,
                (var(), var(), var(), '狐狸', var()),
                (var(), '医生', var(), var(), var()),
                self.units),
            (next,
                (var(), var(), var(), '马', var()),
                (var(), '外交官', var(), var(), var()),
                self.units)
            #                                                                            #
            ##############################################################################
            #################             完成后请记得提交作业             ################# 
            ##############################################################################

        )
    
    def solve(self):
        """
        规则求解器(请勿修改此函数).
        return: 斑马规则求解器给出的答案，共包含五条匹配信息，解唯一.
        """
        
        self.define_rules()
        self.solutions = run(0, self.units, self.rules_zebraproblem)
        return self.solutions


agent = Agent()
solutions = agent.solve()

# 提取解释器的输出
output = [house for house in solutions[0] if '斑马' in house][0][4]
print ('\n{}房子里的人养斑马'.format(output))
output = [house for house in solutions[0] if '矿泉水' in house][0][4]
print ('{}房子里的人喜欢喝矿泉水'.format(output))

# 解释器的输出结果展示
for i in solutions[0]:
    print(i)