import numpy as np
import random

class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id # 智能体编号
        self.current_casino = None  # 当前所在赌场
        self.last_casino = None  # 上一次所在赌场的位置
        self.is_cooperator = random.choice([True, False])  # 初始化策略
        # self.valid_moves = self.compute_valid_actions()  # 初始化合法动作


    def set_current_casino(self, casino):
        """更新所处的赌场位置，同时保存上一次的位置"""
        if self.current_casino is not None:
            self.last_casino = self.current_casino  # 记录上一帧位置
        else:
            self.last_casino = casino  # 第一次初始化时，last_position 和 current_casino 相同
        self.current_casino = casino


    def valid_moves(self, step_size):
        """计算智能体在当前位置的合法移动动作"""
        x, y = self.current_casino
        possible_moves = {0, 1, 2, 3, 4}  # {上, 下, 左, 右, 不动}

        if x <= step_size:
            possible_moves.discard(2)  # 不能向左
        if x >= 1.0:
            possible_moves.discard(3)  # 不能向右
        if y <= step_size:
            possible_moves.discard(1)  # 不能向下
        if y >= 1.0:
            possible_moves.discard(0)  # 不能向上

        return sorted(possible_moves)

    # def update_position(self, new_casino):
    #     """更新智能体位置，并重新计算合法动作"""
    #     self.current_casino = new_casino
    #     self.valid_moves = self.compute_valid_actions()  # 重新计算合法动作  
