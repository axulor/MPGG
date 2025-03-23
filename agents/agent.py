import numpy as np
import random

class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id   # 智能体编号
        self.current_node = None   # 当前所在节点 (row, col)
        self.last_node = None      # 上一次所在的节点
        self.is_cooperator = random.choice([True, False])  # 初始化策略

    def set_current_node(self, node):
        """传入并更新节点位置，同时保存上一次的位置"""
        if self.current_node is not None:
            self.last_node = self.current_node  # 记录上一帧位置
        else:
            self.last_node = node  # 第一次初始化时，last_position 和 current_node 相同
        self.current_node = node
