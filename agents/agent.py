import numpy as np
import random

class Agent:
    def __init__(self, agent_id, state_size, action_size, learning_rate=0.1, discount_factor=0.99):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = self.initialize_q_table()
        self.current_casino = None  # 当前所在赌场
        self.is_cooperator = random.choice([True, False])  # 是否为合作者

    def initialize_q_table(self):
        # 初始化Q表
        return np.zeros((self.state_size, self.action_size))

    def choose_action(self, state, epsilon=0.1):
        # 选择动作的策略
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)  # 探索
        else:
            return np.argmax(self.q_table[state])  # 利用

    def update_q_table(self, state, action, reward, next_state):
        # 更新Q表
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def set_current_casino(self, casino):
        # 设置当前所在赌场
        self.current_casino = casino

    def set_cooperator_status(self, status):
        # 设置是否为合作者
        self.is_cooperator = status

    def save_model(self, file_path):
        # 保存模型
        raise NotImplementedError("This method should be overridden by subclasses.")

    def load_model(self, file_path):
        # 加载模型
        raise NotImplementedError("This method should be overridden by subclasses.") 