from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from agents.agent import Agent
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.lines import Line2D  # 用于创建图例

class MigratoryPGGEnv(ParallelEnv):
    """ 多智能体迁徙公共物品博弈环境 (Migratory Public Goods Game) """
    
    metadata = {"render_modes": ["human"], "name": "migratory_pgg_env"}
    
    def __init__(self, L=5, l=1, r_min=1.5, r_max=4.0, N=150):
        super().__init__()
        
        # 关键参数定义
        self.grid_size = L   # 总区域边长大小
        self.casino_size = l     # 每个赌场边长
        self.r_params = (r_min, r_max) # 增益因子范围
        self.N = N # 智能体数量
        self.casinos = []  # 初始化赌场列表
        self.phase = "pre_game"  # 预设为博弈前阶段


        # 定义智能体
        self.agents = ["agent_" + str(i) for i in range(N)]
        # 定义观测/动作空间
        self.game_action_spaces = {agent: spaces.Discrete(2) for agent in self.agents}  # 0: 不贡献, 1: 贡献
        self.move_action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}  # 0: 上, 1: 下, 2: 左, 3: 右, 4: 不动

        # 博弈后的观测空间
        self.post_game_observation_spaces = {
            agent: spaces.Tuple((
                spaces.Box(low=0.01, high=1.0, shape=(), dtype=np.float32),  # 成本参数 c
                spaces.Box(low=0.01, high=1.0, shape=(), dtype=np.float32),  # 增益参数 alpha
                spaces.Box(low=0, high=self.N, shape=(), dtype=np.int32),  # 赌场的智能体数目
                spaces.Box(low=0, high=self.N, shape=(), dtype=np.int32)   # 赌场的合作智能体数目
            )) for agent in self.agents
        }
        # 博弈前的观测空间
        self.pre_game_observation_spaces = {
            agent: spaces.Tuple((
                spaces.Box(low=0.01, high=1.0, shape=(), dtype=np.float32),  # 成本参数 c
                spaces.Box(low=0.01, high=1.0, shape=(), dtype=np.float32),  # 增益参数 alpha
                spaces.Box(low=0, high=self.N, shape=(), dtype=np.int32)  # 赌场的智能体数目
            )) for agent in self.agents
        }
        self.observation_spaces = self.pre_game_observation_spaces.copy()  # 预设为博弈前状态
        self.action_spaces = self.game_action_spaces.copy()  # 预设为博弈前的动作空间


    # def observation_space(self, agent):
    #     """根据当前阶段返回正确的观测空间"""
    #     if self.phase == "pre_game":
    #         return self.pre_game_observation_spaces[agent]
    #     else:
    #         return self.post_game_observation_spaces[agent]
        
    def observation_space(self, agent):
        """返回正确的 Gym 观测空间"""
        if self.phase == "pre_game":
            return self.pre_game_observation_spaces.get(agent, None)
        else:
            return self.post_game_observation_spaces.get(agent, None)


    def action_space(self, agent):
        """根据当前阶段返回正确的动作空间"""
        if self.phase == "pre_game":
            return self.game_action_spaces[agent]
        else:
            return self.move_action_spaces[agent]


    def action_space(self, agent):
        """根据智能体所在赌场位置，动态调整可选的移动动作"""
        if self.phase == "pre_game":
            return self.game_action_spaces[agent]  # 贡献决策

        # 获取赌场网格信息
        num_casinos_per_row = self.grid_size // self.casino_size
        step = 1.0 / num_casinos_per_row  # 赌场之间的步长

        # 获取智能体位置
        agent_obj = self.agent_objects[self.agents.index(agent)]
        x, y = agent_obj.current_casino  # 赌场坐标

        # 可能的移动动作
        possible_moves = set([0, 1, 2, 3, 4])  # {上, 下, 左, 右, 不动}

        # 移除非法动作
        if x <= step:  # 如果赌场在最左边，则不能向左移动
            possible_moves.discard(2)  # 不能向左
        if x >= 1.0:  # 如果赌场在最右边，则不能向右移动
            possible_moves.discard(3)  # 不能向右
        if y <= step:  # 如果赌场在最下边，则不能向下移动
            possible_moves.discard(1)  # 不能向下
        if y >= 1.0:  # 如果赌场在最上边，则不能向上移动
            possible_moves.discard(0)  # 不能向上

        return spaces.Discrete(len(possible_moves))  # 返回合法动作数量




    def initialize_agents(self):
        """初始化智能体并均匀分布到赌场。"""

        agents = []
        num_casinos_row = self.grid_size // self.casino_size
        total_casinos = num_casinos_row ** 2
        step = 1 / num_casinos_row

        # 创建所有赌场
        self.casinos = []  # 确保赌场信息存储在类属性中
        for row in range(num_casinos_row):
            for col in range(num_casinos_row):
                c = (row + 1) * step
                alpha = (col + 1) * step
                self.casinos.append((c, alpha))

        # 均匀分配智能体到赌场
        agent_id = 0
        for casino in self.casinos:
            for _ in range(self.N // total_casinos):
                agent = Agent(agent_id, state_size=3, action_size=2)
                agent.set_current_casino(casino)
                agents.append(agent)
                agent_id += 1

        # 随机分配剩余的智能体
        random_casinos = random.sample(self.casinos, self.N % total_casinos)
        for casino in random_casinos:
            agent = Agent(agent_id, state_size=3, action_size=2)
            agent.set_current_casino(casino)
            agents.append(agent)
            agent_id += 1

        return agents
        

    def reset(self):
        """重置环境到初始状态"""
        self.phase = "pre_game"
        self.agent_objects = self.initialize_agents()

        # 重置每个智能体的观测
        observations = {
            agent: obs for agent, obs in zip(self.agents, self.get_pre_game_observation())
        }

        # 更新 observation_spaces 和 action_spaces
        self.observation_spaces = self.pre_game_observation_spaces
        self.action_spaces = self.game_action_spaces

        infos = {agent: {} for agent in self.agents}  # PettingZoo 规范
        return observations, infos


    def step(self, actions):
        """根据当前阶段处理不同类型的决策"""
        rewards = {agent: 0 for agent in self.agents}

        if self.phase == "pre_game":
            # 处理贡献决策
            for agent, game_action in actions.items():
                agent_obj = self.agent_objects[self.agents.index(agent)]
                rewards[agent] = self.get_reward(agent_obj, game_action)

            # 进入博弈后阶段
            self.phase = "post_game"
            observations = {agent: obs for agent, obs in zip(self.agents, self.get_post_game_observation())}
            self.observation_spaces = self.post_game_observation_spaces
            self.action_spaces = self.move_action_spaces

        else:
            # 处理移动决策
            for agent, move_action in actions.items():
                agent_obj = self.agent_objects[self.agents.index(agent)]
                self.move_agents(agent_obj, move_action)

            # 重新进入博弈前阶段
            self.phase = "pre_game"
            observations = {agent: obs for agent, obs in zip(self.agents, self.get_pre_game_observation())}
            self.observation_spaces = self.pre_game_observation_spaces
            self.action_spaces = self.game_action_spaces

        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos




    def get_pre_game_observation(self):
        """获取博弈前的观测状态。"""
        observations = []
        for agent in self.agent_objects:  # 使用 agent_objects
            m = agent.current_casino  # 当前赌场
            c, alpha = m  # 成本参数和增益因子
            n = self.get_agent_count(m)  # 当前赌场的智能体数量
            observations.append((c, alpha, n))
        return observations

    def get_post_game_observation(self):
        """获取博弈后的观测状态。"""
        observations = []
        for agent in self.agent_objects:  # 使用 agent_objects
            m = agent.current_casino  # 当前赌场
            c, alpha = m  # 成本参数和增益因子
            n = self.get_agent_count(m)  # 当前赌场的智能体数量
            n_C = self.get_cooperator_count(m)  # 当前赌场的合作者数量
            observations.append((c, alpha, n, n_C))
        return observations

    def get_agent_count(self, casino):
        """获取指定赌场的智能体数量。"""
        return sum(1 for agent in self.agent_objects if agent.current_casino == casino)

    def get_cooperator_count(self, casino):
        """获取指定赌场的合作者数量。"""
        return sum(1 for agent in self.agent_objects if agent.current_casino == casino and agent.is_cooperator)

    def get_reward(self, agent, game_action):
        """计算并返回智能体在当前状态下的奖励。"""
        m = agent.current_casino
        c, alpha = m  # 从赌场索引中获取成本和增益因子
        n = self.get_agent_count(m)
        n_C = self.get_cooperator_count(m)
        
        # 计算增益因子 r
        r = self.r_params[0] + (self.r_params[1] - self.r_params[0]) * alpha
        
        # 计算奖励
        if game_action == 1:  # 如果智能体选择合作
            reward = (r * n_C) / n - c
        else:  # 如果智能体选择背叛
            reward = (r * n_C) / n

        
        return reward

    def check_done(self):
        """检查是否结束。"""
        pass  # 当前不执行任何操作


    def move_agents(self, agent_obj, move_action):
        """根据智能体的迁移策略更新其赌场位置，并防止越界"""

        # 计算赌场网格
        num_casinos_per_row = self.grid_size // self.casino_size  # 计算每行赌场数
        step = 1.0 / num_casinos_per_row  # 每个赌场的步长（网格大小）

        # 获取智能体的当前位置
        old_x, old_y = agent_obj.current_casino

        # 移动增量
        move_delta = {
            0: (0, step),   # 上移
            1: (0, -step),  # 下移
            2: (-step, 0),  # 左移
            3: (step, 0),   # 右移
            4: (0, 0)      # 不动
        }

        dx, dy = move_delta[move_action]

        # 计算新位置，并限制在赌场边界内
        new_x = max(step, min(1.0 - step, old_x + dx))
        new_y = max(step, min(1.0 - step, old_y + dy))

        # 更新智能体的位置
        agent_obj.set_current_casino((new_x, new_y))


    def render(self, mode="human"):
        """使用 Rectangle + Text 绘制通用可扩展网格，确保刻度严格对齐方格中央"""
        fig, ax = plt.subplots(figsize=(4, 4))
        # 计算赌场网格大小
        num_casinos = int(self.grid_size / self.casino_size)  # 计算网格数量（行/列数）
        step = 1.0 / num_casinos  # 计算每个小方格的边长
        # 确保刻度严格在小方格中央
        grid_centers = np.round(np.linspace(step, 1.0, num_casinos), decimals=3)

        # 计算赌场边界（用于画网格线）
        grid_lines = np.linspace(0, 1, num_casinos + 1)

        # 刻度标签的偏移量（让刻度稍微远离轴）
        offset_factor = 0.2 * step  # 适应不同网格大小

        # 绘制网格
        for i in range(num_casinos):
            for j in range(num_casinos):
                x_min = i * step
                y_min = j * step
                # 使用 Rectangle 画方格
                rect = patches.Rectangle((x_min, y_min), step, step, linewidth=1, edgecolor='black', facecolor='none')
                ax.add_patch(rect)

        # 绘制智能体
        for agent in self.agent_objects:
            c, alpha = agent.current_casino  # 赌场坐标
            # 计算赌场所在的网格索引
            row = int((c - step) / step)  # 修正 row 计算
            col = int((alpha - step) / step)  # 修正 col 计算
            # 计算该赌场的左下角坐标
            x_min = col * step
            y_min = row * step
            # 在赌场方格内生成智能体的随机位置
            jitter_x = np.random.uniform(x_min + 0.1 * step, x_min + 0.9 * step)
            jitter_y = np.random.uniform(y_min + 0.1 * step, y_min + 0.9 * step)
            # 按策略着色：合作者蓝色，背叛者红色
            color = 'blue' if agent.is_cooperator else 'red'
            ax.scatter(jitter_x, jitter_y, c=color, s=40, edgecolors='white')

        # 绘制刻度
        for x in grid_centers:
            # 底部 X 轴刻度
            ax.text(x - step / 2, -offset_factor, f"{x:.3f}", ha='center', va='center', fontsize=10)
            # 左侧 Y 轴刻度
            ax.text(-offset_factor, x - step / 2, f"{x:.3f}", ha='center', va='center', fontsize=10)

        # 修正刻度线对齐
        ax.set_xticks(grid_centers - step / 2)
        ax.set_xticklabels([])
        ax.set_yticks(grid_centers - step / 2)
        ax.set_yticklabels([])

        # 网格线设置
        ax.set_xticks(grid_lines, minor=True)
        ax.set_yticks(grid_lines, minor=True)
        ax.grid(True, which='minor', linestyle="--", alpha=0.5)

        # 修正坐标轴边框 & 刻度线
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['left'].set_position(('data', 0))

        # 确保坐标范围
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Cooperator'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Defector')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=12)
        ax.annotate(r"$c$", xy=(0.90, -0.05), xycoords='axes fraction', fontsize=20, ha='left', va='center')
        ax.annotate(r"$\alpha$", xy=(-0.05, 0.90), xycoords='axes fraction', fontsize=20, ha='center', va='bottom')

        # 显示图像
        if mode == "human":
            plt.tight_layout()
            plt.show()
        else:
            print("please use human mode")






