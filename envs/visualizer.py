import pygame
import numpy as np

class PygameVisualizer:
    """Pygame 赌场环境可视化，支持智能体动态移动"""

    def __init__(self, env, width=600, height=600, fps=60):
        """
        初始化可视化器
        :param env: 迁移公共物品博弈环境 (MigratoryPGGEnv)
        :param width: 窗口宽度
        :param height: 窗口高度
        :param fps: 最大帧率
        """
        self.env = env
        self.width = width
        self.height = height
        self.fps = fps
        self.screen = None
        self.clock = pygame.time.Clock()
        self.cell_size = self.width // (env.grid_size // env.casino_size)  # 计算赌场格子大小

    def _initialize_pygame(self):
        """初始化 Pygame 窗口"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Migratory Public Goods Game")

    def set_current_casino(self, casino):
        """更新赌场位置，同时保存上一次的位置"""
        if self.current_casino is not None:
            self.last_position = self.current_casino  # 记录上一帧位置
        else:
            self.last_position = (0.0, 0.0)  # 避免 None，给默认初始值
        self.current_casino = casino

    def _draw_grid(self):
        """绘制赌场网格背景"""
        self.screen.fill((255, 255, 255))  # 清空屏幕（白色背景）
        num_cells = self.env.grid_size // self.env.casino_size  # 赌场网格数量

        for i in range(num_cells):
            for j in range(num_cells):
                pygame.draw.rect(
                    self.screen, (200, 200, 200),  # 灰色网格
                    (i * self.cell_size, j * self.cell_size, self.cell_size, self.cell_size),
                    1  # 线宽
                )

    def render(self):
        """让智能体从上一个赌场平滑移动到新赌场"""
        if self.screen is None:
            self._initialize_pygame()

        self._draw_grid()  

        num_frames = 10  # 让动画有 10 帧

        for frame in range(num_frames):
            self.screen.fill((255, 255, 255))
            self._draw_grid()

            for agent in self.env.agent_objects:
                old_x, old_y = agent.last_position
                new_x, new_y = agent.current_casino

                # 计算中间位置（线性插值）
                px = int((old_x * (num_frames - frame) + new_x * frame) / num_frames * self.width)
                py = int((old_y * (num_frames - frame) + new_y * frame) / num_frames * self.height)

                color = (0, 0, 255) if agent.is_cooperator else (255, 0, 0)
                pygame.draw.circle(self.screen, color, (px, py), 10)

            pygame.display.flip()
            self.clock.tick(60)  # 保持 60 FPS

