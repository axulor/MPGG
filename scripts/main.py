from envs.migratory_pgg_env import MigratoryPGGEnv
from algorithms.rule_based_policy import RuleBasedPolicy
import numpy as np
import time
import matplotlib.pyplot as plt  # 导入绘图库

# 初始化环境参数
env = MigratoryPGGEnv(
    N=100,             # 智能体数量为100
    max_cycles=500,   # 最大运行周期为500步
    size=20,          # 二维空间大小
    speed=0.1,         # 移动速度
    radius=2.0,       # 感知邻居的半径
    cost=1.0,          # 合作成本
    r=4.0,             # 公共物品博弈中贡献的倍增因子
    beta=1.0,          # 策略更新时的参数
    seed =42,          # 随机种子
    visualize=False    # 关闭实时可视化
)

# 初始化规则策略，random, static,complex_cluster
policy = RuleBasedPolicy(strategy_name="complex_cluster", speed=0.1)

# 重置环境，获取初始观测
obs = env.reset()

# 用于存储每一步的合作率、总收益以及步数


steps = []
cooperation_rates = []
# coop_rate = env.cooperation_rate()
# print(f"Initial Cooperation Rate: {coop_rate:.2f}")
# cooperation_rates.append(coop_rate)

total_payoffs = []
# total_payoff = env.total_payoff()
# total_payoffs.append(total_payoff)

# 模拟主循环
for step in range(env.max_cycles):
    actions = {}
    # 针对每个智能体选择动作
    for agent_id in env.agents:
        actions[agent_id] = policy.select_action(agent_id, obs)
    
    # 根据所有智能体动作更新环境状态
    obs, rewards, terminations, truncations, infos = env.step(actions)
    
    # 记录当前步的合作率和总收益
    coop_rate = env.cooperation_rate()
    total_payoff = env.total_payoff()
    cooperation_rates.append(coop_rate)
    total_payoffs.append(total_payoff)
    steps.append(step)
    
    # 输出当前步的统计信息
    print(f"Step {step:03d} | Coop Rate: {coop_rate:.2f} | Total Payoff: {total_payoff:.2f}")
    
    # 若所有智能体均达到终止条件则退出循环
    if all(truncations.values()):
        break

env.close()

# 绘制合作率和总收益随时间变化的曲线
plt.figure(figsize=(12, 5))

# 绘制合作率曲线
plt.subplot(1, 2, 1)
plt.plot(steps, cooperation_rates, label="Cooperation Rate", color="blue")
plt.xlabel("Step")
plt.ylabel("Cooperation Rate")
plt.title("Cooperation Rate vs Time")
plt.legend()

# 绘制总收益曲线
plt.subplot(1, 2, 2)
plt.plot(steps, total_payoffs, label="Total Payoff", color="red")
plt.xlabel("Step")
plt.ylabel("Total Payoff")
plt.title("Total Payoff vs Time")
plt.legend()

plt.tight_layout()
plt.show()
