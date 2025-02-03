from envs.migratory_pgg_env import MigratoryPGGEnv
import random

def test_environment():
    # 创建环境实例
    env = MigratoryPGGEnv(L=9, l=3, r_min=1.2, r_max=5.0, N=100)
    
    # 重置环境
    initial_state = env.reset()
    print("初始状态:", initial_state)
     # 可视化环境
    env.render()
    
    # 执行一个动作
    actions = {agent: random.choice([0, 1]) for agent in env.agents}  # 0: 不贡献, 1: 贡献
    next_state, rewards, done = env.step(actions)
    print("下一状态:", next_state)
    print("奖励:", rewards)
    print("是否结束:", done)
    
    # 可视化环境
    env.render()

if __name__ == "__main__":
    test_environment() 