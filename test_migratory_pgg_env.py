from envs.migratory_pgg_env import MigratoryPGGEnv

# 创建环境
env = MigratoryPGGEnv(num_agents=4)

obs, _ = env.reset()
print("✅ 初始观察:", obs)

for _ in range(10):
    actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
    obs, rewards, dones, truncations, infos = env.step(actions)
    env.render()
    print(f"✅ 观察: {obs}, 奖励: {rewards}, 终止: {dones}")

env.close()
