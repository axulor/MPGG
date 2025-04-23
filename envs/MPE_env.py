import argparse
# 假设合并后的文件名为 marl_env.py
from envs.marl_env import MultiAgentGraphEnv

def GraphMPEEnv(args: argparse.Namespace):
    """
    工厂函数，用于创建合并后的 MultiAgentGraphEnv 实例。

    Args:
        args (argparse.Namespace): 包含环境配置参数的对象。

    Returns:
        MultiAgentGraphEnv: 初始化后的多智能体图环境实例。
    """
    # 直接实例化合并后的环境类
    env = MultiAgentGraphEnv(args=args)
    return env

if __name__ == "__main__":
    # === 示例用法 ===
    parser = argparse.ArgumentParser()
    # 添加必要的参数 (与 MultiAgentGraphEnv 的 __init__ 中使用的匹配)
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument("--world_size", type=float, default=100.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--max_cycles", type=int, default=500)
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--cost", type=float, default=1.0)
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--scenario_name", type=str, default="mpgg_graph")
    parser.add_argument("--discrete_action", type=bool, default=True)
    # 你可能需要添加其他在 args 中使用的参数
    # ...

    args = parser.parse_args() # 解析命令行参数，或者在脚本中直接创建 args 对象

    # 创建环境实例
    env = GraphMPEEnv(args)

    # --- 测试环境接口 ---
    print("环境创建成功!")
    print(f"智能体数量: {env.num_agents}")
    print(f"动作空间 (Agent 0): {env.action_space[0]}")
    print(f"观测空间 (Agent 0): {env.observation_space[0]}")
    print(f"节点特征空间 (Agent 0): {env.node_observation_space[0]}")
    print(f"邻接矩阵空间 (Agent 0): {env.adj_observation_space[0]}")

    # 测试 reset
    print("\n=== 测试 reset ===")
    obs_n, agent_id_n, node_obs_n, adj_n = env.reset()
    print(f"初始观测 obs_n[0] shape: {obs_n[0].shape}")
    print(f"初始 ID agent_id_n[0]: {agent_id_n[0]}")
    print(f"初始节点特征 node_obs_n[0] shape: {node_obs_n[0].shape}")
    print(f"初始邻接矩阵 adj_n[0] shape: {adj_n[0].shape}")
    print(f"Agent 0 初始策略: {env.agents[0].strategy}")

    # 测试 step
    print("\n=== 测试 step ===")
    # 生成随机动作 (假设离散动作空间)
    random_actions = [env.action_space[i].sample() for i in range(env.num_agents)]
    obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n = env.step(random_actions)
    print(f"单步后观测 obs_n[0] shape: {obs_n[0].shape}")
    print(f"单步后奖励 reward_n[0]: {reward_n[0]}")
    print(f"单步后完成 done_n[0]: {done_n[0]}")
    print(f"单步后信息 info_n[0]: {info_n[0]}")
    print(f"Agent 0 单步后策略: {env.agents[0].strategy}")
    print(f"Agent 0 单步后位置: {env.agents[0].pos}")

# --- END OF FILE MPE_env.py ---




# def GraphMPEEnv(args):
#     """
#     Same as MPEEnv but for graph environment
#     """
#     from envs.scenario import Scenario
#     from envs.environment import MultiAgentGraphEnv

#     scenario = Scenario() # 实例化场景

#     # 实例化环境
#     env = MultiAgentGraphEnv(
#         world=scenario.make_world(args=args),                   # 创建世界
#         reset_callback=scenario.reset_world,                    # 重置世界
#         reward_callback=scenario.reward,                        # 计算奖励
#         observation_callback=scenario.observation,              # 返回观测
#         graph_observation_callback=scenario.graph_observation,  # 返回图结构观测
#         update_graph=scenario.update_graph,                     # 更新图结构
#         id_callback=scenario.get_id,                            # 获得智能体 id
#         info_callback=scenario.info_callback,                   # 获得 info
#         scenario_name=args.scenario_name,                       # 场景名称
#     )

#     return env


# if __name__ == "__main__":
#     import argparse
#     import numpy as np
#     from envs.scenario import Scenario
#     from envs.environment import MultiAgentGraphEnv

#     # 构造模拟 args
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--scenario_name", type=str, default="navigation_graph")
#     parser.add_argument("--num_agents", type=int, default=10)
#     parser.add_argument("--num_obstacles", type=int, default=2)
#     parser.add_argument("--num_scripted_agents", type=int, default=0)
#     parser.add_argument("--world_size", type=float, default=1.0)
#     parser.add_argument("--collaborative", type=bool, default=True)
#     parser.add_argument("--max_speed", type=float, default=2.0)
#     parser.add_argument("--collision_rew", type=float, default=1.0)
#     parser.add_argument("--goal_rew", type=float, default=10.0)
#     parser.add_argument("--min_dist_thresh", type=float, default=0.05)
#     parser.add_argument("--use_dones", type=bool, default=True)
#     parser.add_argument("--episode_length", type=int, default=25)
#     parser.add_argument("--graph_feat_type", type=str, default="global")
#     parser.add_argument("--max_edge_dist", type=float, default=1.5)
#     args = parser.parse_args(args=[])

#     # 实例化 scenario 和 world
#     scenario = Scenario()
#     world = scenario.make_world(args)

#     # 测试 graph_observation 返回值和维度
#     print("\n=== 单个 agent 的 graph_observation 返回值和维度 ===")
#     for agent in world.agents:
#         node_obs, adj = scenario.graph_observation(agent, world)
#         print(f"{agent.name}: node_obs type={type(node_obs)}, dtype={node_obs.dtype}, shape={node_obs.shape} | adj type={type(adj)}, dtype={adj.dtype}, shape={adj.shape}")

    # # 拿一个 agent 测试 callback 函数
    # agent = world.agents[0]

    # print("\n=== reset_callback ===")
    # out = scenario.reset_world(world)
    # print("reset_world 返回值:", out)

    # print("\n=== reward_callback ===")
    # rew = scenario.reward(agent, world)
    # print(f"reward(agent, world): {rew} ({type(rew)})")

    # print("\n=== observation_callback ===")
    # obs = scenario.observation(agent, world)
    # print(f"observation(agent, world): {obs.shape} ({type(obs)})")

    # print("\n=== graph_observation_callback ===")
    # node_obs, adj = scenario.graph_observation(agent, world)
    # print(f"node_obs: shape={node_obs.shape}, type={type(node_obs)}")
    # print(f"adj: shape={adj.shape}, type={type(adj)}")

    # print("\n=== update_graph ===")
    # result = scenario.update_graph(world)
    # print(f"update_graph 返回值: {result}")

    # print("\n=== id_callback ===")
    # agent_id = scenario.get_id(agent)
    # print(f"get_id(agent): {agent_id} ({type(agent_id)}, shape={agent_id.shape})")

    # print("\n=== info_callback ===")
    # info = scenario.info_callback(agent, world)
    # print(f"info_callback(agent, world): {info} ({type(info)})")

    # 测试 observation() 返回值
    # print("\n=== 单个 agent 的 observation 返回值和维度 ===")
    # for agent in world.agents:
    #     obs = scenario.observation(agent, world)
    #     obs2 = np.random.rand(6).astype(np.float32)
    #     print(f"{agent.name}: {obs}    shape={obs.shape}")
    #     print(f"{agent.name}: {obs2}    shape={obs2.shape}")
        # agent_id = scenario.get_id(agent)
        # print(f"{agent.name}: id={agent_id}, type={type(agent_id)}, dtype={agent_id.dtype}, shape={agent_id.shape}")

