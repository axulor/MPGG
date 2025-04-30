import numpy as np
import argparse
from envs.marl_env import MultiAgentGraphEnv, Agent

def main():
    # 初始化环境参数
    args = argparse.Namespace(
        num_agents=100,
        world_size=100,
        speed=1.0,
        max_cycles=100,
        radius=2.5,
        cost=1.0,
        r=1.5,
        beta=0.5,
        discrete_action=False
    )

    # 创建环境实例
    env = MultiAgentGraphEnv(args)
    print(f"Environment created: num_agents={env.num_agents}, world_size={env.world_size}")

    # 测试 reset
    env.seed(42)
    obs_n1, id_n1, node_obs1, adj1 = env.reset()

    print("\n-- After reset() --")
    print(f"obs_n: type={type(obs_n1)}, length={len(obs_n1)}")
    print(f"node_obs_n: type = {type(node_obs1)}, length={len(node_obs1)}")
    print(f"obs_n[0]: shape={obs_n1[0].shape}, dtype={obs_n1[0].dtype}")
    print(f"node_obs_n[0]: shape={node_obs1[0].shape}")
    print(f"agent_id_n[0]: shape={id_n1[0].shape}, dtype={id_n1[0].dtype}")
    print(f"adj_n[0]: shape={adj1[0].shape}")

    # 演示 share_obs 和 share_agent_id 转换
    # 假设 n_rollout_threads = 1
    n_rollout_threads = 1
    # 将 list 转为数组: (num_agents, feature_dim)
    obs_array = np.array(obs_n1)         # shape: (N, D)
    agent_id_array = np.array(id_n1)     # shape: (N, 1)

    # 1. 摊平 obs: (1, N*D)
    share_obs = obs_array.reshape(n_rollout_threads, -1)
    # 2. 插入智能体维度: (1, 1, N*D)
    share_obs = np.expand_dims(share_obs, 1)
    # 3. 沿智能体维度复制 N 份: (1, N, N*D)
    share_obs = share_obs.repeat(env.num_agents, axis=1)

    # 同样操作 ID: (1, N) -> (1,1,N) -> (1, N, N)
    share_agent_id = agent_id_array.reshape(n_rollout_threads, -1)
    share_agent_id = np.expand_dims(share_agent_id, 1)
    share_agent_id = share_agent_id.repeat(env.num_agents, axis=1)

    print("\n-- Demonstration of share_obs and share_agent_id --")
    print("\n")
    print(f"share_obs.shape = {share_obs.shape}")
    print(f"share_obs[0,0,:5] sample = {share_obs[0,0,:5]}")
    print(f"share_obs[0,1,:5] should equal share_obs[0,0,:5]: {np.allclose(share_obs[0,0], share_obs[0,1])}")
    print(f"share_agent_id.shape = {share_agent_id.shape}")
    print(f"share_agent_id[0,0] = {share_agent_id[0,0]}")
    print(f"share_agent_id[0,1] = {share_agent_id[0,1]} (should equal share_agent_id[0,0])")

    # 测试 step
    action_n = [0] * env.num_agents
    obs_n2, id_n2, node_obs2, adj2, reward_n, done_n, info_n = env.step(action_n)
    print("\n-- After step() --")
    print(f"obs_n: length={len(obs_n2)}, obs_n[0].shape={obs_n2[0].shape}")
    print(f"reward_n sample: {reward_n[:5]}")
    print(f"done_n sample: {done_n[:5]}")
    print(f"info_n[0]: {info_n[0]}")

    # 测试 calculate_distances 和 update_graph
    env.calculate_distances()
    print("\n-- After calculate_distances() --")
    print(f"cached_dist_vect.shape = {env.cached_dist_vect.shape}")
    print(f"cached_dist_mag.shape  = {env.cached_dist_mag.shape}")

    env.update_graph()
    print("\n-- After update_graph() --")
    print(f"edge_list.shape   = {env.edge_list.shape}")
    print(f"edge_weight.shape = {env.edge_weight.shape}")

    # 测试单个 agent 特征提取
    feat = env._get_agent_feat(env.agents[0])
    print("\n-- Agent feature --")
    print(f"feature shape = {feat.shape}, values = {feat}")

    # 测试 seed 的可重复性
    env.seed(123)
    obs_a, *_ = env.reset()
    env.seed(123)
    obs_b, *_ = env.reset()
    assert np.allclose(obs_a[0], obs_b[0]), "Seed reset not reproducible"
    print("\nSeed reproducibility test passed.")

if __name__ == "__main__":
    main()
