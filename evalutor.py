# --- START OF FILE eval_envs.py (for the updated marl_env.py and MARL policy) ---

import numpy as np
import matplotlib.pyplot as plt
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random # 用于为每个独立运行生成不同的基础种子
import torch # 新增: PyTorch
import os # 新增: 用于获取pid

# 假设 marl_env.py 与 eval_envs.py 在同一目录下
from envs.marl_env import MultiAgentGraphEnv, Agent

# 假设 algorithms 和 utils 目录与 eval_envs.py 在同一父目录下或在PYTHONPATH中
try:
    from algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy
    from algorithms.utils.gnn import GNNBase
except ImportError as e:
    print(f"错误：无法导入 MARL 策略相关的模块: {e}")
    print("请确保 'algorithms' 和 'utils' 目录在 Python 搜索路径中，并且包含必要的文件。")
    exit()


# --- 默认环境参数 (与之前保持一致) ---
DEFAULT_NUM_AGENTS = 25
DEFAULT_WORLD_SIZE = 10.0
DEFAULT_SPEED = 0.05
DEFAULT_EPISODE_LENGTH = 500
DEFAULT_RADIUS = 2.0 # 这个值应与 MARL_MODEL_ARGS.max_edge_dist 匹配
DEFAULT_COST = 1.0
DEFAULT_R_FACTOR = 5.0
DEFAULT_BETA = 1.0
DEFAULT_SEED = 42

DEFAULT_ENV_MAX_STEPS = DEFAULT_EPISODE_LENGTH
DEFAULT_COOP_LOWER_THRESHOLD = 0.01
DEFAULT_COOP_UPPER_THRESHOLD = 0.99
DEFAULT_SUSTAIN_DURATION = 50

# --- 评估参数 ---
NUM_PARALLEL_ENVS = 8
NUM_SEEDS_PER_POLICY = 100

# --- MARL 模型加载参数 ---
# !! 基于你的 train_mpgg.py 脚本中的 all_args !!
MARL_MODEL_ARGS = argparse.Namespace(
    # --- 从 train_mpgg.py 的 all_args 复制并调整 ---

    # === 网络结构与特性 (MLPBase, ACTLayer, PopArt) ===
    hidden_size=64,
    layer_N=2,
    use_ReLU=True,
    use_orthogonal=True,
    gain=0.01,
    use_feature_normalization=True,
    use_popart=True,

    # === GNN 相关参数 (GNNBase) ===
    gnn_hidden_size=64,
    gnn_num_heads=4,
    gnn_concat_heads=True,
    gnn_layer_N=2, # 将传递给 GNNBase 的 num_GNN_layers
    gnn_use_ReLU=True,
    embed_hidden_size=64,
    embed_layer_N=1,
    embed_use_ReLU=True,
    embed_add_self_loop=True,
    
    # !! 关键修改：取消注释并确保值正确 !!
    max_edge_dist=DEFAULT_RADIUS, # 使用环境的交互半径，与 train_mpgg.py 中的 2.0 一致

    # graph_feat_type="relative", # GNNBase的forward可能不直接用，但在构建node_obs时由外部处理
    actor_graph_aggr="node",
    critic_graph_aggr="global",
    global_aggr_type="mean",

    use_edge_feats=False,
    node_input_norm=False,
    use_residual_GNN=True,
    use_attention_GNN=True, # 因为 gnn_num_heads > 0
    num_GNN_layers=2,       # 对应 gnn_layer_N，GNNBase内部可能使用这个名字

    # === Critic 特定 ===
    use_cent_obs=True,

    # === GR_MAPPOPolicy 或 GR_Actor/Critic 构造函数可能需要的其他参数 ===
    lr=1e-4,
    critic_lr=1e-4,
    opti_eps=1e-5,
    weight_decay=0,
    split_batch=False,
    max_batch_size=2048
)

# 模型路径
ACTOR_MODEL_PATH = r"D:\DESKTOP\RL\MPGG\results\local_optimized_N25_L128_H64_GNNH64_Ent0.01\run3\models\actor.pt"
CRITIC_MODEL_PATH = r"D:\DESKTOP\RL\MPGG\results\local_optimized_N25_L128_H64_GNNH64_Ent0.01\run3\models\critic.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_env_args(seed_for_env_init=None):
    args = argparse.Namespace()
    args.num_agents = DEFAULT_NUM_AGENTS
    args.world_size = DEFAULT_WORLD_SIZE
    args.speed = DEFAULT_SPEED
    args.episode_length = DEFAULT_EPISODE_LENGTH
    args.radius = DEFAULT_RADIUS # 环境半径
    args.cost = DEFAULT_COST
    args.r = DEFAULT_R_FACTOR
    args.beta = DEFAULT_BETA
    args.seed = seed_for_env_init if seed_for_env_init is not None else DEFAULT_SEED
    args.env_max_steps = DEFAULT_ENV_MAX_STEPS
    args.cooperation_lower_threshold = DEFAULT_COOP_LOWER_THRESHOLD
    args.cooperation_upper_threshold = DEFAULT_COOP_UPPER_THRESHOLD
    args.sustain_duration = DEFAULT_SUSTAIN_DURATION
    return args

def static_policy(num_agents, agent_action_space, current_obs_package=None):
    return [np.array([0.0, 0.0], dtype=np.float32) for _ in range(num_agents)]

def random_policy(num_agents, agent_action_space, current_obs_package=None):
    actions = []
    for _ in range(num_agents):
        actions.append(agent_action_space.sample())
    return actions

def marl_policy_fn_generator():
    policy_instance_dict = {}

    def marl_policy_act(num_agents, agent_action_space, current_obs_package):
        pid = os.getpid()
        if pid not in policy_instance_dict:
            print(f"Process {pid}: Loading MARL model...")
            temp_env_args = get_env_args()
            temp_env = MultiAgentGraphEnv(temp_env_args)
            _ = temp_env.reset()

            current_marl_args = MARL_MODEL_ARGS
            
            # 确保 max_edge_dist 与环境的 radius 一致 (或至少是训练时的值)
            # 如果 MARL_MODEL_ARGS.max_edge_dist 是基于 DEFAULT_RADIUS 设置的，
            # 并且 get_env_args() 返回的 args.radius 也是 DEFAULT_RADIUS，则它们是一致的。
            if not hasattr(current_marl_args, 'max_edge_dist'):
                 print(f"警告: MARL_MODEL_ARGS 缺少 'max_edge_dist'. GNNBase.forward 可能失败。将使用环境的radius: {temp_env_args.radius}")
                 setattr(current_marl_args, 'max_edge_dist', temp_env_args.radius)
            elif current_marl_args.max_edge_dist != temp_env_args.radius:
                 print(f"警告: MARL_MODEL_ARGS.max_edge_dist ({current_marl_args.max_edge_dist}) 与环境 radius ({temp_env_args.radius}) 不匹配。GNN行为可能与训练时不同。")


            policy = GR_MAPPOPolicy(
                args=current_marl_args,
                obs_space=temp_env.observation_space[0],
                cent_obs_space=temp_env.share_observation_space[0],
                node_obs_space=temp_env.node_observation_space[0],
                edge_obs_space=temp_env.edge_observation_space[0],
                act_space=temp_env.action_space[0],
                device=DEVICE
            )
            try:
                policy.actor.load_state_dict(torch.load(ACTOR_MODEL_PATH, map_location=DEVICE))
                policy.actor.eval()
                policy_instance_dict[pid] = policy
                print(f"Process {pid}: MARL model loaded successfully to {DEVICE}.")
            except FileNotFoundError:
                print(f"错误: 找不到模型文件 {ACTOR_MODEL_PATH}")
                raise
            except Exception as e:
                print(f"加载或初始化模型时出错 (PID: {pid}): {e}")
                import traceback
                traceback.print_exc()
                raise
            del temp_env
        
        policy_instance = policy_instance_dict[pid]
        obs_n, agent_id_n, node_obs_n, adj_n = current_obs_package
        actions_list = []
        with torch.no_grad():
            for i in range(num_agents):
                obs_tensor = torch.tensor(obs_n[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                node_obs_tensor = torch.tensor(node_obs_n[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                adj_tensor = torch.tensor(adj_n[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                agent_id_tensor = torch.tensor(agent_id_n[i], dtype=torch.long).unsqueeze(0).to(DEVICE)
                
                action_tensor = policy_instance.act(obs_tensor, node_obs_tensor, adj_tensor, agent_id_tensor)
                action_np = action_tensor.squeeze(0).cpu().numpy()
                actions_list.append(action_np)
        return actions_list
    return marl_policy_act


def run_single_episode(env_init_args_tuple):
    env_init_args, policy_fn_generator_or_fn, run_seed, policy_name = env_init_args_tuple

    if policy_name == "marl_ppo":
        policy_fn_act = policy_fn_generator_or_fn()
    else:
        policy_fn_act = policy_fn_generator_or_fn

    current_env_args = argparse.Namespace(**vars(env_init_args))
    current_env_args.seed = run_seed
    
    env = MultiAgentGraphEnv(current_env_args)
    
    coop_rates_episode = []
    total_payoffs_episode = []
    
    obs_n_reset, agent_id_n_reset, node_obs_n_reset, adj_n_reset = env.reset()
    current_obs_package = (obs_n_reset, agent_id_n_reset, node_obs_n_reset, adj_n_reset)

    for step_num in range(current_env_args.episode_length):
        actions_list = policy_fn_act(env.num_agents, env.agent_action_space, current_obs_package)
        obs_n_next, agent_id_n_next, node_obs_n_next, adj_n_next, _, done_n, info_n = env.step(actions_list)
        current_obs_package = (obs_n_next, agent_id_n_next, node_obs_n_next, adj_n_next)

        if info_n and isinstance(info_n, list) and len(info_n) > 0 and \
           isinstance(info_n[0], dict) and \
           "step_cooperation_rate" in info_n[0] and \
           "step_avg_reward" in info_n[0]:
            coop_rates_episode.append(info_n[0]["step_cooperation_rate"])
            step_total_payoff = info_n[0]["step_avg_reward"] * env.num_agents
            total_payoffs_episode.append(step_total_payoff)
        else:
            coop_rates_episode.append(np.nan)
            total_payoffs_episode.append(np.nan)
    return np.array(coop_rates_episode), np.array(total_payoffs_episode)


def evaluate_policy(base_env_args, policy_fn_generator_or_fn, policy_name):
    print(f"评估策略: {policy_name}...")
    
    all_coop_rates_across_runs = []
    all_total_payoffs_across_runs = []

    task_args_list = []
    for i in range(NUM_SEEDS_PER_POLICY):
        run_specific_seed = (base_env_args.seed if base_env_args.seed is not None else 0) + i
        task_args_list.append((argparse.Namespace(**vars(base_env_args)),
                               policy_fn_generator_or_fn,
                               run_specific_seed,
                               policy_name))

    actual_parallel_envs = NUM_PARALLEL_ENVS
    if policy_name == "marl_ppo" and DEVICE.type == "cuda":
        if NUM_PARALLEL_ENVS > 1:
             print(f"警告: 为 MARL 策略 '{policy_name}' 在 GPU 上使用 {NUM_PARALLEL_ENVS} 个并行环境。请监控 GPU 显存。")

    # Store futures with their corresponding seeds for better error reporting
    futures_with_seeds = {}
    with ProcessPoolExecutor(max_workers=actual_parallel_envs) as executor:
        for task_args in task_args_list:
            future = executor.submit(run_single_episode, task_args)
            futures_with_seeds[future] = task_args[2] # task_args[2] is run_specific_seed

        for future in tqdm(as_completed(futures_with_seeds.keys()), total=NUM_SEEDS_PER_POLICY, desc=f"运行 {policy_name}"):
            run_seed_for_error = futures_with_seeds[future]
            try:
                coop_rates_ep, total_payoffs_ep = future.result()
                if len(coop_rates_ep) != base_env_args.episode_length or \
                   len(total_payoffs_ep) != base_env_args.episode_length:
                    print(f"警告: 策略 {policy_name} (Seed: {run_seed_for_error}) 的一次运行返回数据长度不匹配，已跳过。")
                    continue
                if not (np.isnan(coop_rates_ep).any() or np.isnan(total_payoffs_ep).any()):
                    all_coop_rates_across_runs.append(coop_rates_ep)
                    all_total_payoffs_across_runs.append(total_payoffs_ep)
                else:
                    print(f"警告: 策略 {policy_name} (Seed: {run_seed_for_error}) 的一次运行包含NaN值，已跳过。")
            except Exception as e:
                print(f"运行 episode 时发生错误 (策略 {policy_name}, Seed: {run_seed_for_error}): {e}")
                import traceback
                traceback.print_exc()


    if not all_coop_rates_across_runs or not all_total_payoffs_across_runs:
        print(f"警告: 策略 {policy_name} 没有足够（或任何）成功的实验数据。")
        return None, None, None, None

    try:
        stacked_coop_rates = np.stack(all_coop_rates_across_runs)
        stacked_total_payoffs = np.stack(all_total_payoffs_across_runs)
    except ValueError as e:
        print(f"错误: 无法堆叠结果数组 (策略 {policy_name}): {e}")
        return None, None, None, None
    
    mean_coop_rates = np.mean(stacked_coop_rates, axis=0)
    std_coop_rates = np.std(stacked_coop_rates, axis=0)
    mean_total_payoffs = np.mean(stacked_total_payoffs, axis=0)
    std_total_payoffs = np.std(stacked_total_payoffs, axis=0)
    
    return mean_coop_rates, std_coop_rates, mean_total_payoffs, std_total_payoffs


def plot_results(steps_axis, means, stds, label, color, ax):
    ax.plot(steps_axis, means, label=label, color=color)
    ax.fill_between(steps_axis, means - stds, means + stds, color=color, alpha=0.2)

def main():
    base_env_args = get_env_args(seed_for_env_init=None)
    
    policies_to_eval = [
        {"name": "static", "fn_generator_or_fn": static_policy, "color": "tab:blue"},
        {"name": "random", "fn_generator_or_fn": random_policy, "color": "tab:orange"},
        {"name": "marl_ppo", "fn_generator_or_fn": marl_policy_fn_generator, "color": "tab:green"}
    ]
    
    results = {}

    if DEFAULT_SEED is None:
        random.seed() 
    else:
        random.seed(DEFAULT_SEED)

    for policy_info in policies_to_eval:
        current_base_env_args = argparse.Namespace(**vars(base_env_args))
        mean_cr, std_cr, mean_tp, std_tp = evaluate_policy(
            current_base_env_args,
            policy_info["fn_generator_or_fn"],
            policy_info["name"]
        )
        if mean_cr is not None and mean_tp is not None:
            results[policy_info["name"]] = {
                "mean_coop_rate": mean_cr, "std_coop_rate": std_cr,
                "mean_total_payoff": mean_tp, "std_total_payoff": std_tp,
                "color": policy_info["color"]
            }

    if not results:
        print("没有有效的评估结果可供绘图。")
        return

    steps_axis = np.arange(base_env_args.episode_length)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    ax_coop = axes[0]
    for policy_name, data in results.items():
        plot_results(steps_axis, data["mean_coop_rate"], data["std_coop_rate"], 
                    policy_name, data["color"], ax_coop)
    ax_coop.set_ylabel("Cooperation Rate")
    ax_coop.set_title(f"Cooperation Rate (N={base_env_args.num_agents}, Eval Steps={base_env_args.episode_length})")
    ax_coop.legend()
    ax_coop.grid(True, linestyle='--')
    
    ax_payoff = axes[1]
    for policy_name, data in results.items():
        plot_results(steps_axis, data["mean_total_payoff"], data["std_total_payoff"], 
                    policy_name, data["color"], ax_payoff)
    ax_payoff.set_xlabel("Step")
    ax_payoff.set_ylabel("Total Payoff")
    ax_payoff.set_title(f"Total Payoff (N={base_env_args.num_agents}, Eval Steps={base_env_args.episode_length})")
    ax_payoff.legend()
    ax_payoff.grid(True, linestyle='--')
    
    plt.tight_layout()
    plt.savefig("evaluation_plots_with_marl_policy.png")
    print("评估图表已保存为 evaluation_plots_with_marl_policy.png")
    plt.show()

if __name__ == "__main__":
    # 确保 MARL_MODEL_ARGS.max_edge_dist 使用的是环境配置的 radius
    # 如果 DEFAULT_RADIUS 更新了，这里也会自动更新
    # 这只是一个额外的保障，因为在 MARL_MODEL_ARGS 定义时已经设置了
    if hasattr(MARL_MODEL_ARGS, 'max_edge_dist') and MARL_MODEL_ARGS.max_edge_dist != DEFAULT_RADIUS:
        print(f"更新 MARL_MODEL_ARGS.max_edge_dist 从 {MARL_MODEL_ARGS.max_edge_dist} 到 {DEFAULT_RADIUS} 以匹配环境配置。")
        MARL_MODEL_ARGS.max_edge_dist = DEFAULT_RADIUS
    elif not hasattr(MARL_MODEL_ARGS, 'max_edge_dist'):
        print(f"MARL_MODEL_ARGS 中未找到 max_edge_dist，将设置为默认环境半径: {DEFAULT_RADIUS}")
        setattr(MARL_MODEL_ARGS, 'max_edge_dist', DEFAULT_RADIUS)

    main()

# --- END OF FILE eval_envs.py ---