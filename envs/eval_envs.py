# --- START OF FILE eval_envs.py (for the updated marl_env.py) ---

import numpy as np
import matplotlib.pyplot as plt
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random # 用于为每个独立运行生成不同的基础种子

# 假设 marl_env.py 与 eval_envs.py 在同一目录下
from marl_env import MultiAgentGraphEnv, Agent # Agent 类也可能需要（虽然这里不直接用）

# --- 默认环境参数 ---
DEFAULT_NUM_AGENTS = 25
DEFAULT_WORLD_SIZE = 10.0
DEFAULT_SPEED = 0.05 # 注意：这个速度值将与单位方向向量相乘
DEFAULT_EPISODE_LENGTH = 1000 # 每轮的step数 (用于评估)
DEFAULT_RADIUS = 2.0
DEFAULT_COST = 1.0
DEFAULT_R_FACTOR = 5.0
DEFAULT_BETA = 1.0
DEFAULT_SEED = 42 # 或者一个具体的整数，如 42

# 新增的环境参数默认值 (对应 marl_env.py 中的新参数)
DEFAULT_ENV_MAX_STEPS = DEFAULT_EPISODE_LENGTH # 环境内部的最大步数，评估时可以设为与评估长度一致或更长
DEFAULT_COOP_LOWER_THRESHOLD = 0.01 # 示例值
DEFAULT_COOP_UPPER_THRESHOLD = 0.99 # 示例值

# --- 评估参数 ---
NUM_PARALLEL_ENVS = 8
NUM_SEEDS_PER_POLICY = 100 # 每种移动策略运行的独立实验次数 (每个实验有自己的种子)

def get_env_args(seed_for_env_init=None):
    """创建环境参数的Namespace对象"""
    args = argparse.Namespace()
    args.num_agents = DEFAULT_NUM_AGENTS
    args.world_size = DEFAULT_WORLD_SIZE
    args.speed = DEFAULT_SPEED
    # episode_length 用于 eval_envs.py 的评估循环长度
    # env_max_steps 是传递给环境的参数，用于环境自身的终止条件
    args.episode_length = DEFAULT_EPISODE_LENGTH # 这个参数会被env.__init__接收，但env内部主要用env_max_steps
    args.radius = DEFAULT_RADIUS
    args.cost = DEFAULT_COST
    args.r = DEFAULT_R_FACTOR
    args.beta = DEFAULT_BETA
    args.seed = seed_for_env_init if seed_for_env_init is not None else DEFAULT_SEED
    
    # 添加 marl_env.py 中新增的参数
    args.env_max_steps = DEFAULT_ENV_MAX_STEPS
    return args

def static_policy(num_agents, agent_action_space):
    """静止策略：所有智能体都输出表示静止的动作"""
    # marl_env.py 中的 update_direction_vector 会处理 [0,0] 为静止
    return [np.array([0.0, 0.0], dtype=np.float32) for _ in range(num_agents)]

def random_policy(num_agents, agent_action_space):
    """随机移动策略：每个智能体输出一个随机的连续动作向量"""
    actions = []
    for _ in range(num_agents):
        actions.append(agent_action_space.sample())
    return actions

def run_single_episode(env_init_args, policy_fn, run_seed):
    """
    运行单个环境的一个完整episode，并收集数据。
    env_init_args: 用于初始化环境的参数 (argparse.Namespace)。
    policy_fn: 生成所有智能体动作列表的函数。
    run_seed: 用于本次特定运行的随机种子，会传递给环境的 _seed 方法。
    """
    current_env_args = argparse.Namespace(**vars(env_init_args)) # 创建副本
    current_env_args.seed = run_seed # 为当前环境实例设置特定的种子
    
    env = MultiAgentGraphEnv(current_env_args) # 初始化环境
    
    coop_rates_episode = []
    total_payoffs_episode = [] # 将记录每一步的总体收益
    
    # env.reset() 返回 obs_n, agent_id_n, node_obs_n, adj_n
    # 我们在这里不需要这些返回值，所以用 _ 忽略
    _ = env.reset() 
    
    # 评估脚本中的 episode_length (来自current_env_args) 控制评估的长度
    # 注意：这与环境内部的 env_max_steps 是独立的。
    # 如果 env_max_steps < episode_length，环境可能提前 done。
    # 但对于固定长度的评估，我们通常会让评估脚本的循环跑完。
    for step_num in range(current_env_args.episode_length):
        actions_list = policy_fn(env.num_agents, env.agent_action_space)
        
        # marl_env.py step 返回: obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n
        # 我们主要关心 info_n
        _, _, _, _, _, done_n, info_n = env.step(actions_list) # 捕获 done_n
        
        # 从info_n中提取全局指标 (所有智能体的info[X]中的全局指标值都一样)
        if info_n and isinstance(info_n, list) and len(info_n) > 0 and \
           isinstance(info_n[0], dict) and \
           "step_cooperation_rate" in info_n[0] and \
           "step_avg_reward" in info_n[0]:
            coop_rates_episode.append(info_n[0]["step_cooperation_rate"])
            # 从平均奖励重建总奖励
            step_total_payoff = info_n[0]["step_avg_reward"] * env.num_agents
            total_payoffs_episode.append(step_total_payoff)
        else:
            coop_rates_episode.append(np.nan)
            total_payoffs_episode.append(np.nan)
            print(f"警告: Seed {run_seed}, step {step_num + 1} 的 info_n ({info_n}) 不完整或缺失期望的键。")

        # 如果环境报告了 done (例如因为合作率或 env_max_steps)，
        # 并且我们希望评估在环境终止时停止，可以在这里 break。
        # 但对于固定长度的基准测试图，通常会跑满 episode_length。
        # if all(done_n): # 或者 done_n[0] 如果 done 对所有智能体都一样
        #     print(f"环境在 step {step_num + 1} 提前终止 (Seed {run_seed}).")
        #     # 如果提前终止，可能需要用 NaN 填充剩余的 steps
        #     remaining_steps = current_env_args.episode_length - (step_num + 1)
        #     if remaining_steps > 0:
        #         coop_rates_episode.extend([np.nan] * remaining_steps)
        #         total_payoffs_episode.extend([np.nan] * remaining_steps)
        #     break


    return np.array(coop_rates_episode), np.array(total_payoffs_episode)


def evaluate_policy(base_env_args, policy_fn, policy_name):
    """评估单个策略，运行多次(NUM_SEEDS_PER_POLICY)并平均结果"""
    print(f"评估策略: {policy_name}...")
    
    all_coop_rates_across_runs = []
    all_total_payoffs_across_runs = []

    # 为 ProcessPoolExecutor 生成独立的种子序列
    # python_random_seed = random.randint(0, 2**32 - 1) if DEFAULT_SEED is None else DEFAULT_SEED
    # print(f"为 {policy_name} 的 ProcessPoolExecutor 任务生成器使用基础种子: {python_random_seed}")
    # local_random = random.Random(python_random_seed)


    with ProcessPoolExecutor(max_workers=NUM_PARALLEL_ENVS) as executor:
        futures = []
        for i in range(NUM_SEEDS_PER_POLICY):
            # 每个环境实例的种子基于 (base_env_args.seed 或 0) + 偏移量 i
            # 这确保了即使 eval_envs.py 多次运行 (若DEFAULT_SEED固定)，特定实验(i)的环境种子是固定的
            # 如果 DEFAULT_SEED 为 None 且 base_env_args.seed 为 None，则每次运行 eval_envs.py 时，
            # 这一系列种子都会从 0, 1, 2... 开始，这是可复现的。
            run_specific_seed = (base_env_args.seed if base_env_args.seed is not None else 0) + i
            futures.append(executor.submit(run_single_episode, base_env_args, policy_fn, run_specific_seed))
        
        for future in tqdm(as_completed(futures), total=NUM_SEEDS_PER_POLICY, desc=f"运行 {policy_name}"):
            try:
                coop_rates_ep, total_payoffs_ep = future.result()
                
                # 检查长度是否符合预期
                if len(coop_rates_ep) != base_env_args.episode_length or \
                   len(total_payoffs_ep) != base_env_args.episode_length:
                    print(f"警告: 策略 {policy_name} 的一次运行返回数据长度不匹配 ({len(coop_rates_ep)} vs {base_env_args.episode_length})，已跳过。")
                    continue

                if not (np.isnan(coop_rates_ep).any() or np.isnan(total_payoffs_ep).any()):
                    all_coop_rates_across_runs.append(coop_rates_ep)
                    all_total_payoffs_across_runs.append(total_payoffs_ep)
                else:
                    print(f"警告: 策略 {policy_name} 的一次运行包含NaN值 (可能来自info缺失或提前终止未填充)，已跳过。")

            except Exception as e:
                print(f"运行 episode 时发生错误: {e}")
                import traceback
                traceback.print_exc()

    if not all_coop_rates_across_runs or not all_total_payoffs_across_runs:
        print(f"警告: 策略 {policy_name} 没有足够（或任何）成功的实验数据。")
        return None, None, None, None

    # 确保所有收集到的episode数据长度一致才进行stack
    # （尽管上面的检查应该已经处理了，这里再确认一下）
    if not all(len(arr) == base_env_args.episode_length for arr in all_coop_rates_across_runs) or \
       not all(len(arr) == base_env_args.episode_length for arr in all_total_payoffs_across_runs):
        print(f"错误: 策略 {policy_name} 的数据在聚合前长度不一致，无法堆叠。")
        # 可以选择进一步调试，打印出各个数组的长度
        # for i, arr in enumerate(all_coop_rates_across_runs): print(f"Run {i} coop rates length: {len(arr)}")
        # for i, arr in enumerate(all_total_payoffs_across_runs): print(f"Run {i} payoffs length: {len(arr)}")
        return None, None, None, None

    try:
        stacked_coop_rates = np.stack(all_coop_rates_across_runs)
        stacked_total_payoffs = np.stack(all_total_payoffs_across_runs)
    except ValueError as e:
        print(f"错误: 无法堆叠结果数组，可能由于长度不一致 (尽管有前置检查): {e}")
        return None, None, None, None
    
    mean_coop_rates = np.mean(stacked_coop_rates, axis=0)
    std_coop_rates = np.std(stacked_coop_rates, axis=0)
    
    mean_total_payoffs = np.mean(stacked_total_payoffs, axis=0)
    std_total_payoffs = np.std(stacked_total_payoffs, axis=0)
    
    return mean_coop_rates, std_coop_rates, mean_total_payoffs, std_total_payoffs

def plot_results(steps_axis, means, stds, label, color, ax):
    """绘制平均曲线和标准差区域"""
    ax.plot(steps_axis, means, label=label, color=color)
    ax.fill_between(steps_axis, means - stds, means + stds, color=color, alpha=0.2)

def main():
    # base_env_args.seed (如果非None) 将用作生成 run_specific_seed 的基础
    # 如果 base_env_args.seed 为 None，则 run_specific_seed 从 0, 1, ... 开始
    base_env_args = get_env_args(seed_for_env_init=None) 
    
    policies_to_eval = [
        {"name": "static", "fn": static_policy, "color": "tab:blue"},
        {"name": "random", "fn": random_policy, "color": "tab:orange"}
    ]
    
    results = {}

    # 初始化Python的全局random模块种子。
    # 这会影响 random.randint 等，但不会直接影响 ProcessPoolExecutor 任务内环境的 np_random，
    # 因为环境的种子是由 run_specific_seed 控制的。
    # 如果 DEFAULT_SEED 是 None，则每次运行 eval_envs.py 都是随机的（基于时间等）。
    # 如果 DEFAULT_SEED 是固定值，则 eval_envs.py 的整体行为（如任务提交顺序可能影响的微小计时差异）会更可复现。
    # 关键的环境种子 (run_specific_seed) 的生成不依赖此处的 random.seed。
    if DEFAULT_SEED is None:
        random.seed() 
    else:
        random.seed(DEFAULT_SEED)


    for policy_info in policies_to_eval:
        mean_cr, std_cr, mean_tp, std_tp = evaluate_policy(base_env_args, policy_info["fn"], policy_info["name"])
        if mean_cr is not None and mean_tp is not None: # 确保返回了有效数据
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
    ax_coop.set_title(f"Cooperation Rate (MARL Env, N={base_env_args.num_agents}, Eval Steps={base_env_args.episode_length})")
    ax_coop.legend()
    ax_coop.grid(True, linestyle='--')
    
    ax_payoff = axes[1]
    for policy_name, data in results.items():
        plot_results(steps_axis, data["mean_total_payoff"], data["std_total_payoff"], 
                    policy_name, data["color"], ax_payoff)
    ax_payoff.set_xlabel("Step")
    ax_payoff.set_ylabel("Total Payoff") # 标签保持 "Total Payoff"
    ax_payoff.set_title(f"Total Payoff (MARL Env, N={base_env_args.num_agents}, Eval Steps={base_env_args.episode_length})")
    ax_payoff.legend()
    ax_payoff.grid(True, linestyle='--')
    
    plt.tight_layout()
    plt.savefig("evaluation_plots_marl_env_updated.png")
    print("评估图表已保存为 evaluation_plots_marl_env_updated.png")
    plt.show()

if __name__ == "__main__":
    main()

# --- END OF FILE eval_envs.py (for the updated marl_env.py) ---