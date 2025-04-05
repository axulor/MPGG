import numpy as np
import matplotlib.pyplot as plt
from envs.migratory_pgg_env_v4 import MigratoryPGGEnv
from policy.rule_based_policy import RuleBasedPolicy
import multiprocessing as mp
from tqdm import tqdm

# ==== 单次模拟运行函数（不使用封装） ==== #
def run_experiment(args):
    strategy, seed = args
    env = MigratoryPGGEnv(
        N=100,
        max_cycles=500,
        size=20,
        speed=0.1,
        radius=2.0,
        cost=1.0,
        r=4.0,
        beta=1.0,
        seed=seed,
        visualize=False
    )
    policy = RuleBasedPolicy(strategy_name=strategy, speed=0.1)
    obs = env.reset()
    cooperation_rates = []
    total_payoffs = []
    for _ in range(env.max_cycles):
        actions = {agent: policy.select_action(agent, obs) for agent in env.agents}
        obs, _, terminations, truncations, _ = env.step(actions)
        cooperation_rates.append(env.cooperation_rate())
        total_payoffs.append(env.total_payoff())
        if all(truncations.values()):
            break
    env.close()
    return strategy, cooperation_rates, total_payoffs

# ==== 主函数入口 ==== #
def main():
    # ==== 配置 ==== #
    strategies = ["random", "static", "complex_cluster"]
    seeds = list(range(140))  # 共140个种子
    num_batches = 10
    batch_size = 14

    # ==== 初始化结果容器 ==== #
    all_results = {s: {"cooperation": [], "payoff": []} for s in strategies}

    # ==== 分批执行并行任务 ==== #
    for batch_idx in tqdm(range(num_batches), desc="Batches"):
        batch_seeds = seeds[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        args_list = [(s, seed) for s in strategies for seed in batch_seeds]

        with mp.Pool(processes=batch_size) as pool:
            results = list(tqdm(pool.imap(run_experiment, args_list), total=len(args_list), desc=f"Batch {batch_idx+1}"))

        for strategy, coop_data, payoff_data in results:
            all_results[strategy]["cooperation"].append(coop_data)
            all_results[strategy]["payoff"].append(payoff_data)

    # ==== 聚合并绘图 ==== #
    steps = np.arange(500)
    plt.figure(figsize=(12, 5))

    # 合作率图
    plt.subplot(1, 2, 1)
    for strategy in strategies:
        coop_array = np.array(all_results[strategy]["cooperation"])
        coop_mean = np.mean(coop_array, axis=0)
        coop_std = np.std(coop_array, axis=0)
        plt.plot(steps, coop_mean, label=f"{strategy}")
        plt.fill_between(steps, coop_mean - coop_std, coop_mean + coop_std, alpha=0.2)
    plt.title("Cooperation Rate over Time")
    plt.xlabel("Step")
    plt.ylabel("Cooperation Rate")
    plt.legend()

    # 总收益图
    plt.subplot(1, 2, 2)
    for strategy in strategies:
        payoff_array = np.array(all_results[strategy]["payoff"])
        payoff_mean = np.mean(payoff_array, axis=0)
        payoff_std = np.std(payoff_array, axis=0)
        plt.plot(steps, payoff_mean, label=f"{strategy}")
        plt.fill_between(steps, payoff_mean - payoff_std, payoff_mean + payoff_std, alpha=0.2)
    plt.title("Total Payoff over Time")
    plt.xlabel("Step")
    plt.ylabel("Total Payoff")
    plt.legend()

    plt.tight_layout()
    plt.show()

# ==== Windows / 多平台入口保护 ==== #
if __name__ == "__main__":
    main()