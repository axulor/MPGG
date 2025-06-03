#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MPGG 环境的优化版训练脚本入口

此脚本整合了参数配置、环境创建、Runner 初始化和训练流程
所有参数直接在脚本内定义，移除了对 config.py 和 argparse 的依赖
配置基于之前验证可运行的版本。
"""

import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
import os
import sys
from types import SimpleNamespace # 用于创建参数对象

# 将项目根目录添加到 Python 路径
# 假设此脚本位于 "scripts/" 目录下
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 明确导入所需模块
from envs.marl_env import MultiAgentGraphEnv # 导入 MPGG 环境类
from envs.env_wrappers import GraphSubprocVecEnv # 导入并行环境包装器
from runner.graph_mpe_runner import GMPERunner as Runner # 导入图环境 Runner
from utils.util import print_box, print_args # 打印工具

# ==============================================================================
# == 1. 参数配置 (硬编码所有最终参数) ==
# ==============================================================================
# 使用 SimpleNamespace 创建配置对象 all_args，包含所有必需的参数。
all_args = SimpleNamespace(
    # --- 实验标识与基本设置 ---
    user_name="local_optimized",      # MODIFICATION: Changed user_name for new runs
    seed=1,
    cuda=True,
    cuda_deterministic=False,
    n_training_threads=8,
    n_rollout_threads=8,            # 并行环境数 
    num_env_steps=1000000,          # MODIFICATION: Increased total steps for more learning

    # --- 环境特定参数 ---
    num_agents=25,
    world_size=10,
    speed=0.05,
    radius=2.0,
    cost=1.0, 
    r=5.0,
    beta=1.0,
    episode_length=100,             # 原来是 64. 尝试 256, 512, 或 1024.
                                    # 需要确保内存足够存储 (L+1)*N_threads*N_agents*obs_dim 等
    env_max_steps = 2000,           # 环境内部逻辑回合最大步数


    # === 网络结构与特性 ===
    share_policy=True,
    # MODIFICATION: Increased hidden_size
    hidden_size=64,                # 原来是 32. 尝试 64, 128.
    layer_N=2,                      # MODIFICATION: MLP层数可以尝试增加到2
    use_ReLU=True,
    use_orthogonal=True,
    gain=0.01,
    use_feature_normalization=True,
    # MODIFICATION: Temporarily disable PopArt/ValueNorm for debugging stability
    use_popart=False,               # 原来是 True.
    use_valuenorm=True,            # 保持 False.
    split_batch=True,
    max_batch_size=1024,

    # === GNN 相关参数 ===
    use_gnn_policy=True,
    gnn_hidden_size=64,            # 原来是 64. 尝试 128.
    gnn_num_heads=4,
    gnn_concat_heads=True,
    gnn_layer_N=2,
    gnn_use_ReLU=True,
    embed_hidden_size=64,          # 原来是 64. 尝试 128.
    embed_layer_N=1,                # 可以尝试增加到 2
    embed_use_ReLU=True,
    embed_add_self_loop=True,
    max_edge_dist=2.0,
    graph_feat_type="relative",
    actor_graph_aggr="node",
    critic_graph_aggr="global",
    global_aggr_type="mean",
    use_cent_obs=True,


    # === PPO 算法参数 ===
    ppo_epoch=10,                   # PPO 更新时数据重复利用次数
    num_mini_batch=32,               # Mini-batch 数量。如果 episode_length * n_rollout_threads 很大，可以适当增加
                                    # 例如，如果 L=512, N_threads=8, 总样本=4096*num_agents.
                                    # mini_batch_size = 4096*num_agents / 8. 确保 mini_batch_size 合理。
    entropy_coef=0.01,              # 原来是 0.05. 尝试 0.01, 0.005.
    value_loss_coef=1.0,
    lr=1e-4,                        # 学习率可以稍后调整，先看大结构是否稳定
    critic_lr=1e-4,                 # 同上
    clip_param=0.2,
    opti_eps=1e-5,
    max_grad_norm=10.0,
    use_max_grad_norm=True,
    use_clipped_value_loss=True,
    use_gae=True,
    gamma=0.99,
    gae_lambda=0.95,
    use_huber_loss=False,
    huber_delta=10.0,
    weight_decay=0,

    # === 运行参数 ===
    use_linear_lr_decay=True,

    # === 保存与日志 ===
    save_interval=20,               # MODIFICATION: Increased save interval as episodes are longer
    log_interval=5,                 # MODIFICATION: Increased log interval
    global_reset_interval = 5,

    # === 评估参数 ===
    use_eval=True,
    n_eval_rollout_threads=8,       # 评估并行环境数 (可以与训练并行数不同)
    eval_interval=50,              # MODIFICATION: Increased eval interval
    eval_rounds = 80,
    eval_steps_per_round = 500,     # 评估时每轮的步数

    # === 是否加载预训练模型 ===
    model_dir = None, 
)

# ==============================================================================
# == 2. 环境创建函数  ==
# ==============================================================================

def make_train_env(all_args: SimpleNamespace):
    """ 
    创建并行训练环境 
    - Return: 并行环境类 GraphSubprocVecEnv
    
    """
    def get_env_fn(rank: int):
        def init_env():
            current_seed = all_args.seed + rank * 1000 
            print(f"  (训练环境初始化 rank {rank}) 使用种子: {current_seed}")
            env_args = SimpleNamespace(**vars(all_args))
            env = MultiAgentGraphEnv(env_args)
            env.seed(current_seed)
            return env
        return init_env 

    print(f"  创建 {all_args.n_rollout_threads} 个并行 SubprocVecEnv 用于训练...")
    return GraphSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args: SimpleNamespace):
    """ 
    创建并行评估环境 
    - Return: 并行环境类 GraphSubprocVecEnv 
    """
    def get_env_fn(rank: int):
        def init_env():
            eval_seed = all_args.seed * 10000 + rank * 1000 + 100 # Different seed for eval
            print(f"  (评估环境初始化 rank {rank}) 使用种子: {eval_seed}")            
            eval_env_args = SimpleNamespace(**vars(all_args))
            env = MultiAgentGraphEnv(eval_env_args)
            env.seed(eval_seed)
            return env
        return init_env

    print(f"  创建 {all_args.n_eval_rollout_threads} 个并行 SubprocVecEnv 用于评估")
    return GraphSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

# ==============================================================================
# == 3. 主函数 (保持不变) ==
# ==============================================================================

def main():
    """主训练函数 (优化版，直接使用 all_args 对象)"""

    # --- 1. 设置设备 ---
    if all_args.cuda and torch.cuda.is_available():
        print_box("选择使用 GPU...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            print("  启用 CUDA 确定性模式")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print_box("选择使用 CPU...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # --- 2. 打印最终配置 ---
    print("--- Final Configuration ---")
    print_args(all_args)
    print("-" * 50)

    # --- 3. 设置日志目录 ---
    run_dir = (Path(project_root) / "results")
    # MODIFICATION: Add user_name to run_dir path for better organization
    experiment_name = f"{all_args.user_name}_N{all_args.num_agents}_L{all_args.episode_length}_H{all_args.hidden_size}_GNNH{all_args.gnn_hidden_size}_Ent{all_args.entropy_coef}"
    run_dir = run_dir / experiment_name
    run_num = 1
    while (run_dir / f"run{run_num}").exists():
        run_num += 1
    run_dir = run_dir / f"run{run_num}"
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    print(f"日志和模型将保存在: {run_dir}")

    # --- 4. 设置进程标题 ---
    setproctitle.setproctitle(f"{str(run_dir)}") # MODIFICATION: More descriptive process title

    # --- 5. 设置随机种子 ---
    print(f"设置随机种子: {all_args.seed}")
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # --- 6. 初始化环境 ---
    print("初始化训练环境...")
    try:
        envs = make_train_env(all_args)
        print("  训练环境初始化成功.")
    except Exception as e:
        print(f"错误：初始化训练环境失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    eval_envs = None
    if all_args.use_eval:
        print("初始化评估环境...")
        try:
            eval_envs = make_eval_env(all_args) 
            print("  评估环境初始化成功.")
        except Exception as e:
            print(f"警告：初始化评估环境失败: {e}")
            eval_envs = None 

    # --- 7. 准备 Runner 配置 ---
    print("准备 Runner 配置...")
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,     
        "run_dir": run_dir,
    }

    # --- 8. 初始化 Runner ---
    print(f"初始化 Runner ({Runner.__name__})...")
    try:
        runner = Runner(config)
        print("  Runner 初始化成功.")
    except Exception as e:
        print(f"错误：初始化 Runner 失败: {e}")
        import traceback
        traceback.print_exc()
        envs.close()
        if eval_envs: eval_envs.close()
        sys.exit(1)

    # --- 9. 打印网络结构 ---
    try:
        print_box("Actor Network", 80)
        if hasattr(runner, 'policy') and runner.policy and hasattr(runner.policy, 'actor'):
            print(runner.policy.actor)
        else: print("  无法访问 Actor 网络结构。")
        print_box("Critic Network", 80)
        if hasattr(runner, 'policy') and runner.policy and hasattr(runner.policy, 'critic'):
            print(runner.policy.critic)
        else: print("  无法访问 Critic 网络结构。")
    except Exception as e: print(f"  打印网络结构时出错: {e}")

    # --- 10. 开始训练 ---
    print_box("开始训练流程...")
    training_failed = False
    try:
        runner.run()
        print("训练循环正常结束。")
    except KeyboardInterrupt:
        training_failed = True
        print("错误：训练被用户中断 (KeyboardInterrupt)。")
    except Exception as e:
        training_failed = True
        print(f"错误：训练过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- 11. 结束清理 ---
        print("训练结束或中断，正在关闭环境...")
        try:
            envs.close()
            if all_args.use_eval and eval_envs is not None:
                eval_envs.close()
            print("  环境已关闭。")
        except Exception as e:
            print(f"  关闭环境时出错: {e}")

        # --- 12. 保存本地日志 ---
        if hasattr(runner, 'writter') and runner.writter is not None:
            print("正在导出 TensorBoard 日志...")
            try:
                summary_path = str(run_dir / "summary.json")
                runner.writter.export_scalars_to_json(summary_path)
                runner.writter.close()
                print(f"  TensorBoard 摘要已导出到: {summary_path}")
            except Exception as e:
                print(f"  导出 TensorBoard 摘要时出错: {e}")
        elif hasattr(runner, 'log_dir') and runner.log_dir is not None:
            print(f"  本地日志（如果有）保存在: {runner.log_dir}")
        else:
            print("  未找到 TensorBoard writer 或日志目录，跳过摘要导出。")

    print(f"脚本执行完毕。{'训练失败或被中断。' if training_failed else ''}")
    if training_failed:
        sys.exit(1)

# ==============================================================================
# == 4. 脚本入口 ==
# ==============================================================================
if __name__ == "__main__":
    main()