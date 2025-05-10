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
import os
import sys
from types import SimpleNamespace # 用于创建参数对象

# 将项目根目录添加到 Python 路径
# 假设此脚本位于 "scripts/" 目录下
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 明确导入所需模块
from envs.marl_env import MultiAgentGraphEnv # 导入 MPGG 环境类
from envs.env_wrappers import GraphDummyVecEnv, GraphSubprocVecEnv # 导入两种 VecEnv 环境包装器
from runner.graph_mpe_runner import GMPERunner as Runner # 导入图环境 Runner
from utils.util import print_box, print_args # 打印工具

# ==============================================================================
# == 1. 参数配置 (硬编码所有最终参数) ==
# ==============================================================================
# 使用 SimpleNamespace 创建配置对象 all_args，包含所有必需的参数。
# 值基于 .bat 脚本指定 和 config.py 默认值，并应用了之前的关键修正。
all_args = SimpleNamespace(
    # --- 实验标识与基本设置 ---
    env_name="GraphMPE",            # 环境名称 (固定为我们使用的图环境接口)
    user_name="local",              # 用户名 (用于本地路径)
    seed=1,                         # 随机种子
    cuda=True,                      # 尝试使用 GPU
    cuda_deterministic=False,       # 不启用 CUDA 确定性 (为了速度)
    n_training_threads=2,           # PyTorch 训练线程数
    n_rollout_threads=1,            # 并行环境数 (固定为 1)
    n_eval_rollout_threads=1,       # 评估环境数 (固定为 1)
    num_env_steps=1000000,            # 总训练环境步数
    verbose=True,                   # 启用详细信息打印

    # --- 环境特定参数 (MPGG) ---
    # 确保这些参数与 MultiAgentGraphEnv 初始化和内部逻辑一致
    num_agents=100,                 # 智能体数量
    world_size=20.0,             # MPGG 世界大小
    speed=0.1,                    # MPGG 智能体移动速度
    radius=2.0,                  # MPGG 交互半径
    cost=1.0,                     # MPGG 合作成本
    r=4.0,                        # MPGG PGG 乘数因子 (示例值)
    beta=1.0,                     # MPGG Fermi 噪声参数 (示例值)
    max_cycles=50,                # MPGG 环境最大步数
    discrete_action = False,      # 默认为连续动作
    episode_length=50,           # Replay Buffer 中存储的时序轨迹的长度  


    # === Replay Buffer / Rollout 参数 ===

    # === 网络结构与特性 ===
    share_policy=True,              # 智能体共享策略网络
    use_centralized_V=True,         # 使用中心化 Critic (CTDE)
    hidden_size=64,                 # Actor/Critic MLP 隐藏层维度
    layer_N=1,                      # Actor/Critic MLP 层数
    use_ReLU=True,                  # 使用 ReLU 激活函数
    use_orthogonal=True,            # 使用正交初始化
    gain=0.01,                      # 正交初始化增益
    use_feature_normalization=True, # 输入特征使用 LayerNorm
    use_popart=True,                # 使用 PopArt
    use_valuenorm=False,            # 不使用 ValueNorm
    split_batch=True,               # 是否在前向传播时拆分大 batch
    max_batch_size=1024,            # 拆分 batch 的大小 (如果 split_batch=True)

    # === GNN 相关参数 ===
    use_gnn_policy=True,            # 策略网络使用 GNN
    gnn_hidden_size=64,             # GNN 隐藏层维度
    gnn_num_heads=4,                # GNN Transformer 注意力头数
    gnn_concat_heads=True,          # GNN 是否拼接注意力头输出
    gnn_layer_N=2,                  # GNN 层数
    gnn_use_ReLU=True,              # GNN 层是否使用 ReLU
    embed_hidden_size=64,           # Embedding 后 MLP 隐藏层大小
    embed_layer_N=1,                # Embedding 后 MLP 层数
    embed_use_ReLU=True,            # Embedding 后 MLP 是否用 ReLU
    embed_add_self_loop=True,       # GNN 是否添加自环
    max_edge_dist=2.0,              # GNN 构建边的最大距离 (应等于 MPGG radius)
    graph_feat_type="relative",     # GNN 节点特征类型 (来自原始可运行配置)
    actor_graph_aggr="node",        # Actor GNN 聚合方式 (来自原始可运行配置)
    critic_graph_aggr="global",     # Critic 使用全局聚合
    global_aggr_type="mean",        # 指定全局聚合类型
    use_cent_obs=True,              # Critic MLP 是否额外接收中心化观测


    # === PPO 算法参数 ===
    ppo_epoch=10,                   # PPO 更新迭代次数
    num_mini_batch=4,               # Mini-batch 数量 (1 表示不拆分)
    entropy_coef=0.01,              # 熵正则化系数
    value_loss_coef=1.0,            # 值函数损失系数
    lr=7e-4,                        # Actor 学习率
    critic_lr=7e-4,                 # Critic 学习率
    clip_param=0.2,                 # PPO 裁剪范围
    opti_eps=1e-5,                  # Adam/RMSprop epsilon
    max_grad_norm=10.0,             # 最大梯度范数
    use_max_grad_norm=True,         # 启用梯度裁剪
    use_clipped_value_loss=True,    # 启用裁剪的值损失
    use_gae=True,                   # 启用 GAE
    gamma=0.99,                     # 折扣因子
    gae_lambda=0.95,                # GAE lambda 参数
    use_proper_time_limits=False,   # GAE 是否考虑 episode 结束
    use_huber_loss=False,           # 不使用 Huber Loss
    huber_delta=10.0,               # Huber loss delta (如果使用)
    weight_decay=0,                 # 权重衰减

    # === 运行参数 ===
    use_linear_lr_decay=True,       # 使用学习率线性衰减

    # === 保存与日志 ===
    save_interval=10,               # 模型保存频率 (按训练次数或回合数，取决于 runner 实现)
    log_interval=1,                 # 日志打印频率 (按训练次数或回合数)

    # === 评估参数 ===
    use_eval=False,                  # 启用周期性评估
    eval_interval=20,               # 评估频率 (按训练次数或回合数)
    eval_episodes=32,               # 每次评估运行的回合数

    # === 是否加载预训练模型 ===
    model_dir = None,               #
)

# ==============================================================================
# == 2. 环境创建函数 (保持不变) ==
# ==============================================================================

def make_train_env(all_args: SimpleNamespace):
    """创建训练环境，根据 n_rollout_threads 决定使用 DummyVecEnv 或 SubprocVecEnv。"""
    print(f"[DEBUG make_train_env] n_rollout_threads: {all_args.n_rollout_threads}")
    def get_env_fn(rank: int): # rank 用于为每个并行环境设置不同种子
        """内部函数，返回一个创建单个环境实例的函数。"""
        def init_env():
            # rank 用于确保每个并行环境有不同的初始状态或随机性
            current_seed = all_args.seed + rank * 1000
            print(f"  (训练环境初始化 rank {rank}) 使用种子: {current_seed}")
            env = MultiAgentGraphEnv(all_args)
            env.seed(current_seed)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        print("  创建单线程 DummyVecEnv 用于训练...")
        return GraphDummyVecEnv([get_env_fn(0)])
    else:
        print(f"  创建 {all_args.n_rollout_threads} 个并行 SubprocVecEnv 用于训练...")
        # 创建包含 n_rollout_threads 个环境初始化函数的列表
        return GraphSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args: SimpleNamespace):
    """创建评估环境，根据 n_eval_rollout_threads 决定。"""
    def get_env_fn(rank: int):
        def init_env():
            # 使用与训练环境不同的种子基数
            eval_seed = all_args.seed * 50000 + rank * 10000 + 1
            print(f"  (评估环境初始化 rank {rank}) 使用种子: {eval_seed}")
            env = MultiAgentGraphEnv(all_args)
            env.seed(eval_seed)
            return env
        return init_env

    if all_args.n_eval_rollout_threads == 1:
        print("  创建单线程 DummyVecEnv 用于评估...")
        return GraphDummyVecEnv([get_env_fn(0)])
    else:
        print(f"  创建 {all_args.n_eval_rollout_threads} 个并行 SubprocVecEnv 用于评估...")
        return GraphSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

# ==============================================================================
# == 3. 主函数 (优化版) ==
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
    if all_args.verbose:
        print("--- Final Configuration ---")
        print_args(all_args)
        print("-" * 50)

    # --- 3. 设置日志目录 ---
    # 路径拼接逻辑保持不变
    run_dir = (
        Path(project_root) / "results"
        / all_args.env_name
    )
    run_num = 1
    while (run_dir / f"run{run_num}").exists():
        run_num += 1
    run_dir = run_dir / f"run{run_num}"
    os.makedirs(str(run_dir))
    print(f"日志和模型将保存在: {run_dir}")

    # --- 4. 设置进程标题 ---
    setproctitle.setproctitle(
        f"{all_args.env_name}@{all_args.user_name}"
    )

    # --- 5. 设置随机种子 ---
    print(f"设置随机种子: {all_args.seed}")
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # --- 6. 初始化环境 ---
    print("初始化训练环境...")
    try:
        envs = make_train_env(all_args) # 直接传递 all_args 对象
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
            eval_envs = make_eval_env(all_args) # 直接传递 all_args 对象
            print("  评估环境初始化成功.")
        except Exception as e:
            print(f"警告：初始化评估环境失败: {e}")
            eval_envs = None # 评估失败不影响继续

    # --- 7. 准备 Runner 配置 ---
    print("准备 Runner 配置...")
    # Runner 配置直接使用 all_args 对象和已创建的资源
    config = {
        "all_args": all_args,   # 参数配置
        "envs": envs,           # 训练环境
        "eval_envs": eval_envs, # 评估环境
        "device": device,       # 计算设备     
        "run_dir": run_dir,     # 结果目录
    }

    # --- 8. 初始化 Runner ---
    print(f"初始化 Runner ({Runner.__name__})...")
    try:
        runner = Runner(config)         # 实例化
        print("  Runner 初始化成功.")
    except Exception as e:
        print(f"错误：初始化 Runner 失败: {e}")
        import traceback
        traceback.print_exc()
        envs.close()
        if eval_envs: eval_envs.close()
        sys.exit(1)

    # --- 9. 打印网络结构 ---
    if all_args.verbose:
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
        runner.run()        # 执行训练循环
        print("训练循环正常结束。")
    except Exception as e:
        training_failed = True
        print(f"错误：训练过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc() # 打印详细错误堆栈
    finally:
        # --- 11. 结束清理 (无论训练是否成功都执行) ---
        print("训练结束或中断，正在关闭环境...")
        try:
            envs.close()
            if all_args.use_eval and eval_envs is not None:
                eval_envs.close()
            print("  环境已关闭。")
        except Exception as e:
            print(f"  关闭环境时出错: {e}")

        # --- 12. 保存本地日志 ---
        # 检查 Runner 是否有 writter 属性 (用于 TensorBoard)
        if hasattr(runner, 'writter') and runner.writter is not None:
            print("正在导出 TensorBoard 日志...")
            try:
                # 确保使用 run_dir (Path 对象) 构建路径
                summary_path = str(run_dir / "summary.json")
                runner.writter.export_scalars_to_json(summary_path)
                runner.writter.close()
                print(f"  TensorBoard 摘要已导出到: {summary_path}")
            except Exception as e:
                print(f"  导出 TensorBoard 摘要时出错: {e}")
        # 如果没有 writter，但有 log_dir，打印日志目录
        elif hasattr(runner, 'log_dir') and runner.log_dir is not None:
            print(f"  本地日志（如果有）保存在: {runner.log_dir}")
        else:
            print("  未找到 TensorBoard writer 或日志目录，跳过摘要导出。")

    print(f"脚本执行完毕。{'训练失败。' if training_failed else ''}")
    # 如果训练中途失败，脚本以非零状态码退出
    if training_failed:
        sys.exit(1)

# ==============================================================================
# == 4. 脚本入口 ==
# ==============================================================================
if __name__ == "__main__":
    # 直接调用 main 函数，所有配置已在顶部的 all_args 对象中定义好
    main()