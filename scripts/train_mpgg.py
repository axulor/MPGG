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
from envs.env_wrappers import GraphDummyVecEnv # 使用单线程环境包装器
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
    scenario_name="mpgg_graph",     # 场景名称 (对应我们的 MPGG 实现)
    algorithm_name="rmappo",        # 使用的算法
    experiment_name="win_check_optimized", # 优化后的实验名称
    project_name="win_local",       # 项目名 (用于本地路径)
    user_name="local",              # 用户名 (用于本地路径)
    seed=1,                         # 随机种子
    cuda=True,                      # 尝试使用 GPU
    cuda_deterministic=False,       # 不启用 CUDA 确定性 (为了速度)
    n_training_threads=2,           # PyTorch 训练线程数
    n_rollout_threads=1,            # 并行环境数 (固定为 1)
    n_eval_rollout_threads=1,       # 评估环境数 (固定为 1)
    n_render_rollout_threads=1,     # 渲染环境数 (即使不用也要定义，base_runner 需要)
    num_env_steps=20000,            # 总训练环境步数
    use_wandb=False,                # 禁用 wandb
    verbose=True,                   # 启用详细信息打印

    # --- 环境特定参数 (MPGG) ---
    # 确保这些参数与 MultiAgentGraphEnv 初始化和内部逻辑一致
    num_agents=100,                 # 智能体数量
    world_size=100.0,             # MPGG 世界大小
    speed=1.0,                    # MPGG 智能体移动速度
    radius=10.0,                  # MPGG 交互半径
    cost=1.0,                     # MPGG 合作成本
    r=3.0,                        # MPGG PGG 乘数因子 (示例值)
    beta=0.5,                     # MPGG Fermi 噪声参数 (示例值)
    max_cycles=25,                # MPGG 环境最大步数
    discrete_action = False,      # 默认为连续动作
    episode_length=50,           # Replay Buffer 中存储的时序轨迹的长度  
    # --- 其他场景兼容参数 (设为 MPGG 的合理值或 0) ---
    num_obstacles=0,              # MPGG 无障碍物
    num_scripted_agents=0,        # MPGG 无脚本智能体
    num_landmarks=0,              # MPGG 无地标
    collaborative=True,           # 假设是协作任务目标
    max_speed=1.0,                # 与 MPGG speed 一致

    # === Replay Buffer / Rollout 参数 ===
    data_chunk_length=10,         # RNN 训练时的数据块长度

    # === 网络结构与特性 ===
    share_policy=True,              # 智能体共享策略网络
    use_centralized_V=True,         # 使用中心化 Critic (CTDE)
    hidden_size=64,                 # Actor/Critic MLP/RNN 隐藏层维度
    layer_N=1,                      # Actor/Critic MLP/RNN 层数
    use_ReLU=True,                  # 使用 ReLU 激活函数
    use_orthogonal=True,            # 使用正交初始化
    gain=0.01,                      # 正交初始化增益
    use_feature_normalization=True, # 输入特征使用 LayerNorm
    use_popart=True,                # 修正: 不使用 PopArt
    use_valuenorm=False,            # 修正: 使用 ValueNorm
    stacked_frames=1,               # 通常 GNN 不需要手动堆叠帧
    # use_stacked_frames=False,
    split_batch=False,              # 是否在前向传播时拆分大 batch
    max_batch_size=32,              # 拆分 batch 的大小 (如果 split_batch=True)

    # === GNN 相关参数 ===
    use_gnn_policy=True,          # 策略网络使用 GNN
    gnn_hidden_size=64,           # GNN 隐藏层维度
    gnn_num_heads=4,              # GNN Transformer 注意力头数
    gnn_concat_heads=True,        # GNN 是否拼接注意力头输出
    gnn_layer_N=2,                # GNN 层数
    gnn_use_ReLU=True,            # GNN 层是否使用 ReLU
    num_embeddings=1,             # **关键:** MPGG 只有 Agent 一种实体类型
    embedding_size=32,            # 实体类型嵌入维度
    embed_hidden_size=64,         # Embedding 后 MLP 隐藏层大小
    embed_layer_N=1,              # Embedding 后 MLP 层数
    embed_use_ReLU=True,          # Embedding 后 MLP 是否用 ReLU
    embed_add_self_loop=True,     # GNN 是否添加自环
    max_edge_dist=10.0,           # **关键:** GNN 构建边的最大距离 (应等于 MPGG radius)
    graph_feat_type="relative",   # GNN 节点特征类型 (来自原始可运行配置)
    actor_graph_aggr="node",      # Actor GNN 聚合方式 (来自原始可运行配置)
    critic_graph_aggr="global",   # **关键修正:** Critic 使用全局聚合
    global_aggr_type="mean",      # 指定全局聚合类型
    use_cent_obs=True,            # Critic MLP 是否额外接收中心化观测

    # === 循环策略 (RNN) ===
    use_recurrent_policy=False,   # **修正:** 不使用标准 GRU/LSTM
    use_naive_recurrent_policy=True, # **修正:** 使用 Naive RNN (RMAPPO 需求)
    recurrent_N=1,                # RNN 层数 (Naive RNN 可能不直接用)

    # === PPO 算法参数 ===
    ppo_epoch=10,                 # PPO 更新迭代次数
    num_mini_batch=4,             # Mini-batch 数量 (1 表示不拆分)
    entropy_coef=0.01,            # 熵正则化系数
    value_loss_coef=1.0,          # 值函数损失系数
    lr=7e-4,                      # Actor 学习率
    critic_lr=7e-4,               # Critic 学习率
    clip_param=0.2,               # PPO 裁剪范围
    opti_eps=1e-5,                # Adam/RMSprop epsilon
    max_grad_norm=10.0,           # 最大梯度范数
    use_max_grad_norm=True,       # 启用梯度裁剪
    use_clipped_value_loss=True,  # 启用裁剪的值损失
    use_gae=True,                 # 启用 GAE
    gamma=0.99,                   # 折扣因子
    gae_lambda=0.95,              # GAE lambda 参数
    use_proper_time_limits=False, # GAE 是否考虑 episode 结束
    use_huber_loss=False,         # 不使用 Huber Loss
    use_value_active_masks=False,  # 在值损失中屏蔽 padding 数据
    use_policy_active_masks=False, # 在策略损失中屏蔽 padding 数据
    huber_delta=10.0,             # Huber loss delta (如果使用)
    weight_decay=0,               # 权重衰减

    # === 运行参数 ===
    use_linear_lr_decay=True,    # 不使用学习率线性衰减

    # === 保存与日志 ===
    save_interval=50,             # 模型保存频率 (按训练次数或回合数，取决于 runner 实现)
    log_interval=5,               # 日志打印频率 (按训练次数或回合数)

    # === 评估参数 ===
    use_eval=True,                # 启用周期性评估
    eval_interval=20,             # 评估频率 (按训练次数或回合数)
    eval_episodes=32,             # 每次评估运行的回合数

    # === 渲染参数 (禁用) ===
    save_gifs=False,
    use_render=False,
    render_episodes=5,
    ifi=0.1,
    render_eval=False,

    # === 预训练模型 ===
    model_dir = None,               # 不加载预训练模型

    # === 其他兼容性参数 ===
    use_obs_instead_of_state=False # Critic 是否用 obs 代替 state (通常 False for MAPPO)
)

# ==============================================================================
# == 2. 环境创建函数 (保持不变) ==
# ==============================================================================

def make_train_env(all_args: SimpleNamespace):
    """创建单线程训练环境。"""
    def get_env_fn():
        """内部函数，用于延迟创建环境实例。"""
        print(f"  (环境初始化) 使用种子: {all_args.seed}")
        env = MultiAgentGraphEnv(all_args)  # 实例化环境
        env.seed(all_args.seed)             # 设置环境种子
        return env
    # 将创建函数包装在列表中传递给 DummyVecEnv
    return GraphDummyVecEnv([get_env_fn])

def make_eval_env(all_args: SimpleNamespace):
    """创建单线程评估环境。"""
    def get_env_fn():
        """内部函数，用于延迟创建评估环境实例。"""
        # 使用与训练环境不同的种子
        eval_seed = all_args.seed * 50000 + 1
        print(f"  (评估环境初始化) 使用种子: {eval_seed}")
        env = MultiAgentGraphEnv(all_args)  # 实例化环境
        env.seed(eval_seed)                 # 设置环境种子
        return env
    # 将创建函数包装在列表中传递给 DummyVecEnv
    return GraphDummyVecEnv([get_env_fn])

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
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    run_num = 1
    while (run_dir / f"run{run_num}").exists():
        run_num += 1
    run_dir = run_dir / f"run{run_num}"
    os.makedirs(str(run_dir))
    print(f"日志和模型将保存在: {run_dir}")

    # --- 4. 设置进程标题 ---
    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.scenario_name}@{all_args.user_name}"
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

    # --- 9. (可选) 打印网络结构 ---
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

        # --- 12. 本地日志收尾 ---
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