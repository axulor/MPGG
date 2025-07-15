#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MPGG 环境的优化版训练脚本入口
"""
import setproctitle
import numpy as np
from pathlib import Path
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
import os
import sys
from types import SimpleNamespace 
import yaml  

# 将项目根目录添加到 Python 路径
# 假设此脚本位于 "scripts/" 目录下
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 明确导入所需模块
from envs.marl_env import MultiAgentGraphEnv # 导入 MPGG 环境类
from envs.env_wrappers import GraphSubprocVecEnv # 导入并行环境包装器
from runner.graph_mpe_runner import GMPERunner as Runner # 导入图环境 Runner
from utils.util import print_box, print_args # 打印工具

# 在这里指定配置文件名
CONFIG_NAME = "N25_L100.yaml"

def load_config(config_name):
    """从YAML文件加载配置并转换为SimpleNamespace对象"""
    config_path = Path(project_root) / "config" / config_name
    if not config_path.exists():
        print(f"错误: 配置文件不存在于 {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 将字典转换为SimpleNamespace，以保持 all_args.param 的访问方式
    return SimpleNamespace(**config_dict)

# 加载配置
all_args = load_config(CONFIG_NAME)

# 环境创建函数
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
            print(f"(评估环境初始化 rank {rank}) 使用种子: {eval_seed}")            
            eval_env_args = SimpleNamespace(**vars(all_args))
            env = MultiAgentGraphEnv(eval_env_args)
            env.seed(eval_seed)
            return env 
        return init_env

    print(f"  创建 {all_args.n_eval_rollout_threads} 个并行 SubprocVecEnv 用于评估")
    return GraphSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

# 主函数

def main():
    """主训练函数 (优化版，直接使用 all_args 对象)"""

    # 设置设备
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

    # 打印最终配置
    print("--- Final Configuration ---")
    print_args(all_args)
    print("-" * 50)

    # 设置日志目录
    run_dir = (Path(project_root) / "results")
    experiment_name = f"{all_args.user_name}_N{all_args.num_agents}_L{all_args.episode_length}_K{all_args.k_neighbors}_R{all_args.r}"
    run_dir = run_dir / experiment_name
    run_num = 1
    while (run_dir / f"run{run_num}").exists():
        run_num += 1
    run_dir = run_dir / f"run{run_num}"
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    print(f"日志和模型将保存在: {run_dir}")

    # 设置进程标题
    setproctitle.setproctitle(f"{str(run_dir)}") 

    # 设置随机种子
    print(f"设置随机种子: {all_args.seed}")
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # 初始化环境
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

    # 准备 Runner 配置
    print("准备 Runner 配置...")
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,     
        "run_dir": run_dir,
    }

    # 初始化 Runner
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

    # 打印网络结构
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

    # 开始训练
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
        # 结束清理
        print("训练结束或中断，正在关闭环境...")
        try:
            envs.close()
            if all_args.use_eval and eval_envs is not None:
                eval_envs.close()
            print("  环境已关闭。")
        except Exception as e:
            print(f"  关闭环境时出错: {e}")

        # 保存本地日志
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



# 脚本入口 ==
if __name__ == "__main__":
    main()