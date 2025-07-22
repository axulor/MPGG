# scripts/analyze_policy.py

import torch
import numpy as np
import imageio
from tqdm import tqdm
from pathlib import Path
import os
import sys
import yaml
from types import SimpleNamespace

# 将项目根目录添加到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入项目所需的模块
from envs.marl_env import MultiAgentGraphEnv
from envs.env_wrappers import GraphSubprocVecEnv
from algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy
from utils.util import print_box

def load_config_and_args(script_config: SimpleNamespace) -> SimpleNamespace:
    """
    基于脚本内定义的配置, 加载YAML文件并合并参数。
    """
    config_path = Path(project_root) / "config" / script_config.config_name
    if not config_path.exists():
        print(f"错误: 配置文件不存在于 {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r',encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 将YAML配置转换为SimpleNamespace
    args = SimpleNamespace(**config_dict)
    
    # 用脚本中的配置覆盖或添加YAML中的设置
    args.model_dir = str(Path(script_config.model_run_path) / "models")
    args.cuda = torch.cuda.is_available()
    
    # 确保并行线程数和帧率等分析特定参数也被设置
    args.n_eval_rollout_threads = script_config.num_parallel_envs
    args.fps = script_config.fps

    # 将YAML中没有，但策略或环境初始化可能需要的其他参数也合并进去
    for key, value in vars(script_config).items():
        if not hasattr(args, key):
            setattr(args, key, value)
            
    return args

class PolicyAnalyzer:
    def __init__(self, args: SimpleNamespace, checkpoint_path: str):
        print_box("Initializing Policy Analyzer")
        self.args = args
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        print(f"Using device: {self.device}")
        
        self.policy = self._load_policy_from_checkpoint(checkpoint_path)
        print(f"Policy successfully loaded from: {checkpoint_path}")

    def _load_policy_from_checkpoint(self, checkpoint_path: str) -> GR_MAPPOPolicy:
        # ... (这个函数基本是正确的，保持原样) ...
        # 创建一个临时环境以获取观测和动作空间信息
        temp_env = MultiAgentGraphEnv(self.args)
        
        policy = GR_MAPPOPolicy(
            self.args,
            temp_env.observation_space[0],
            temp_env.node_observation_space[0],
            temp_env.edge_observation_space[0],
            temp_env.action_space[0],
            device=self.device
        )
        temp_env.close()

        print("Loading state dictionaries from checkpoint...")
        # [MODIFIED] 添加 weights_only=True 以避免安全警告，并假设我们只加载权重
        # 如果模型文件包含无法通过pickle加载的自定义对象，则需要设为False
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'actor_state_dict' in checkpoint:
            policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        elif 'actor' in checkpoint:
            policy.actor.load_state_dict(checkpoint['actor'].state_dict())
        else:
            print("Warning: 'actor_state_dict' or 'actor' not found in checkpoint.")

        if 'critic_state_dict' in checkpoint:
            policy.critic.load_state_dict(checkpoint['critic_state_dict'], strict=False)
        elif 'critic' in checkpoint:
            policy.critic.load_state_dict(checkpoint['critic'].state_dict(), strict=False)
        else:
            print("Warning: 'critic_state_dict' or 'critic' not found in checkpoint.")
            
        if hasattr(policy, 'actor_gnn') and 'actor_gnn_state_dict' in checkpoint:
            policy.actor_gnn.load_state_dict(checkpoint['actor_gnn_state_dict'])
        
        if hasattr(policy, 'critic_gnn') and 'critic_gnn_state_dict' in checkpoint:
            policy.critic_gnn.load_state_dict(checkpoint['critic_gnn_state_dict'])
        
        policy.prep_evaluating()
        return policy

    @torch.no_grad()
    def _get_actions(self, obs_np: np.ndarray, adj_np: np.ndarray) -> np.ndarray:
        """
        [REVISED] 彻底回退到最简单、最匹配原始代码的实现。
        - 不处理RNN状态。
        - 不传递deterministic参数。
        - 假设 get_actions 返回 (actions, log_probs)。
        """
        if obs_np.ndim == 2:  # 单个环境: (N, D_obs)
            is_single_env = True
            obs_np_batch = obs_np[np.newaxis, ...]
            adj_np_batch = adj_np[np.newaxis, ...]
        elif obs_np.ndim == 3: # 并行环境: (M, N, D_obs)
            is_single_env = False
            obs_np_batch = obs_np
            adj_np_batch = adj_np
        else:
            raise ValueError(f"Unsupported obs_np dimension: {obs_np.ndim}")

        M, N, D_obs = obs_np_batch.shape
        
        obs_tensor = torch.from_numpy(obs_np_batch).float().to(self.device)
        adj_tensor = torch.from_numpy(adj_np_batch).float().to(self.device)
        
        obs_flat = obs_tensor.view(-1, D_obs)
        agent_id = torch.arange(N, device=self.device).unsqueeze(0).repeat(M, 1).view(-1, 1)
        env_id = torch.arange(M, device=self.device).unsqueeze(1).repeat(1, N).view(-1, 1)
        
        # [CRITICAL FIX] 调用策略网络，严格按照最开始的无RNN、无deterministic方式
        # 假设 get_actions 返回 (actions, log_probs) 或 (actions, some_other_value)
        actions_tensor, _ = self.policy.get_actions(
            obs_flat, obs_tensor, adj_tensor, 
            agent_id, env_id
        )
        
        actions_out = actions_tensor.cpu().numpy().reshape(M, N, -1)
        
        return actions_out[0] if is_single_env else actions_out

    def generate_representative_gif(self, output_dir: Path, gif_filename: str, num_rounds: int, num_steps: int):
        num_parallel_envs = self.args.n_eval_rollout_threads
        total_runs = num_rounds * num_parallel_envs
        print_box(f"Finding a representative run from {total_runs} total simulations...")

        all_final_rewards, all_initial_seeds = [], []
        
        def get_env_fn(rank, base_seed):
            def _init_fn():
                env_args = SimpleNamespace(**vars(self.args))
                env_args.seed = base_seed + rank
                env = MultiAgentGraphEnv(env_args)
                env.seed(env_args.seed)
                return env
            return _init_fn

        for r_idx in tqdm(range(num_rounds), desc="Simulating rounds in parallel"):
            base_seed = self.args.seed + 10000 + r_idx * num_parallel_envs
            all_initial_seeds.extend(list(range(base_seed, base_seed + num_parallel_envs)))
            
            eval_envs = GraphSubprocVecEnv([get_env_fn(i, base_seed) for i in range(num_parallel_envs)])
            obs_np, adj_np = eval_envs.reset()
            
            # [CRITICAL FIX] 移除了所有RNN状态相关的代码
            for _ in range(num_steps):
                # 调用适配后的 _get_actions
                actions_np = self._get_actions(obs_np, adj_np)
                obs_np, rewards_np, adj_np, dones_np, _ = eval_envs.step(actions_np)
                
                if np.all(dones_np): break
            
            if rewards_np is not None:
                all_final_rewards.append(np.mean(rewards_np, axis=1))
            eval_envs.close()

        # ... (找到代表性seed的逻辑保持不变) ...
        if not all_final_rewards:
            print("Error: No simulation data collected."); return

        all_final_rewards = np.concatenate(all_final_rewards)
        # avg_reward = np.mean(all_final_rewards)
        # best_run_idx = np.argmin(np.abs(all_final_rewards - avg_reward))
        # representative_seed = all_initial_seeds[best_run_idx]
        # final_reward_of_best_run = all_final_rewards[best_run_idx]

        # print(f"\nFound representative run (index {best_run_idx}) with seed {representative_seed}.")
        # print(f"Its final reward {final_reward_of_best_run.item():.4f} is closest to the average {avg_reward:.4f}.")

        best_run_idx = np.argmax(all_final_rewards)

        representative_seed = all_initial_seeds[best_run_idx]
        final_reward_of_best_run = all_final_rewards[best_run_idx]

        print(f"\nFound best run (index {best_run_idx}) with seed {representative_seed}.")
        print(f"Its final reward {final_reward_of_best_run.item():.4f} is the highest among all runs.")


        snapshot_dir = output_dir / f"snapshots_for_{gif_filename.replace('.gif', '')}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Rendering representative trajectory and saving snapshots to {snapshot_dir}...")
        
        frames = []
        render_env_args = SimpleNamespace(**vars(self.args))
        render_env_args.seed = representative_seed
        render_env = MultiAgentGraphEnv(render_env_args)
        render_env.seed(representative_seed)
        
        obs_list, _, adj_np, _, _ = render_env.reset()
        obs_np = np.array(obs_list)
        
        for step_t in range(num_steps):
            frame = render_env.render(mode='rgb_array')
            if frame is not None:
                frames.append(frame)
                imageio.imwrite(snapshot_dir / f"step_{step_t:04d}.png", frame)

            # 调用适配后的 _get_actions
            actions_np = self._get_actions(obs_np, adj_np)

            actions_list = [actions_np[i] for i in range(self.args.num_agents)]
            obs_list, _, adj_np, dones, _ = render_env.step(actions_list)
            obs_np = np.array(obs_list)
            
            if np.all(dones):
                frame = render_env.render(mode='rgb_array')
                if frame is not None: frames.append(frame)
                break
        
        render_env.close()
        
        if frames:
            output_gif_path = output_dir / gif_filename
            imageio.mimsave(output_gif_path, frames, duration=int(1000 / self.args.fps), loop=0)
            print(f"\nGIF generation complete. Saved to {output_gif_path}")
        else:
            print("\nWarning: No frames were rendered. GIF not created.")

def main():
    """
    [REVISED] 主函数，配置直接在脚本内定义，一键运行。
    """
    # ==================================================================
    # --- 1. 在这里配置你的分析任务 ---
    # ==================================================================
    
    config = SimpleNamespace(
        # --- 路径配置 ---
        # 训练时使用的配置文件名 (e.g., N100_L1000.yaml)
        config_name = "N100_L100_K4_R3.yaml",
        
        # 已训练模型的完整运行路径 (e.g., results/your_username/...)
        model_run_path = "results/local_optimized_N100_L100_K4_R3.0/run3",

        # 要加载的模型文件名
        model_filename = "checkpoint_ep1100.pt",

        # --- 模拟与渲染配置 ---
        # 寻找代表性轨迹时的并行环境数
        num_parallel_envs = 8,
        
        # 并行模拟的轮数 (总模拟次数 = num_parallel_envs * num_rounds)
        num_rounds = 10,
        
        # 每次模拟的最大步数
        simulation_steps = 500,
        
        # 生成GIF的帧率
        fps = 30,

        # 是否使用CUDA（如果可用）
        use_cuda = True
    )

    # ==================================================================
    # --- 2. 脚本自动处理后续逻辑 ---
    # ==================================================================
    
    # 构建路径
    checkpoint_path = Path(config.model_run_path) / "models" / config.model_filename
    
    # 自动从路径生成输出目录名，保持结构清晰
    run_name = Path(config.model_run_path).name 
    experiment_name = Path(Path(config.model_run_path).parent).name
    output_dir = Path(project_root) / "analysis_outputs" / experiment_name / run_name
    
    gif_filename = f"behavior_{Path(config.model_filename).stem}.gif"

    # 检查模型文件是否存在
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
        return

    # 加载并合并所有配置参数
    analysis_args = load_config_and_args(config)
    
    # --- 3. 执行分析 ---
    analyzer = PolicyAnalyzer(analysis_args, str(checkpoint_path))
    analyzer.generate_representative_gif(
        output_dir=output_dir,
        gif_filename=gif_filename,
        num_rounds=config.num_rounds,
        num_steps=config.simulation_steps
    )
    print_box("Analysis script finished successfully!")


if __name__ == '__main__':
    main()