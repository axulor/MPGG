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

# --- 1. 将项目根目录添加到Python路径，确保能找到其他模块 ---
# 这使得脚本可以从任何地方被一键运行
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. 导入项目所需的模块 ---
from envs.marl_env import MultiAgentGraphEnv
from envs.env_wrappers import GraphSubprocVecEnv # [NEW] 导入并行环境包装器
from algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy
from utils.util import print_box

def load_config_for_analysis(config_name, model_dir):
    config_path = Path(project_root) / "config" / config_name
    if not config_path.exists():
        print(f"错误: 配置文件不存在于 {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 覆盖模型路径
    config_dict['model_dir'] = model_dir
    return SimpleNamespace(**config_dict)

class PolicyAnalyzer:
    def __init__(self, args: SimpleNamespace, checkpoint_path: str):
        print_box("Initializing Policy Analyzer")
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.policy = self._load_policy_from_checkpoint(checkpoint_path)
        print(f"Policy successfully loaded from: {checkpoint_path}")

# scripts/analyze_policy.py -> _load_policy_from_checkpoint (Corrected)

    def _load_policy_from_checkpoint(self, checkpoint_path: str) -> GR_MAPPOPolicy:
        """
        Initializes a policy object and loads weights for each sub-module from a checkpoint.
        """
        temp_env = MultiAgentGraphEnv(self.args)
        
        # 1. Create a fresh policy instance with the correct structure
        policy = GR_MAPPOPolicy(
            self.args,
            temp_env.observation_space[0],
            temp_env.node_observation_space[0],
            temp_env.edge_observation_space[0],
            temp_env.action_space[0],
            device=self.device
        )
        temp_env.close()

        # 2. Load the entire checkpoint dictionary
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 3. [FIXED] Load the state_dict for EACH nn.Module sub-component individually
        
        # Use .get() for safety, in case a key is missing in older checkpoints
        if 'actor_state_dict' in checkpoint:
            policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        else:
            print("Warning: 'actor_state_dict' not found in checkpoint.")

        if 'critic_state_dict' in checkpoint:
            # Using strict=False is a good practice here, as it allows loading
            # even if there are minor mismatches (e.g., PopArt vs Linear)
            # though our config should prevent this.
            policy.critic.load_state_dict(checkpoint['critic_state_dict'], strict=False)
        else:
            print("Warning: 'critic_state_dict' not found in checkpoint.")
            
        if hasattr(policy, 'actor_gnn') and 'actor_gnn_state_dict' in checkpoint:
            policy.actor_gnn.load_state_dict(checkpoint['actor_gnn_state_dict'])
        
        if hasattr(policy, 'critic_gnn') and 'critic_gnn_state_dict' in checkpoint:
            policy.critic_gnn.load_state_dict(checkpoint['critic_gnn_state_dict'])
        
        # Set the policy to evaluation mode
        policy.prep_evaluating()
        
        return policy

    @torch.no_grad()
    def _get_actions(self, obs_np: np.ndarray, adj_np: np.ndarray) -> np.ndarray:
        """
        辅助函数，从策略中获取动作。能自动处理单个或并行环境的输入。
        """
        if obs_np.ndim == 2:  # 单个环境: (N, D)
            is_single_env = True
            obs_np_batch = obs_np[np.newaxis, ...]
            adj_np_batch = adj_np[np.newaxis, ...]
        elif obs_np.ndim == 3: # 并行环境: (M, N, D)
            is_single_env = False
            obs_np_batch = obs_np
            adj_np_batch = adj_np
        else:
            raise ValueError(f"Unsupported obs_np dimension: {obs_np.ndim}")

        M, N, D = obs_np_batch.shape
        
        obs_tensor = torch.from_numpy(obs_np_batch).float().to(self.device)
        adj_tensor = torch.from_numpy(adj_np_batch).float().to(self.device)
        
        obs_flat = obs_tensor.view(-1, D)
        agent_id = torch.arange(N, device=self.device).unsqueeze(0).repeat(M, 1).view(-1, 1)
        env_id = torch.arange(M, device=self.device).unsqueeze(1).repeat(1, N).view(-1, 1)
        
        actions_tensor, _ = self.policy.get_actions(obs_flat, obs_tensor, adj_tensor, agent_id, env_id)
        
        actions_out = actions_tensor.cpu().numpy().reshape(M, N, -1)
        
        return actions_out[0] if is_single_env else actions_out

    def generate_representative_gif(self, output_dir: Path, gif_filename: str, num_rounds: int, num_steps: int):
        num_parallel_envs = self.args.n_eval_rollout_threads
        total_runs = num_rounds * num_parallel_envs
        print_box(f"Finding a representative run from {total_runs} total simulations...")

        all_final_rewards, all_initial_seeds = [], []
        
        def get_env_fn(rank, base_seed):
            def _init_fn():
                env_args = SimpleNamespace(**vars(self.args)); env_args.seed = base_seed + rank
                env = MultiAgentGraphEnv(env_args); env.seed(env_args.seed)
                return env
            return _init_fn

        for r_idx in tqdm(range(num_rounds), desc="Simulating rounds in parallel"):
            base_seed = self.args.seed + 10000 + r_idx * num_parallel_envs
            all_initial_seeds.extend(list(range(base_seed, base_seed + num_parallel_envs)))
            eval_envs = GraphSubprocVecEnv([get_env_fn(i, base_seed) for i in range(num_parallel_envs)])
            obs_np, adj_np = eval_envs.reset()
            
            rewards_np = None
            for _ in range(num_steps - 1):
                actions_np = self._get_actions(obs_np, adj_np)
                obs_np, rewards_np, adj_np, dones_np, _ = eval_envs.step(actions_np)
                if np.all(dones_np): break
            
            if rewards_np is not None:
                all_final_rewards.append(np.mean(rewards_np, axis=1))
            eval_envs.close()

        if not all_final_rewards:
            print("Error: No simulation data collected."); return

        all_final_rewards = np.concatenate(all_final_rewards)
        avg_reward = np.mean(all_final_rewards)
        best_run_idx = np.argmin(np.abs(all_final_rewards - avg_reward))
        representative_seed = all_initial_seeds[best_run_idx]
        final_reward_of_best_run = all_final_rewards[best_run_idx]

        print(f"\nFound representative run (index {best_run_idx}) with seed {representative_seed}.")
        print(f"Its final reward {final_reward_of_best_run.item():.4f} is closest to the average {avg_reward:.4f}.")

        snapshot_dir = output_dir / f"snapshots_for_{gif_filename.replace('.gif', '')}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Rendering representative trajectory and saving snapshots to {snapshot_dir}...")
        
        frames = []
        render_env = MultiAgentGraphEnv(self.args)
        render_env.seed(representative_seed)
        obs_list, _, adj_np, _, _ = render_env.reset()
        obs_np = np.array(obs_list)

        for step_t in range(num_steps):
            frames.append(render_env.render(mode='rgb_array'))
            imageio.imwrite(snapshot_dir / f"step_{step_t:04d}.png", frames[-1])
            if step_t == num_steps - 1: break
            
            actions_np = self._get_actions(obs_np, adj_np)
            actions_list = [actions_np[i] for i in range(self.args.num_agents)]
            obs_list, _, adj_np, dones, _ = render_env.step(actions_list)
            obs_np = np.array(obs_list)
            
            if np.all(dones): break
        
        render_env.close()
        
        output_gif_path = output_dir / gif_filename
        imageio.mimsave(output_gif_path, frames, duration=1000/30, loop=0)
        print(f"\nGIF generation complete. Saved to {output_gif_path}")

def main():
    """
    主函数，一键加载模型并生成代表性行为的GIF。
    """
    # --- 1. 定义分析配置 ---
    # [USER ACTION REQUIRED]: 修改为你自己的实验运行文件夹名和模型文件名
    experiment_name = "local_optimized_N100_L800_H64_GNNH64_Ent0.01/run2" 
    model_filename = "checkpoint_ep1300.pt" # 推荐使用 best_model.pt

    # 定义模拟参数
    num_rounds = 10 # 8个并行环境跑10轮，总共80次模拟
    simulation_steps = 800
    parser = argparse.ArgumentParser(description="Analysis script for trained MARL policies.")
    parser.add_argument("--config", type=str, required=True, help="与训练时使用的配置文件名相同 (e.g., N100_L1000.yaml)。")
    parser.add_argument("--model_run_path", type=str, required=True, help="已训练模型的完整运行路径 (e.g., results/your_username/N100_L1000_k4_r4/run1)。")
    args = parser.parse_args()

    # 构造模型目录的完整路径
    model_dir = Path(args.model_run_path) / "models"

    # --- 2. 构建路径和获取配置 ---
    checkpoint_path = Path(project_root) / "results" / experiment_name / "models" / model_filename
    output_dir = Path(project_root) / "analysis_outputs" / experiment_name
    gif_filename = f"behavior_{Path(model_filename).stem}.gif"

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
        return

    analysis_args = load_config_for_analysis(args.config, str(model_dir))

    # --- 3. 执行分析 ---
    analyzer = PolicyAnalyzer(analysis_args, str(checkpoint_path))
    analyzer.generate_representative_gif(
        output_dir=output_dir,
        gif_filename=gif_filename,
        num_rounds=num_rounds,
        num_steps=simulation_steps
    )

if __name__ == '__main__':
    main()