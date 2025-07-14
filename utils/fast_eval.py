# fast_eval.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from tqdm import tqdm 

def _t2n(x: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if x is None: return None
    return x.detach().cpu().numpy()

class FastEvaluate:
    def __init__(self,
                all_args: Any, 
                policy: Any,
                eval_envs: Any,
                run_dir: Path,
                ):
        # This __init__ method is already well-designed and does not need changes.
        # It correctly infers all necessary parameters.
        print(f"--- Initializing FastEvaluate Class ---")
        self.all_args = all_args
        self.policy = policy 
        self.eval_envs = eval_envs
        
        if hasattr(all_args, 'device') and isinstance(all_args.device, torch.device):
            self.device = all_args.device
        elif hasattr(policy, 'device') and isinstance(policy.device, torch.device):
            self.device = policy.device
        else:
            self.device = torch.device("cuda" if all_args.cuda and torch.cuda.is_available() else "cpu")
            print(f"Warning: Device not explicitly found in all_args or policy. Defaulting FastEvaluate device to {self.device}")

        if not isinstance(run_dir, Path):
            run_dir = Path(run_dir)
            
        self.plot_save_dir = run_dir / "evaluation_plots"
        self.plot_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_agents = getattr(all_args, 'num_agents')
        self.n_eval_rollout_threads = getattr(all_args, 'n_eval_rollout_threads')
        
        self.eval_rounds = getattr(all_args, 'eval_rounds')
        self.eval_steps_per_round = getattr(all_args, 'eval_steps_per_round')
        
        if self.eval_envs.action_space and isinstance(self.eval_envs.action_space, list) and len(self.eval_envs.action_space) > 0:
            self.action_dim = self.eval_envs.action_space[0].shape[0]
        else:
            try:
                self.action_dim = self.eval_envs.action_space.shape[0]
            except:
                print(f"Warning: Could not determine action_dim from eval_envs.action_space. Type: {type(self.eval_envs.action_space)}. Defaulting to 2.")
                self.action_dim = 2 

        self._baseline_results_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._baselines_evaluated = False

        print(f"  FastEvaluator Config: Rounds={self.eval_rounds}, Steps/Round={self.eval_steps_per_round}, Device={self.device}")
        print(f"  Evaluation plots will be saved to: {self.plot_save_dir}")
        print(f"DEBUG: FastEvaluate INSTANCE CREATED with id: {id(self)}")
        print("--- FastEvaluate Class Initialized ---")

    def get_policy_names(self) -> List[str]:
        # This method is correct.
        names = []
        if self.policy is not None: 
            names.append("marl")
        # names.extend(["random_walk", "static"])
        return list(set(names))

    @torch.no_grad()
    def _run_one_policy_evaluation(self, policy_name: str) -> Dict[str, np.ndarray]:
        """
        [重构后] 使用分批并行的方式高效执行评估。
        """
        # 最终存储所有 round 数据的容器
        all_round_coop_trajectories = np.full((self.eval_rounds, self.eval_steps_per_round), np.nan, dtype=np.float32)
        all_round_reward_trajectories = np.full((self.eval_rounds, self.eval_steps_per_round), np.nan, dtype=np.float32)

        if policy_name == "marl" and self.policy is not None:
            self.policy.prep_evaluating()
        
        # 计算需要多少个批次来完成所有 rounds
        num_batches = int(np.ceil(self.eval_rounds / self.n_eval_rollout_threads))
        
        print(f"    Evaluating policy '{policy_name}':")
        # 使用 tqdm 显示总的批次进度
        for batch_idx in tqdm(range(num_batches), desc=f"      Policy '{policy_name}' Batches", leave=False, ncols=100):
            
            # 1. 重置当前批次的所有并行环境
            obs_np, adj_np = self.eval_envs.reset()
            
            # 2. 在当前批次的所有并行环境上，完整地运行 eval_steps_per_round
            for step_idx in range(self.eval_steps_per_round):
                # --- 动作生成部分 (与您原代码完全一致，无需修改) ---
                actions_to_env_np = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.action_dim), dtype=np.float32)

                if policy_name == "marl":
                    flat_obs_t = torch.from_numpy(obs_np.reshape(-1, obs_np.shape[-1])).float().to(self.device)
                    node_obs_for_gnn_t = torch.from_numpy(obs_np).float().to(self.device)
                    adj_for_gnn_t = torch.from_numpy(adj_np).float().to(self.device)
                    
                    # Simplified agent and env ID generation
                    agent_id_t = torch.arange(self.num_agents, device=self.device).view(1, -1, 1).repeat(self.n_eval_rollout_threads, 1, 1).view(-1, 1)
                    env_id_t = torch.arange(self.n_eval_rollout_threads, device=self.device).view(-1, 1, 1).repeat(1, self.num_agents, 1).view(-1, 1)

                    torch_actions, _ = self.policy.get_actions(
                        flat_obs_t, node_obs_for_gnn_t, adj_for_gnn_t, agent_id_t, env_id_t
                    )
                    actions_from_policy_flat_np = _t2n(torch_actions)
                    if actions_from_policy_flat_np is None: 
                        actions_from_policy_flat_np = np.random.rand(self.n_eval_rollout_threads * self.num_agents, self.action_dim) * 2 - 1
                    
                    actions_to_env_np = actions_from_policy_flat_np.reshape(self.n_eval_rollout_threads, self.num_agents, -1)

                elif policy_name == "random_walk":
                    action_space_sample = self.eval_envs.action_space[0]
                    low, high = action_space_sample.low, action_space_sample.high
                    # 修复后的代码
                    target_shape = (self.n_eval_rollout_threads, self.num_agents, self.action_dim)
                    actions_to_env_np = np.random.uniform(low=low, high=high, size=target_shape).astype(np.float32)
                
                elif policy_name == "static":
                    actions_to_env_np.fill(0.0)
                
                # 3. 与环境交互
                next_obs_np, rewards, next_adj_np, dones, infos = self.eval_envs.step(actions_to_env_np)
                
                # 4. [修改] 将当前批次的结果存入总容器的正确位置
                # --- 数据记录部分 (重构后) ---
                start_round_idx = batch_idx * self.n_eval_rollout_threads
                end_round_idx = start_round_idx + self.n_eval_rollout_threads
                
                # 处理 infos，一次性提取所有并行线程的数据
                batch_coop_rates = [info.get("step_cooperation_rate", np.nan) for info in infos]
                batch_rewards = np.mean(rewards, axis=1).flatten() # (n_eval_rollout_threads,)

                # 将数据填充到总轨迹矩阵的对应行和列
                # 注意：如果最后一个批次不足 n_eval_rollout_threads，只填充有效部分
                actual_threads_in_batch = len(batch_coop_rates)
                effective_end_round_idx = start_round_idx + actual_threads_in_batch
                
                all_round_coop_trajectories[start_round_idx:effective_end_round_idx, step_idx] = batch_coop_rates
                all_round_reward_trajectories[start_round_idx:effective_end_round_idx, step_idx] = batch_rewards
                
                obs_np, adj_np = next_obs_np, next_adj_np
                
                if np.all(dones): break 
        
        # 5. [修改] 对最终结果进行计算 (与原代码逻辑一致，但现在处理的是完整的数据)
        # 确保只对实际运行过的rounds计算均值和方差，防止最后一个批次未满时引入偏差
        valid_rounds_data_coop = all_round_coop_trajectories[:self.eval_rounds]
        valid_rounds_data_reward = all_round_reward_trajectories[:self.eval_rounds]

        mean_coop_curve = np.nanmean(valid_rounds_data_coop, axis=0)
        std_coop_curve = np.nanstd(valid_rounds_data_coop, axis=0)
        mean_reward_curve = np.nanmean(valid_rounds_data_reward, axis=0)
        std_reward_curve = np.nanstd(valid_rounds_data_reward, axis=0)

        policy_eval_results = {
            "cooperation_rate_curve": mean_coop_curve,
            "cooperation_rate_std_curve": std_coop_curve,
            "mean_reward_curve": mean_reward_curve,
            "mean_reward_std_curve": std_reward_curve,
        }
        return policy_eval_results


    def eval_policy(self) -> Dict[str, Any]:
        # This method's logic is about caching and orchestration. It is correct and does not need changes.
        all_policies_results: Dict[str, Dict[str, np.ndarray]] = {}
        policy_names_to_evaluate_this_run = self.get_policy_names()
        print(f"  Starting evaluation for policies: {policy_names_to_evaluate_this_run}...")
        
        if "marl" in policy_names_to_evaluate_this_run:
            print(f"    Re-evaluating MARL policy...")
            results_marl = self._run_one_policy_evaluation("marl")
            all_policies_results["marl"] = results_marl

        baseline_policy_names = ["random_walk", "static"]
        baselines_actually_ran_this_time = False

        for policy_name in baseline_policy_names:
            if policy_name in policy_names_to_evaluate_this_run:
                if not self._baselines_evaluated or policy_name not in self._baseline_results_cache:
                    if not self._baselines_evaluated:
                         print(f"    Evaluating baseline policy '{policy_name}' (first time or cache miss)...")
                    else:
                         print(f"    WARNING: Baseline '{policy_name}' expected in cache but not found! Re-evaluating.")
                    
                    results_baseline = self._run_one_policy_evaluation(policy_name)
                    self._baseline_results_cache[policy_name] = results_baseline
                    all_policies_results[policy_name] = results_baseline
                    baselines_actually_ran_this_time = True
                else:
                    print(f"    Loading baseline policy '{policy_name}' from cache.")
                    all_policies_results[policy_name] = self._baseline_results_cache[policy_name]
        
        if baselines_actually_ran_this_time:
            self._baselines_evaluated = True
            print(f"  DEBUG: Baselines that were run have been cached. self._baselines_evaluated is now True.")
                    
        return all_policies_results

    def plot_results(self, 
                     all_policies_results: Dict[str, Dict[str, np.ndarray]], 
                     eval_episode_num: Optional[int] = None 
                    ):
        # ... (这个函数与你上一版完全相同，用于绘图，不需要修改) ...
        if not all_policies_results:
            print("FastEvaluate: No evaluation results to plot.")
            return

        plot_labels_x_axis = np.arange(1, self.eval_steps_per_round + 1)
        seed = getattr(self.all_args, 'seed', 'unknown')
        
        if eval_episode_num is not None:
            file_suffix = f"ep{eval_episode_num}_s{seed}"
            eval_id_for_title = f"Eval@{eval_episode_num}" 
        else:
            current_time_str = time.strftime("%Y%m%d-%H%M%S")
            file_suffix = f"t{current_time_str}_s{seed}"
            eval_id_for_title = f"Time {current_time_str}"

        colors = {'marl': 'tab:blue', 'random_walk': 'tab:orange', 'static': 'tab:green', 'default': 'tab:grey'}
        linestyles = {'marl': '-', 'random_walk': '--', 'static': ':', 'default': '-.'}
        policy_display_names = {'marl': 'MARL', 'random_walk': 'Random Walk', 'static': 'Static'}

        fig_coop, ax_coop = plt.subplots(figsize=(12, 7))
        has_coop_data = False
        for policy_name, results in all_policies_results.items(): 
            if not results: continue 
            mean_coop = results.get("cooperation_rate_curve")
            std_coop = results.get("cooperation_rate_std_curve")
            if mean_coop is not None and len(mean_coop) == self.eval_steps_per_round:
                if not np.all(np.isnan(mean_coop)):
                    ax_coop.plot(plot_labels_x_axis, mean_coop, 
                                 label=policy_display_names.get(policy_name, policy_name), 
                                 color=colors.get(policy_name, colors['default']), 
                                 linestyle=linestyles.get(policy_name, linestyles['default']), 
                                 linewidth=2)
                    if std_coop is not None and policy_name == "marl" and not np.all(np.isnan(std_coop)):
                        ax_coop.fill_between(plot_labels_x_axis, mean_coop - std_coop, mean_coop + std_coop, 
                                             color=colors.get(policy_name, colors['default']), alpha=0.15)
                    has_coop_data = True
        
        if has_coop_data:
            ax_coop.set_xlabel("Evaluation Steps", fontsize=14)
            ax_coop.set_ylabel("Average Cooperation Rate", fontsize=14)
            title_coop = f"Cooperation Rate (N={self.num_agents}, Eval Steps={self.eval_steps_per_round}, " \
                         f"Avg over {self.eval_rounds} Rnds, ID: {eval_id_for_title})"
            ax_coop.set_title(title_coop, fontsize=16)
            ax_coop.legend(fontsize=12)
            ax_coop.grid(True, linestyle='--', alpha=0.7)
            ax_coop.set_ylim(-0.05, 1.05)
            plt.tight_layout()
            coop_plot_path = self.plot_save_dir / f"coop_rate_{file_suffix}.png"
            plt.savefig(coop_plot_path)
            print(f"Cooperation rate plot saved to: {coop_plot_path}")
        else:
            print("FastEvaluate: No valid cooperation rate data to plot.")
            plt.close(fig_coop)
        
        fig_reward, ax_reward = plt.subplots(figsize=(12, 7))
        has_reward_data = False
        for policy_name, results in all_policies_results.items(): 
            if not results: continue
            mean_reward = results.get("mean_reward_curve")
            std_reward = results.get("mean_reward_std_curve")
            if mean_reward is not None and len(mean_reward) == self.eval_steps_per_round:
                if not np.all(np.isnan(mean_reward)):
                    ax_reward.plot(plot_labels_x_axis, mean_reward, 
                                   label=policy_display_names.get(policy_name, policy_name), 
                                   color=colors.get(policy_name, colors['default']), 
                                   linestyle=linestyles.get(policy_name, linestyles['default']), 
                                   linewidth=2)
                    if std_reward is not None and policy_name == "marl" and not np.all(np.isnan(std_reward)):
                        ax_reward.fill_between(plot_labels_x_axis, mean_reward - std_reward, mean_reward + std_reward, 
                                               color=colors.get(policy_name, colors['default']), alpha=0.15)
                    has_reward_data = True

        if has_reward_data:
            ax_reward.set_xlabel("Evaluation Steps", fontsize=14)
            ax_reward.set_ylabel("Average Mean Reward (per agent)", fontsize=14)
            title_reward = f"Mean Reward (N={self.num_agents}, Eval Steps={self.eval_steps_per_round}, " \
                           f"Avg over {self.eval_rounds} Rnds, ID: {eval_id_for_title})"
            ax_reward.set_title(title_reward, fontsize=16)
            ax_reward.legend(fontsize=12)
            ax_reward.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            reward_plot_path = self.plot_save_dir / f"mean_reward_{file_suffix}.png"
            plt.savefig(reward_plot_path)
            print(f"Mean reward plot saved to: {reward_plot_path}")
        else:
            print("FastEvaluate: No valid mean reward data to plot.")
            plt.close(fig_reward)