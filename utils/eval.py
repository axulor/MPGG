# utils/eval.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path # 确保导入 Path
from tensorboardX import SummaryWriter # 假设使用 tensorboardX

def _t2n(x: Optional[torch.Tensor]) -> Optional[np.ndarray]: # 允许 x 为 None
    """将 PyTorch Tensor 转换为 NumPy Array。"""
    if x is None: return None
    return x.detach().cpu().numpy()


class Evaluate:
    """
    策略评估类，用于评估 MARL 训练的策略以及基准策略。
    """
    def __init__(self,
                all_args,
                policy, # MARL policy network
                eval_envs,
                run_dir: Path): # 明确 run_dir 类型为 Path
        """
        初始化评估器。
        """
        print("--- Initializing Evaluate Class ---")
        self.all_args = all_args
        self.policy = policy # This is the MARL policy to be evaluated
        self.eval_envs = eval_envs
        
        # Ensure run_dir is a Path object for consistency
        if not isinstance(run_dir, Path):
            run_dir = Path(run_dir)
            
        self.save_dir = run_dir / "evaluation_plots"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_agents = getattr(all_args, 'num_agents')
        self.n_eval_rollout_threads = getattr(all_args, 'n_eval_rollout_threads')
        self.eval_rounds = getattr(all_args, 'eval_rounds')
        self.eval_steps_per_round = getattr(all_args, 'eval_steps_per_round')

        # Store action dim once
        self.action_dim = self.eval_envs.action_space[0].shape[0]

        print(f"  Evaluator Config: Rounds={self.eval_rounds}, Steps/Round={self.eval_steps_per_round}, Agents={self.num_agents}")
        print(f"  Evaluation plots will be saved to: {self.save_dir}")
        print("--- Evaluate Class Initialized ---")

    def eval_policy(self) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        执行所有策略类型的评估。
        返回:
            Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
                键为 "策略名_指标名" (e.g., "marl_cooperation_rate")。
                值为一个元组 (mean_over_rounds, std_over_rounds)。
                std_over_rounds 对基准策略可能为 None。
        """
        if self.policy is not None and hasattr(self.policy, 'actor'):
            self.policy.actor.eval() # Set MARL policy to evaluation mode
        elif "marl" in self.get_policy_names() and self.policy is None :
             print("Warning: MARL policy evaluation requested but policy is not initialized.")


        all_results = {}
        policy_names = self.get_policy_names()

        for policy_name in policy_names:
            if policy_name == "marl" and self.policy is None:
                print(f"Skipping MARL policy evaluation as policy is not initialized.")
                continue
            
            print(f"  Starting evaluation for strategy '{policy_name}'...")
            # Store data for each round for this strategy
            # Each element in these lists will be a list of floats (one per step)
            per_round_coop_rates_trajektoria: List[List[float]] = []
            per_round_mean_rewards_trajektoria: List[List[float]] = []

            for r_idx in range(self.eval_rounds):
                print(f"    Round {r_idx + 1}/{self.eval_rounds} for '{policy_name}'...")
                # Reset eval_envs for each round to ensure independent trials
                # self.current_eval_state is now set inside _run_one_round_steps
                
                round_coop_trajectory, round_reward_trajectory = self._run_one_round_steps(policy_name)
                
                per_round_coop_rates_trajektoria.append(round_coop_trajectory)
                per_round_mean_rewards_trajektoria.append(round_reward_trajectory)
                print(f"      Round {r_idx + 1} completed. Avg Coop: {np.nanmean(round_coop_trajectory):.3f}, Avg Reward: {np.nanmean(round_reward_trajectory):.3f}")

            # Process collected trajectories for this policy
            if per_round_coop_rates_trajektoria: # If any rounds were run
                # Pad trajectories to the same length (self.eval_steps_per_round) with NaNs
                max_len = self.eval_steps_per_round
                
                padded_coop_rates = [traj[:max_len] + [np.nan] * (max_len - len(traj)) if len(traj) < max_len else traj[:max_len] 
                                     for traj in per_round_coop_rates_trajektoria]
                coop_rates_np = np.array(padded_coop_rates)
                
                padded_mean_rewards = [traj[:max_len] + [np.nan] * (max_len - len(traj)) if len(traj) < max_len else traj[:max_len]
                                       for traj in per_round_mean_rewards_trajektoria]
                mean_rewards_np = np.array(padded_mean_rewards)

                avg_coop_over_rounds = np.nanmean(coop_rates_np, axis=0)
                std_coop_over_rounds = np.nanstd(coop_rates_np, axis=0) if policy_name == "marl" else None # STD only for MARL
                all_results[f"{policy_name}_cooperation_rate"] = (avg_coop_over_rounds, std_coop_over_rounds)

                avg_reward_over_rounds = np.nanmean(mean_rewards_np, axis=0)
                std_reward_over_rounds = np.nanstd(mean_rewards_np, axis=0) if policy_name == "marl" else None # STD only for MARL
                all_results[f"{policy_name}_mean_reward"] = (avg_reward_over_rounds, std_reward_over_rounds)
            
            print(f"  Strategy '{policy_name}' evaluation completed.")

        return all_results

    def get_policy_names(self) -> List[str]:
        """Returns a list of policy names to evaluate."""
        # Can be made more dynamic if needed
        return ["marl", "random_walk", "static"]

    @torch.no_grad()
    def _run_one_round_steps(self, policy_name: str) -> Tuple[List[float], List[float]]:
        """
        辅助函数：针对一种策略运行 eval_steps_per_round 步。
        """
        round_step_avg_coop_rates: List[float] = []
        round_step_avg_rewards: List[float] = []

        # Reset is crucial here for each independent evaluation round
        obs, agent_id, node_obs, adj = self.eval_envs.reset()

        for step_idx in range(self.eval_steps_per_round):
            actions_np_batch = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.action_dim), dtype=np.float32)

            if policy_name == "marl":
                if self.policy is None: # Should have been caught earlier, but as a safeguard
                    raise ValueError("MARL policy requested for evaluation but not provided/initialized.")
                
                obs_input = np.concatenate(obs)
                node_obs_input = np.concatenate(node_obs)
                adj_input = np.concatenate(adj)
                agent_id_input = np.concatenate(agent_id)

                action_out = self.policy.act(obs_input, node_obs_input, adj_input, agent_id_input)
                # Assuming self.policy.act returns only actions tensor now
                actions_np_flat = _t2n(action_out) 
                if actions_np_flat is None : # Should not happen if policy.act is correct
                    print(f"Warning: policy.act returned None for MARL at step {step_idx}")
                    actions_np_flat = np.random.rand(self.n_eval_rollout_threads * self.num_agents, self.action_dim) * 2 - 1 # Fallback random

                actions_np_batch = actions_np_flat.reshape(self.n_eval_rollout_threads, self.num_agents, -1)
            
            elif policy_name == "random_walk":
                for i in range(self.n_eval_rollout_threads):
                    for j in range(self.num_agents):
                        agent_action_space = self.eval_envs.action_space[j] # Assumes action_space is a list
                        actions_np_batch[i, j] = agent_action_space.sample()
            
            elif policy_name == "static":
                actions_np_batch.fill(0.0)
            
            else:
                raise ValueError(f"Unknown strategy type: {policy_name}")

            next_obs, next_agent_id, next_node_obs, next_adj, rewards, dones, infos = self.eval_envs.step(actions_np_batch)
            
            # Aggregate metrics for this step
            current_step_mean_reward_across_threads = np.mean(rewards) # Mean over threads and agents
            round_step_avg_rewards.append(current_step_mean_reward_across_threads)

            current_step_coop_rates_in_threads: List[float] = []
            for thread_idx in range(self.n_eval_rollout_threads):
                if infos[thread_idx] and len(infos[thread_idx]) > 0 and isinstance(infos[thread_idx][0], dict):
                    coop_rate_for_thread = infos[thread_idx][0].get("step_cooperation_rate")
                    if coop_rate_for_thread is not None:
                        current_step_coop_rates_in_threads.append(coop_rate_for_thread)
            
            round_step_avg_coop_rates.append(np.mean(current_step_coop_rates_in_threads) if current_step_coop_rates_in_threads else np.nan)

            obs, agent_id, node_obs, adj = next_obs, next_agent_id, next_node_obs, next_adj
            
            # If all parallel environments are done (e.g. hit env_max_steps within one), break early for this round
            if np.all(dones):
                # print(f"    Note: All eval environments finished at step {step_idx+1}/{self.eval_steps_per_round} for policy '{policy_name}'.")
                break 
                
        return round_step_avg_coop_rates, round_step_avg_rewards

    def plot_results(self, results: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]):
        """
        根据评估结果绘制并保存曲线图。
        """
        if not results:
            print("No evaluation results to plot.")
            return

        steps_axis = np.arange(self.eval_steps_per_round) # From 0 to N-1 for indexing
        plot_labels = np.arange(1, self.eval_steps_per_round + 1) # For plot x-axis display
        seed = getattr(self.all_args, 'seed', 'unknown')
        
        # Helper to plot mean and std
        def plot_with_std(ax, x_axis_plot, mean_data, std_data, label, color, linestyle, alpha_fill=0.15):
            line, = ax.plot(x_axis_plot, mean_data, label=label, color=color, linestyle=linestyle, linewidth=2)
            if std_data is not None:
                ax.fill_between(x_axis_plot, mean_data - std_data, mean_data + std_data, color=color, alpha=alpha_fill)
            return line

        # Colors and linestyles
        colors = {'marl': 'tab:blue', 'random_walk': 'tab:orange', 'static': 'tab:green'}
        linestyles = {'marl': '-', 'random_walk': '--', 'static': ':'}

        # --- Plot Cooperation Rate ---
        fig_coop, ax_coop = plt.subplots(figsize=(12, 7))
        plot_lines_coop = []
        
        for policy_prefix in self.get_policy_names():
            result_key = f"{policy_prefix}_cooperation_rate"
            if result_key in results and results[result_key][0] is not None:
                mean_data, std_data = results[result_key]
                if len(mean_data) == self.eval_steps_per_round: # Ensure data length matches axis
                    line = plot_with_std(ax_coop, plot_labels, mean_data, std_data, 
                                        label=policy_prefix.replace("_", " ").title(), 
                                        color=colors.get(policy_prefix, 'black'),
                                        linestyle=linestyles.get(policy_prefix, '-'))
                    plot_lines_coop.append(line)

        if plot_lines_coop:
            ax_coop.set_xlabel("Evaluation Steps", fontsize=14)
            ax_coop.set_ylabel("Average Cooperation Rate", fontsize=14)
            ax_coop.set_title(f"Cooperation Rate (N={self.num_agents}, Eval Steps={self.eval_steps_per_round}, Avg over {self.eval_rounds} Rnds, Seed {seed})", fontsize=16)
            ax_coop.legend(fontsize=12)
            ax_coop.grid(True, linestyle='--', alpha=0.7)
            ax_coop.set_ylim(-0.05, 1.05)
            plt.tight_layout()
            coop_plot_path = self.save_dir / f"cooperation_rate_s{seed}.png"
            plt.savefig(coop_plot_path)
            print(f"Cooperation rate plot saved to: {coop_plot_path}")
        plt.close(fig_coop)

        # --- Plot Mean Reward ---
        fig_reward, ax_reward = plt.subplots(figsize=(12, 7))
        plot_lines_reward = []

        for policy_prefix in self.get_policy_names():
            result_key = f"{policy_prefix}_mean_reward"
            if result_key in results and results[result_key][0] is not None:
                mean_data, std_data = results[result_key]
                if len(mean_data) == self.eval_steps_per_round:
                    line = plot_with_std(ax_reward, plot_labels, mean_data, std_data,
                                         label=policy_prefix.replace("_", " ").title(),
                                         color=colors.get(policy_prefix, 'black'),
                                         linestyle=linestyles.get(policy_prefix, '-'))
                    plot_lines_reward.append(line)
        
        if plot_lines_reward:
            ax_reward.set_xlabel("Evaluation Steps", fontsize=14)
            ax_reward.set_ylabel("Average Mean Reward (per agent)", fontsize=14) # Clarified y-axis
            ax_reward.set_title(f"Mean Reward (N={self.num_agents}, Eval Steps={self.eval_steps_per_round}, Avg over {self.eval_rounds} Rnds, Seed {seed})", fontsize=16)
            ax_reward.legend(fontsize=12)
            ax_reward.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            reward_plot_path = self.save_dir / f"mean_reward_s{seed}.png"
            plt.savefig(reward_plot_path)
            print(f"Mean reward plot saved to: {reward_plot_path}")
        plt.close(fig_reward)
        
        return fig_coop, fig_reward # Return figure objects for potential TB logging

    def log_data(self, 
                 eval_results: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]], 
                 writer: Optional[SummaryWriter], # writer can be None
                 total_num_steps: int,
                 figures: Optional[Tuple[plt.Figure, plt.Figure]] = None): # Pass figures
        """
        记录评估结果到控制台，以及可选的 TensorBoard writer。
        """
        print("\n--- Evaluation Summary ---")
        if not eval_results:
            print("  No evaluation results to log.")
            return

        for policy_key, (mean_series, std_series) in eval_results.items():
            # policy_key is like "marl_cooperation_rate"
            if mean_series is not None:
                # Calculate the mean of the means (average performance over the eval_steps_per_round)
                overall_mean_performance = np.nanmean(mean_series)
                print(f"  Metric '{policy_key}': Overall Mean = {overall_mean_performance:.4f}")
                
                if writer is not None and total_num_steps is not None:
                    # Log this single scalar summary value to TensorBoard
                    tb_tag = f"eval_summary/{policy_key}_overall_mean"
                    writer.add_scalar(tb_tag, overall_mean_performance, total_num_steps)
            else:
                print(f"  Metric '{policy_key}': No data.")
        
        if writer is not None and figures is not None and total_num_steps is not None:
            fig_coop, fig_reward = figures
            if fig_coop is not None : # Check if figure was actually created
                writer.add_figure("eval_plots/Cooperation_Rate_Evolution", fig_coop, global_step=total_num_steps)
            if fig_reward is not None :
                writer.add_figure("eval_plots/Mean_Reward_Evolution", fig_reward, global_step=total_num_steps)
            print("  Evaluation plots have been attempted to be logged to TensorBoard.")
            
        print("--- Evaluation Summary End ---\n")