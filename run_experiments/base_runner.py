import wandb
import io
import os
from tensorboardX import SummaryWriter
from utils.util import linear_schedule_to_0,linear_schedule_to_1, round_up, FileManager, find_latest_file
import time, sys
import torch
import numpy as np
from stable_baselines3.common.type_aliases import (
    MaybeCallback,
    TrainFreq,
    TrainFrequencyUnit,
)
from typing import (
    Union,
)
import pathlib
from stable_baselines3.common.callbacks import ConvertCallback
from utils.callback import CheckpointCallback, BaseCallback
from torchinfo import summary
from stable_baselines3.common.save_util import load_from_pkl
import zipfile


class Runner(object):
    """
    基础的runner类, 用于RL训练
    """

    def __init__(
        self,
        config,
    ):
        self.all_args = config["all_args"] # 配置参数
        self.envs = config["envs"] # 环境
        self.eval_envs = config['eval_envs'] # 评估环境
        self.device = config["device"] # 设备
        self.num_agents = config["num_agents"] # 智能体数量

        # 参数
        self.num_env_steps = self.all_args.num_env_steps # 环境步数
        self.episode_length = self.all_args.episode_length # 每个episode的长度
        self.n_rollout_threads = self.all_args.n_rollout_threads # 训练线程数
        self.n_eval_rollout_threads=self.all_args.n_eval_rollout_threads # 评估线程数
        self.algorithm_name = self.all_args.algorithm_name # 算法名称
        self.experiment_name = self.all_args.experiment_name # 实验名称
        self.env_name = self.all_args.env_name # 环境名称
        self.use_wandb = self.all_args.use_wandb # 是否使用wandb

        self.learning_starts = self.all_args.learning_starts # 学习开始步数

        # 训练频率参数, 稍后转换为TrainFreq对象
        train_freq = round_up(self.all_args.train_freq / self.n_rollout_threads, 0)
        self.train_freq = (train_freq, self.all_args.freq_type)
        # 将训练频率参数转换为TrainFreq对象
        self._convert_train_freq()

        # 设置学习率 (schedule或float)
        if self.all_args.use_linear_lr_decay:
            self.lr = linear_schedule_to_0(self.all_args.lr)
        else:
            self.lr = self.all_args.lr
        if self.all_args.use_linear_beta_growth:
            self.beta = linear_schedule_to_1(self.all_args.prioritized_replay_beta)
        else:
            self.beta = self.all_args.prioritized_replay_beta

        # 间隔
        self.log_interval = self.all_args.log_interval # 日志间隔
        self.video_interval = self.all_args.video_interval # 视频间隔
        self.save_interval = self.all_args.save_interval # 保存间隔
        self.verbose = 2 # 详细程度
        self.use_eval = self.all_args.use_eval # 是否使用评估
        self.eval_interval = self.all_args.eval_interval # 评估间隔


        # dir
        self.model_dir = self.all_args.model_dir

        print("===================")
        if self.all_args.train_pattern =='seperate': # 分离训练模式
            print("Strategy observation_space: ", self.envs.observation_spaces["agent_0"]) # 策略观察空间
            print("Interaction observation_space: ", self.envs.interact_observation_spaces["agent_0"]) # 交互观察空间
            print("Strategy action_space: ", self.envs.action_spaces["agent_0"][0]) # 策略动作空间
            print("Interaction action_space: ", self.envs.action_spaces["agent_0"][1]) # 交互动作空间
        else: # 联合训练模式
            print("observation_space(together): ", self.envs.observation_spaces["agent_0"]) # 观察空间
            if self.all_args.train_pattern =='together': # 联合训练模式
                print("action_space(together): ", self.envs.action_spaces["agent_0"][2]) # 动作空间
            else:  # train_pattern =='strategy'
                print("action_space(strategy): ", self.envs.action_spaces["agent_0"][0]) # 动作空间


        # 根据配置选择训练算法和动作策略
        if self.all_args.algorithm_name == "DQN": # DQN算法
            from algorithms.dqn.dqn_trainer import Strategy_DQN as TrainAlgo # 策略DQN训练器
            from algorithms.dqn.policy import DQN_Policy as Policy # DQN策略

            if self.all_args.replay_scheme == "uniform": # 均匀重放
                from utils.separated_buffer import SeparatedReplayBuffer as ReplayBuffer # 分离重放缓冲区
            else: # 优先重放
                from utils.separated_buffer import (
                    PrioritizedReplayBuffer as ReplayBuffer,
                )
        else: # 其他算法
            from algorithms.dqn.dqn_trainer import Strategy_DQN as TrainAlgo # 策略DQN训练器
            from algorithms.dqn.policy import DQN_Policy as Policy # DQN策略
            from utils.separated_buffer import SeparatedRolloutBuffer as ReplayBuffer # 分离重放缓冲区

        self._setup_learn(config) # 设置学习

        self.trainer = [] # 训练器
        self.iteract_trainer=[] # 交互训练器
        self.buffer = [] # 缓冲区
        self.interact_buffer = [] # 交互缓冲区

        # 即使策略从文件加载，仍然需要初始化训练器
        for agent_id in range(self.num_agents):
            # 为每个 agent 实例化 TrainAlgo类
            tr = TrainAlgo(
                all_args=self.all_args, # 所有参数      
                logger=self.logger, # 日志记录器
                env=self.envs, # 环境
                gamma=self.all_args.gamma, # 折扣因子
                policy_class=Policy, # 策略类
                learning_rate=self.lr, # 学习率
                prioritized_replay_beta=self.beta, # 优先重放系数
                prioritized_replay_eps=self.all_args.prioritized_replay_eps, # 优先重放epsilon
                exploration_fraction=self.all_args.exploration_fraction, # 探索比例
                exploration_final_eps=self.all_args.strategy_final_exploration, # 策略最终探索epsilon
                device=self.device, # 设备
                action_flag=0 if self.all_args.train_pattern=='strategy' or self.all_args.train_pattern=='seperate' else 2 # 动作标志
            )
            self.trainer.append(tr) # 合并训练器

            if self.all_args.train_pattern == "seperate": # 分离训练模式
                iteract_tr=TrainAlgo(
                    all_args=self.all_args, # 所有参数
                    logger=self.logger, # 日志记录器
                    env=self.envs, # 环境
                    gamma=self.all_args.gamma, # 折扣因子
                    policy_class=Policy, # 策略类
                    learning_rate=self.lr, # 学习率
                    prioritized_replay_beta=self.beta, # 优先重放系数
                    prioritized_replay_eps=self.all_args.prioritized_replay_eps, # 优先重放epsilon
                    exploration_fraction=self.all_args.exploration_fraction, # 探索比例
                    exploration_final_eps=self.all_args.insteraction_final_exploration, # 交互最终探索epsilon       
                    device=self.device, # 设备
                    action_flag=1 # 动作标志
                )
                self.iteract_trainer.append(iteract_tr) # 交互训练器

            
            

        # 如果指定了模型目录，则恢复一个预训练的模型
        have_load_buffer=False # 是否加载缓冲区 
        if self.model_dir is not None: # 如果指定了模型目录
            have_load_buffer=self.restore() # 恢复模型
        
        # 评估过程不需要重放缓冲区
        if not have_load_buffer: # 如果未加载缓冲区
            for agent_id in range(self.num_agents): # 遍历所有智能体
                bu = ReplayBuffer(
                    self.all_args, # 所有参数
                    self.envs.observation_spaces["agent_{}".format(agent_id)], # 观察空间
                    device=self.device, # 设备
                )
                self.buffer.append(bu) # 缓冲区

                if self.all_args.train_pattern == "seperate": # 分离训练模式
                    interact_bu=ReplayBuffer(
                    self.all_args, # 所有参数
                    self.envs.interact_observation_spaces["agent_{}".format(agent_id)], # 交互观察空间
                    device=self.device, # 设备
                )
                    self.interact_buffer.append(interact_bu) # 交互缓冲区
                
        
        print("\nReport Model Structure...") # 报告模型结构
        tensor_date = {} # 张量数据
        for key, value in self.envs.observation_spaces["agent_0"].items(): # 遍历观察空间
            tensor_date[key] = torch.tensor(value.sample(), device=self.device) # 张量数据
        summary(self.trainer[0].policy.q_net, input_data=[tensor_date]) # 总结
        if self.all_args.train_pattern == "seperate": # 分离训练模式
            tensor_date = {} # 张量数据
            for key, value in self.envs.interact_observation_spaces["agent_0"].items(): # 遍历交互观察空间
                tensor_date[key] = torch.tensor(value.sample(), device=self.device) # 张量数据
            # print(tensor_date)
            summary(self.iteract_trainer[0].policy.q_net, input_data=[tensor_date]) # 总结
        print("\nStrat Training...\n") # 开始训练

        # input()

        # 设置回调函数
        callback = None # 回调函数
        if self.save_interval > 0: # 保存间隔大于0  
            callback = CheckpointCallback(
                save_freq=self.save_interval, # 保存频率
                save_path=self.save_dir, # 保存路径
                name_prefix=self.experiment_name, # 实验名称
                save_replay_buffer=self.all_args.save_replay_buffer, # 保存重放缓冲区
                max_files=self.all_args.max_files, # 最大文件数
                verbose=2, # 详细程度
            )
            self.buffer_manager = FileManager(
                self.save_dir, max_files=self.all_args.max_files, suffix="pkl" # 文件管理器
            )

        # 创建评估回调函数
        self.callback = self._init_callback(callback)

    def _setup_learn(self, config): # 设置学习
        # 使用wandb进行日志记录
        if self.use_wandb: # 使用wandb
            self.save_dir = str(wandb.run.dir) # 保存路径
            self.run_dir = str(wandb.run.dir) # 运行路径
            self.logger = None # 日志记录器
            self.gif_dir = str(self.run_dir + "/gifs") # 动图路径
            self.plot_dir=str(self.run_dir + "/plots") # 绘图路径
        else: # 手动配置日志记录和模型保存目录
            self.run_dir = config["run_dir"] # 运行路径
            self.log_dir = str(self.run_dir / "logs") # 日志路径
            if not os.path.exists(self.log_dir): # 如果日志路径不存在
                os.makedirs(self.log_dir) # 创建日志路径
            self.logger = SummaryWriter(self.log_dir) # 日志记录器
            self.save_dir = str(self.run_dir / "models") # 保存路径
            if not os.path.exists(self.save_dir): # 如果保存路径不存在  
                os.makedirs(self.save_dir) # 创建保存路径
            self.gif_dir = str(self.run_dir / "gifs") # 动图路径
            self.plot_dir=str(self.run_dir / "plots") # 绘图路径    

        if not os.path.exists(self.gif_dir): # 如果动图路径不存在
            os.makedirs(self.gif_dir) # 创建动图路径

        if not os.path.exists(self.plot_dir): # 如果绘图路径不存在
            os.makedirs(self.plot_dir) # 创建绘图路径

    def run(self):
        raise NotImplementedError # 未实现

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def restore(self):
        """
        Restoring Model
        存储模型
        """
        have_load_buffer=False # 是否加载缓冲区 
        latest_model_file = find_latest_file(self.model_dir, "zip") # 最新模型文件
        self.load_trainer(latest_model_file) # 加载训练器

        if any(file.endswith(".pkl") for file in os.listdir(self.model_dir)): # 如果模型目录中存在pkl文件
            latest_buffer_file = find_latest_file(self.model_dir, "pkl") # 最新缓冲区文件
            self.load_replay_buffer(latest_buffer_file) # 加载缓冲区
            have_load_buffer=True # 加载缓冲区
        elif self.all_args.eval_mode: # 评估模式
            have_load_buffer=True # 加载缓冲区
        return have_load_buffer # 返回是否加载缓冲区

    def collect_rollouts(self):
        """
        收集经验并存储到 ``ReplayBuffer`` 中。
        """
        # 采样动作
        actions,interactions = self.collect()
        
        # one step to environment
        if self.all_args.train_pattern == "together" or self.all_args.train_pattern == "seperate": # 联合或分离训练模式
            combine_action=np.dstack((actions, interactions)) # 合并动作
            next_obs,i_next_obs, rews, terminations, truncations, infos = self.envs.step(combine_action) # 环境步进
        else:
            next_obs, i_next_obs,rews, terminations, truncations, infos = self.envs.step(actions) # 环境步进

        
        # handle `final_observation` for trunction
        # where the next_obs for calculate TD Q-value is differnt then predict action
        real_next_obs = next_obs.copy() # 真实下一个观测
        
        real_next_i_obs = i_next_obs.copy() # 真实下一个交互观测

        for idx, trunc in enumerate(truncations): # 遍历截断
            if trunc: # 如果截断
                real_next_obs[idx] = infos[idx]["final_observation"] # 真实下一个观测
                if self.all_args.train_pattern=='seperate': # 分离训练模式
                    real_next_i_obs[idx] = infos[idx]["final_i_observation"] # 真实下一个交互观测

        data = real_next_obs,real_next_i_obs, rews, terminations, truncations, actions,interactions # 数据

        # insert data into buffer
        self.insert(data, self.obs,self.interact_obs) # 插入数据
        self.obs = next_obs.copy() # 观测
        self.interact_obs = i_next_obs.copy() # 交互观测

        return infos # 信息 

    def train(self):
        """
        Train policies with data in buffer.
        # 使用缓冲区中的数据训练策略
        """
        # print(self.num_timesteps)
        train_infos = [] # 训练信息
        for agent_id in torch.randperm(self.num_agents): # 随机排列智能体
            if self.all_args.train_pattern == 'together': # 联合训练模式
                ti = self.trainer[agent_id].train(
                    batch_size=self.all_args.mini_batch, replay_buffer=self.buffer[agent_id],action_flag=2 # 动作标志
            )
            else: # 只训练策略
                ti = self.trainer[agent_id].train(
                    batch_size=self.all_args.mini_batch, replay_buffer=self.buffer[agent_id],action_flag=0 # 动作标志
                )
            if self.all_args.train_pattern == 'seperate': # train anohter interaction model 
                self.iteract_trainer[agent_id].train(
                    batch_size=self.all_args.mini_batch, replay_buffer=self.interact_buffer[agent_id],action_flag=1 # 动作标志
                )
            if self.all_args.algorithm_name != "DQN": # 如果不是DQN算法
                # if True:
                self.buffer[agent_id].after_update() # 更新缓冲区
            train_infos.append(ti) # 训练信息

        loss_values = np.array( # 损失值
            [entry["train/loss"] for entry in train_infos if "train/loss" in entry] # 训练损失
        )
        coop_reward = np.array( # 合作奖励
            [entry["train/cooperation_reward"] for entry in train_infos], dtype=float # 训练合作奖励
        )
        defect_reward = np.array( # 背叛奖励
            [entry["train/defection_reward"] for entry in train_infos], dtype=float # 训练背叛奖励
        )
        ti = {
            "train/loss": np.mean(loss_values), # 训练损失
            "train/n_updates": ti["train/n_updates"], # 训练更新次数
            "train/lr": ti["train/lr"], # 学习率
            "train/prioritized_replay_beta": ti["train/prioritized_replay_beta"], # 优先重放系数
            "train/cooperation_reward": np.nanmean(coop_reward), # 合作奖励
            "train/defection_reward": np.nanmean(defect_reward), # 背叛奖励
        }
        return ti # 训练信息

    def _dump_logs(self, episode) -> None: # 写入日志
        """
        Write log. 
        """
        log_info = {} # 日志信息
        time_elapsed = max( # 时间流逝
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon # 时间流逝
        )
        # Number of frames per seconds (includes time taken by gradient update) 每秒帧数（包括梯度更新时间）
        fps = int( # 每秒帧数
            (self.num_timesteps - self._num_timesteps_at_start) # 时间流逝
            * self.n_rollout_threads # 滚动线程数
            / time_elapsed # 时间流逝
        )
        log_info["time/fps"] = fps # 每秒帧数
        log_info["time/episode"] = episode # 当前场景

        print(
            "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.all_args.scenario_name, # 场景名称
                self.algorithm_name, # 算法名称
                self.experiment_name, # 实验名称
                episode, # 当前场景
                self.episodes, # 总场景
                self.num_timesteps, # 总时间步
                self.num_env_steps, # 总环境步
                fps, # 每秒帧数
            )
        )

        for k, v in log_info.items(): # 遍历日志信息
            if self.use_wandb: # 使用wandb
                wandb.log({k: v}, step=self.num_timesteps) # 写入wandb
            else:
                self.logger.add_scalars(
                    k, {k: v}, self.num_timesteps * self.n_rollout_threads # 写入日志
                )

    def log_train(self, train_infos): # 写入训练信息
        """
        Log training info. # 写入训练信息
        :param train_infos: (dict) information about training update. # 训练信息
        :param total_num_steps: (int) total number of training env steps. # 总训练环境步
        """
        # print(train_infos)
        for k, v in train_infos.items(): # 遍历训练信息
            if self.use_wandb: # 使用wandb
                wandb.log({k: v}, step=self.num_timesteps) # 写入wandb
            else:
                self.logger.add_scalars(    
                    k, {k: v}, self.num_timesteps * self.n_rollout_threads # 写入日志
                )

    def log_eval(self, train_infos): # 写入评估信息 
        """
        Log training info. # 写入训练信息
        :param train_infos: (dict) information about training update. # 训练信息
        :param total_num_steps: (int) total number of training env steps. # 总训练环境步
        """
        # print(train_infos)
        for k, v in train_infos.items(): # 遍历训练信息
            if self.use_wandb: # 使用wandb
                wandb.log({k: v}, step=self.num_timesteps) # 写入wandb
            else:
                self.logger.add_scalars(
                    k, {k: v}, self.num_timesteps * self.n_eval_rollout_threads # 写入日志
                )


    def log_rollout(self, rollout_info): # 写入滚动信息
        """
        Log rollout info. # 写入滚动信息
        """
        for k, v in rollout_info.items(): # 遍历滚动信息
            if self.use_wandb: # 使用wandb
                wandb.log({k: v}, step=self.num_timesteps) # 写入wandb
            else:
                self.logger.add_scalars(k, {k: v}, self.num_timesteps) # 写入日志

    def print_train(self, train_infos, extra_info): # 打印训练信息
        """
        print train info # 打印训练信息     
        """
        (
            episode_loss, # 损失
            cooperation_reward_during_training, # 合作奖励
            defection_reward_during_training, # 背叛奖励
            episode_exploration_rate, # 探索率
        ) = extra_info
        print("-" * 44) # 打印分隔符
        print("| Payoff/ {:>33}|".format(" " * 10)) # 打印收益
        print(
            "|    Cooperation Episode Payoff  {:>9.4f} |".format(
                train_infos["payoff/cooperation_episode_payoff"] # 合作收益
            )
        )
        print(
            "|    Defection Episode Payoff  {:>11.4f} |".format(
                train_infos["payoff/defection_episode_payoff"] # 背叛收益
            )
        )
        print(
            "|    Average Episode Payoff  {:>13.4f} |".format(
                train_infos["payoff/episode_payoff"] # 平均收益
            )
        )
        print("| Reward/ {:>32} |".format(" " * 10)) # 奖励
        print(
            "|    Cooperation Episode Rewards  {:>8.4f} |".format(
                train_infos["results/coopereation_episode_rewards"] # 合作奖励
            )
        )
        print(
            "|    Defection Episode Rewards  {:>10.4f} |".format(
                train_infos["results/defection_episode_rewards"] # 背叛奖励
            )
        )
        print("| Train/ {:>34}|".format(" " * 10)) # 训练
        print(
            "|    Average Coop Level  {:>17.2f} |".format(
                train_infos["results/episode_cooperation_level"] # 合作水平
            )
        )
        print(
            "|    Final Coop Level  {:>19.2f} |".format(
                train_infos["results/episode_final_cooperation_performance"] # 最终合作水平
            )
        )
        print(
            "|    Termination Proportion  {:>13.2f} |".format(
                train_infos["results/termination_proportion"] # 终止比例
            )
        )
        print(
            "|    Average Coop Reward  {:>16.4f} |".format(
                cooperation_reward_during_training # 合作奖励
            )
        )
        print(
            "|    Average Defect Reward  {:>14.4f} |".format(
                defection_reward_during_training # 背叛奖励
            )
        )
        print("|    Average Train Loss  {:>17.2f} |".format(episode_loss)) # 平均训练损失
        print(
            "|    Average Exploration Rate  {:>11.2f} |".format(
                episode_exploration_rate # 探索率
            )
        )
        print("|    n_updates  {:>26.0f} |".format(train_infos["train/n_updates"])) # 更新次数
        print("|    Learning Rate  {:>22.2f} |".format(train_infos["train/lr"])) # 学习率
        print(
            "|    Prioritized Replay Beta  {:>12.2f} |".format(
                train_infos["train/prioritized_replay_beta"] # 优先重放系数
            )
        )
        print("| Robutness/ {:>30}|".format(" " * 10)) # 鲁棒性
        print(
            "|    Average Coop Robutness  {:>13.2f} |".format(
                train_infos["robutness/average_cooperation_length"] # 平均合作鲁棒性
            )
        )
        print(
            "|    Average Defection Robutness  {:>8.2f} |".format(
                train_infos["robutness/average_defection_length"] # 平均背叛鲁棒性
            )
        )
        print(
            "|    Best Cooperation Robutness  {:>9.2f} |".format(
                train_infos["robutness/best_cooperation_length"], # 最佳合作鲁棒性
            )
        )
        print(
            "|    Best Defection Robutness  {:>11.2f} |".format(
                train_infos["robutness/best_defection_length"] # 最佳背叛鲁棒性
            )
        )        
        print("-" * 44, "\n") # 打印分隔符

    def write_to_video(self, all_frames, episode,video_type='train'):
        """
        记录这个场景并保存gif到本地或wandb
        """
        import imageio

        images = [] # 图像
        for png in all_frames: # 遍历图像
            img = imageio.imread(png) # 读取图像
            images.append(img) # 添加图像
        # print(len(images))
        if self.all_args.use_wandb: # 使用wandb
            import wandb

            imageio.mimsave(
                str(self.gif_dir) + "/episode.gif",
                images,
                duration=self.all_args.ifi,
            )
            wandb.log(
                {
                    video_type: wandb.Video(
                        str(self.gif_dir) + "/episode.gif", fps=4, format="gif"
                    )
                },
                step=self.num_timesteps,
            )

        elif self.all_args.save_gifs: # 保存gif     
            imageio.mimsave(
                str(self.gif_dir) + "/{}_episode_{}.gif".format(video_type,episode),
                images,
                duration=self.all_args.ifi,
            )

    def _convert_train_freq(self) -> None: # 转换训练频率
        """
        Convert `train_freq` parameter (int or tuple)
        to a TrainFreq object.
        """
        if not isinstance(self.train_freq, TrainFreq): # 如果不是TrainFreq对象      
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple): # 如果不是元组
                train_freq = (train_freq, "step") # 训练频率

            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1])) # 训练频率  
            except ValueError as e: # 如果训练频率不是step或episode
                raise ValueError(
                    f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!" # 训练频率不是step或episode
                ) from e

            if not isinstance(train_freq[0], int): # 如果训练频率不是整数
                raise ValueError(
                    f"The frequency of `train_freq` must be an integer and not {train_freq[0]}" # 训练频率不是整数
                )

            self.train_freq = TrainFreq(*train_freq) # 训练频率

    # 初始化回调    
    def _init_callback(
        self,
        callback: MaybeCallback,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert functional callback to object
        if not isinstance(callback, BaseCallback): # 如果不是BaseCallback对象       
            callback = ConvertCallback(callback) # 转换回调

        callback.init_callback(self) # 初始化回调
        return callback # 回调

    # 保存重放缓冲区
    def save_replay_buffer(
        self, path: Union[str, pathlib.Path, io.BufferedIOBase]
    ) -> None:
        """
        保存重放缓冲区为pickle文件。

        :param path: 重放缓冲区保存路径。
            如果path是str或pathlib.Path, 则自动创建路径。
        """
        assert self.buffer is not None, "The replay buffer is not defined" # 重放缓冲区未定义
        self.buffer_manager.create_file(self.buffer, path) # 创建文件

    # 加载重放缓冲区
    def load_replay_buffer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
    ) -> None:
        """
        从pickle文件加载重放缓冲区。

        :param path: 重放缓冲区保存路径。
        """
        self.buffer = load_from_pkl(path, self.verbose) # 加载重放缓冲区
        for b in self.buffer:
            b.device = self.device # 设备

    # 加载训练模型
    def load_trainer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
    ) -> None:
        """
        加载每个智能体的训练模型。

        :param path: 训练模型保存路径。
        """
        # np.random.seed(self.all_args.seed)
        # 打开zip文件
        with zipfile.ZipFile(path, "r") as zipf:
            idx_list = np.arange(self.num_agents) 
            # 使用shuffle()方法
            if self.all_args.eval_mode:
                np.random.shuffle(idx_list)
            for agent_id in idx_list:
                # 提取每个智能体的数据
                agent_data = zipf.read(f"{agent_id}.pt")
                # 使用torch.load加载数据
                pt_data = io.BytesIO(agent_data)
                state_dict = torch.load(pt_data)
                # 使用state_dict初始化每个智能体的训练模型
                self.trainer[agent_id].policy.q_net.load_state_dict(state_dict)
