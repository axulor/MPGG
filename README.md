# 项目目录结构

```
my_project/
├── envs/
│   └── migratory_pgg_env_v4.py                         # 自定义环境文件 :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
├── algorithms/
│   ├── graph_actor_critic.py                           # Actor/Critic 网络定义 :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
│   ├── graph_mappo.py                                  # MAPPO Trainer 类 :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}
│   ├── graph_MAPPOPolicy.py                            # Policy 包装器 :contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7}
│   └── __init__.py
├── utils/
│   ├── util.py              # 参数校验、学习率衰减等通用函数
│   ├── gnn.py               # GNNBase 与图处理工具
│   ├── mlp.py               # MLPBase
│   ├── rnn.py               # RNNLayer
│   ├── act.py               # ACTLayer
│   ├── graph_buffer.py      # GraphReplayBuffer
│   ├── valuenorm.py         # ValueNorm
│   ├── popart.py            # PopArt 归一化
│   └── __init__.py
├── train/                    # 训练脚本入口
│   ├── train_graph_grmappo.py  # Graph‑MAPPO 的主训练循环
│   ├── train_graph_grippo.py   # 如果需要 IPPO，可照此结构
│   └── args.py               # 命令行参数解析或超参管理
├── configs/
│   └── default_config.yaml   # YAML 格式的实验超参数
├── scripts/
│   └── test_env.py           # 快速检查 env 输出 obs／action 格式
├── results/                  # 存放模型、日志、TensorBoard 数据
├── README.md                 # 项目说明、运行示例
└── requirements.txt          # 依赖列表

```



## 项目说明

- **agents/**: 包含智能体的实现。
  - `agent.py`: 智能体类的定义，负责智能体的状态、动作选择和 Q 表更新。

- **envs/**: 包含环境的实现。
  - `migratory_pgg_env.py`: 多智能体迁徙公共物品博弈环境的定义，提供了智能体交互的环境。

- **runner/**: 包含训练和评估的运行器。
  - `base_runner.py`: 基础运行器类，提供了强化学习训练的基本框架。
  - `lattice_runner.py`: 继承自 `base_runner.py`，实现了具体的训练和评估逻辑。

- **algorithms/**: 包含算法实现。
  - `dqn.py`: DQN 算法的实现。
  - `policy.py`: 策略相关的实现。
  - `basePolicy.py`: 基础策略类的定义。

- **utils/**: 包含工具函数和类。
  - `separated_buffer.py`: 实现了经验回放缓冲区。
  - `util.py`: 各种实用工具函数。

- **run_experiments/**: 包含实验运行脚本。
  - `train_iql.py`: 用于训练独立 Q-learning 智能体的脚本。

- **test_migratory_pgg_env.py**: 用于测试环境的脚本。

- **.gitignore**: 指定需要忽略的文件和目录。

- **README.md**: 项目的说明文档。

## 使用说明

1. **安装依赖**: 请确保安装了 `requirements.txt` 中列出的所有依赖。

   ```bash
   pip install -r requirements.txt
   ```

2. **运行测试**: 使用 `test_migratory_pgg_env.py` 测试环境设置。

   ```bash
   python test_migratory_pgg_env.py
   ```

3. **训练模型**: 使用 `run_experiments/train_iql.py` 进行模型训练。

   ```bash
   python run_experiments/train_iql.py
   ```

4. **查看结果**: 训练过程中生成的日志和模型保存在指定的目录中，可以使用工具（如 TensorBoard）进行可视化。

## 贡献

欢迎对本项目进行贡献！请提交 Pull Request 或报告问题。
