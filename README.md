# 项目目录结构

```
MPGG/
├── agents/
│   ├── __init__.py
│   └── agent.py
├── envs/
│   ├── __init__.py
│   └── migratory_pgg_env.py
├── runner/
│   ├── __init__.py
│   └── runner.py
├── algorithms/
│   ├── __init__.py
│   ├── dqn.py
│   ├── ppo.py
│   ├── a2c.py
│   └── maddpg.py
├── utils/
│   ├── __init__.py
│   └── util.py
├── experiments/
│   └── experiments_1.py
├── test.py
├── .gitignore
└── README.md
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
