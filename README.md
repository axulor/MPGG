# 项目目录结构

```
MPGG/
├── agents/
│   ├── __init__.py
│   └── agent.py
├── envs/
│   ├── __init__.py
│   └── migratory_pgg_env.py
├── run_experiments/
│   └── train_iql.py
├── test_migratory_pgg_env.py
├── .gitignore
└── README.md
```

## 项目说明

- **agents/**: 包含智能体的实现。
  - `agent.py`: 智能体类的定义。
  
- **envs/**: 包含环境的实现。
  - `migratory_pgg_env.py`: 多智能体迁徙公共物品博弈环境的定义。
  
- **run_experiments/**: 包含实验运行脚本。
  - `train_iql.py`: 用于训练独立 Q-learning 智能体的脚本。
  
- **test_migratory_pgg_env.py**: 用于测试环境的脚本。

- **.gitignore**: 指定需要忽略的文件和目录。

- **README.md**: 项目的说明文档。
