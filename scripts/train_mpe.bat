@echo off

REM === 切换到项目根目录 ===
cd /d %~dp0
cd ..

REM === 激活 Conda 虚拟环境（请根据实际环境名修改） ===
call activate InforMARL

REM === 参数设置 ===
set ENV=GraphMPE
set SCENARIO=navigation_graph
set NUM_AGENTS=3
set NUM_OBSTACLES=3
set ALGO=rmappo
set EXP=win_check
set SEED_MAX=1

echo env is %ENV%, scenario is %SCENARIO%, algo is %ALGO%, exp is %EXP%, max seed is %SEED_MAX%

FOR /L %%S IN (1,1,%SEED_MAX%) DO (
    echo Running seed %%S...

    python scripts\train_mpe.py ^
    --env_name %ENV% ^
    --scenario_name %SCENARIO% ^
    --algorithm_name %ALGO% ^
    --experiment_name %EXP% ^
    --project_name win_local ^
    --user_name local ^
    --use_wandb False ^
    --seed %%S ^
    --num_agents %NUM_AGENTS% ^
    --num_scripted_agents 0 ^
    --num_obstacles %NUM_OBSTACLES% ^
    --episode_length 25 ^
    --num_env_steps 20000 ^
    --n_rollout_threads 1 ^
    --n_eval_rollout_threads 1 ^
    --n_training_threads 1 ^
    --ppo_epoch 10 ^
    --num_mini_batch 1 ^
    --use_valuenorm ^
    --use_popart ^
    --use_ReLU ^
    --use_naive_recurrent_policy ^
    --graph_feat_type relative ^
    --actor_graph_aggr node ^
    --critic_graph_aggr node ^
    --use_centralized_V True ^
    --lr 7e-4 ^
    --critic_lr 7e-4 ^
    --verbose
)

pause
