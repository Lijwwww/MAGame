#!/bin/bash

# 1. 定义算法列表
# MODELS=("TD3" "SAC" "PPO" "DDPG" "CrossQ" "TRPO" "TQC")
MODELS=("DDPG")

# 2. 定义环境列表
ENVS=("default" "speed0" "speed1" "obstacle0" "obstacle1" "poses0" "poses1" "friction0" "friction1")

echo "=== 开始批量交叉测试 (共 $((${#ENVS[@]} * ${#MODELS[@]})) 个任务) ==="

for env in "${ENVS[@]}"; do
    for model in "${MODELS[@]}"; do
        
        echo ">>> 正在测试: Env=[$env] | Model=[$model]"
        
        # 自动组合参数
        /isaac-sim/python.sh scripts/test.py test=True +model_name=$model +env_name=$env +env_type=search
        
        echo "--------------------------------------------------"
        sleep 5 # 稍微歇一秒
    done
done
