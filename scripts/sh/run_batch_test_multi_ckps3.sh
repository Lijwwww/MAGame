#!/bin/bash

# 1. 定义环境列表
# ENVS=("default" "speed0" "speed1" "obstacle0" "obstacle1" "poses0" "poses1" "friction0" "friction1")
ENVS=("obstacle1" "friction0")

# 2. 定义模型名列表
# MODELS=("TRPO" "TQC")
MODELS=("PPO")

echo "=== 开始批量交叉测试 (共 $((${#ENVS[@]} * ${#MODELS[@]})) 个任务) ==="

for env in "${ENVS[@]}"; do
    for model in "${MODELS[@]}"; do
        
        echo ">>> 正在测试: Env=[$env] | Model=[$model]"
        
        # 自动组合参数
        /isaac-sim/python.sh scripts/test_multi_ckps.py headless=True test=True +model_name=$model +env_name=$env +checkpoint_name=$model +env_type=search
        
        echo "--------------------------------------------------"
        sleep 5 # 稍微歇一秒
    done
done
