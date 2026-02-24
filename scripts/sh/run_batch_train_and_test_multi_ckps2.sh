#!/bin/bash

# ================= 配置区域 =================

# 1. 定义测试环境列表 (训练完成后，要拿去这些环境里跑分)
TEST_ENVS=("default" "speed0" "speed1" "obstacle0" "obstacle1" "poses0" "poses1" "friction0" "friction1")

# 2. 定义要训练和测试的模型
# 注意：确保这些算法已经在 TRAIN_MULTI_CKPT_MAP (Python脚本) 中注册过
# MODELS=("TD3" "SAC" "PPO" "DDPG" "CrossQ" "TRPO" "TQC")
MODELS=("DDPG" "CrossQ")

# ===========================================

echo "=== 开始流程：在 Default 环境训练 -> 全环境交叉测试 ==="

for model in "${MODELS[@]}"; do
    
    # ------------------------------------------------------------------
    # 阶段 1: 训练 (Training)
    # 目标: 在 Default 环境训练
    # ------------------------------------------------------------------

    echo -e "\n\n"
    echo "################################################################"
    echo ">>> [阶段1: 训练] 算法: $model | 环境: Default (代码内置)"
    echo ">>> 预期保存路径: checkpoints/multi_ckps/$model/${model}_xxxxx_steps.zip"
    echo "################################################################"
    
    /isaac-sim/python.sh scripts/train_multi_ckps.py \
        +model_name=$model \
        +checkpoint_name=$model \
        +env_name="default"

    # 检查训练是否成功
    # if [ $? -ne 0 ]; then
    #     echo "!!! 错误: 算法 $model 训练失败，跳过后续测试 !!!"
    #     continue
    # fi

    # ------------------------------------------------------------------
    # 阶段 2: 评估 (Evaluation)
    # ------------------------------------------------------------------
    echo -e "\n"
    echo ">>> [阶段2: 测试] 正在评估模型 [$model] 的泛化能力..."
    
    for test_env in "${TEST_ENVS[@]}"; do
        echo "    -> 正在测试: 训练源=[default] vs 测试环境=[$test_env]"
        
        /isaac-sim/python.sh scripts/test_multi_ckps.py \
            test=True \
            +model_name=$model \
            +env_name=$test_env \
            +checkpoint_name=$model
        
        sleep 2 # 缓冲一下
    done
    
    echo "--------------------------------------------------"
    echo "完成算法 $model 的所有流程。"
        
done

echo "=== 所有任务执行完毕 ==="