import json
import pandas as pd
import os
import numpy as np

# ================= 配置区域 =================
# 1. 环境开关: 可选 'search' 或 'capture'
ENV_TYPE = 'capture'

if ENV_TYPE == 'search':
    JSON_PATH = 'results/eval/search/merged_results.json'
    OUTPUT_EXCEL = 'results/excel/search/averaged_rank_slopegraph.xlsx'
    
    ALGORITHMS = ['TD3', 'PPO', 'DDPG']
    # 参与计算的所有环境
    SELECTED_ENVS = ['default', 'obstacle0', 'obstacle1', 
                'friction0', 'friction1']
    # 要展示的指标 (X轴)
    DISPLAY_METRICS = ["SuccessRate", "Reward", "Steps", "Collision"]

else: # Capture
    JSON_PATH = 'results/eval/capture/merged_results.json'
    OUTPUT_EXCEL = 'results/excel/capture/averaged_rank_slopegraph.xlsx'
    
    ALGORITHMS = ['TD3', 'SAC', 'PPO', 'DDPG', 'CrossQ', 'TRPO', 'TQC']
    # 参与计算的所有环境
    SELECTED_ENVS = ['default', 'speed0', 'speed1', 'obstacle0', 'obstacle1', 
            'poses0', 'poses1', 'friction0', 'friction1']
    # 要展示的指标 (X轴)
    DISPLAY_METRICS = ["SuccessRate", "Reward", "Steps", "Collision", "Evasion"]

# 2. 指标详细配置
# 格式: "显示名": ("JSON键名", higher_is_better)
# higher_is_better=True : 数值越大越好 -> 排名越前 (Rank 1)
# higher_is_better=False: 数值越小越好 -> 排名越前 (Rank 1)
METRIC_CONFIG = {
    "SuccessRate": ("SR", True),
    "Reward":       ("avg_reward", True),  
    "Steps":        ("avg_step", False), 
    "Collision":    ("avg_collision", False), 
    "Evasion":   ("avg_evasion", False)
}

CHINESE_METRIC_MAP = {
    "SuccessRate": "成功率",
    "Reward": "平均奖励",
    "Steps": "平均步数",
    "Collision": "平均碰撞次数",
    "Evasion": "平均逃逸次数"
}

# ================= 核心逻辑 =================

def get_averaged_rank_df(data, env_list, metric_list):
    """
    计算逻辑：
    1. 遍历每个环境
    2. 在该环境内，对每个指标分别计算算法排名 (1=Best)
    3. 收集所有环境的排名
    4. 对排名取平均值
    """
    # 数据结构: ranks[metric][alg] = [rank_env1, rank_env2, ...]
    ranks_collection = {m: {alg: [] for alg in ALGORITHMS} for m in metric_list}

    print(f"正在处理环境: {env_list}")

    for env in env_list:
        # 检查数据是否存在
        if env not in data[ALGORITHMS[0]]:
            print(f"警告: 环境 '{env}' 在数据中未找到，跳过。")
            continue

        for metric in metric_list:
            if metric not in METRIC_CONFIG:
                continue
            
            json_key, higher_is_better = METRIC_CONFIG[metric]

            # 1. 获取该环境下，所有算法的原始数值
            current_env_values = {}
            for alg in ALGORITHMS:
                try:
                    val = data[alg].get(env, {}).get(json_key, 0)
                    if isinstance(val, list):
                        val = np.mean(val)
                    current_env_values[alg] = val
                except Exception:
                    current_env_values[alg] = 0

            # 2. 计算该环境下的排名
            # ascending=False (值大Rank1), ascending=True (值小Rank1)
            series = pd.Series(current_env_values)
            # method='min': 并列第1则都算1，下一个是3
            current_ranks = series.rank(ascending=not higher_is_better, method='min')

            # 3. 存入集合
            for alg in ALGORITHMS:
                ranks_collection[metric][alg].append(current_ranks[alg])

    # 4. 计算平均排名
    # 构建最终 DataFrame: 行=Metric, 列=Algorithm
    final_data = {}
    for metric in metric_list:
        if metric not in METRIC_CONFIG: continue
        
        final_data[metric] = {}
        for alg in ALGORITHMS:
            rank_list = ranks_collection[metric][alg]
            if rank_list:
                # 计算平均排名
                avg_rank = np.mean(rank_list)
                final_data[metric][alg] = avg_rank
            else:
                final_data[metric][alg] = 0

    df = pd.DataFrame.from_dict(final_data, orient='index')
    
    # 整理列顺序
    df = df[ALGORITHMS]
    
    # 整理行索引 (变成第一列 Metrics)
    df.reset_index(inplace=True)
    df.columns = ['Metrics'] + ALGORITHMS
    
    return df

def main():
    print(f"当前任务类型: {ENV_TYPE}")
    print(f"读取数据: {JSON_PATH}")

    if not os.path.exists(JSON_PATH):
        print(f"错误: 找不到文件 {JSON_PATH}")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)
    
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        # 计算所有环境的平均排名
        df = get_averaged_rank_df(data, SELECTED_ENVS, DISPLAY_METRICS)

        # 执行重命名
        df['Metrics'] = df['Metrics'].map(CHINESE_METRIC_MAP).fillna(df['Metrics'])
        
        # 写入 Excel
        sheet_name = "Averaged_Ranks"
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"\n[成功] 已生成表单: {sheet_name}")
        print(f"  包含指标: {DISPLAY_METRICS}")
        print("  数据预览 (值 = 平均排名, 越小越好):")
        print(df.to_string(index=False))

    print(f"\n文件已保存至: {OUTPUT_EXCEL}")
    print("-" * 30)
    print("作图提示 (Origin):")
    print("1. 将 'Metrics' 列设为 X 轴。")
    print("2. 选中所有算法列，绘制 'Parallel Plot' (平行坐标图) 或 'Line + Symbol'。")
    print("3. 观察重点：")
    print("   - 'Collision' (高权重) 这一列的排名点，是否与 'Reward'/'Steps' 的排名点连线比较平直（代表一致性高）。")
    print("   - 'SuccessRate' (低权重) 这一列，是否出现了较多的连线交叉（代表与其他指标冲突）。")

if __name__ == '__main__':
    main()