import json
import pandas as pd
import os
import numpy as np

# ================= 配置区域 =================
# 1. 环境开关: 可选 'search' 或 'capture'
ENV_TYPE = 'capture'

if ENV_TYPE == 'search':
    # Search 环境配置
    JSON_PATH = 'results/eval/search/merged_results.json'
    OUTPUT_EXCEL = 'results/excel/search/slopegraph_table.xlsx'
    
    # Search 算法列表
    ALGORITHMS = ['TD3', 'PPO', 'DDPG']
    
    # Search 环境列表 (无 poses，替换为 friction 以保持3个对比)
    # 列表里的顺序就是图上 X 轴从左到右的顺序
    SELECTED_ENVS = ["friction1", "default", "obstacle0"]

else:
    # Capture 环境配置
    JSON_PATH = 'results/eval/capture/merged_results.json'
    OUTPUT_EXCEL = 'results/excel/capture/slopegraph_table.xlsx'
    
    # Capture 算法列表
    ALGORITHMS = ['TD3', 'SAC', 'PPO', 'DDPG', 'CrossQ', 'TRPO', 'TQC']
    
    # Capture 环境列表
    SELECTED_ENVS = ["poses1", "default", "obstacle0"]

# Sheet配置：(指标键名, 是否越大越好, Sheet名称)
TARGETS = [
    # 子图 A (左): 稳定组 -> SR (越大越好)
    ("SR", True, "SubplotA_Stable_SR"), 
    
    # 子图 B (右): 混乱组 -> 碰撞 (越小越好)
    ("avg_collision", False, "SubplotB_Chaotic_Collision")
]

# ================= 核心逻辑 =================

def get_transposed_rank_df(data, metric, env_list, ascending):
    # 1. 提取数据
    raw_values = {}
    for alg in ALGORITHMS:
        raw_values[alg] = {}
        for env in env_list:
            # 注意：若 env 在 JSON 中不存在会报错，请确保 SELECTED_ENVS 在当前环境数据中存在
            val = data[alg][env][metric]
            if isinstance(val, list):
                val = np.mean(val)
            raw_values[alg][env] = val

    # 2. 转为 DataFrame (行=算法，列=环境)
    df_raw = pd.DataFrame.from_dict(raw_values, orient='index')
    df_raw = df_raw[env_list] 

    # 3. 计算排名 (Rank 1 = 最好)
    # ascending=False (值越大越好 -> SR)
    # ascending=True  (值越小越好 -> Collision)
    df_rank = df_raw.rank(ascending=ascending, method='min')
    
    # 4. 关键步骤：转置 (Transpose)
    # 变成：行=环境，列=算法
    df_final = df_rank.T
    
    # 5. 整理格式
    # 把行索引(环境名)变成第一列
    df_final.reset_index(inplace=True)
    df_final.columns = ['Environment'] + ALGORITHMS
    
    return df_final

def main():
    print(f"当前环境: {ENV_TYPE}")
    print(f"读取路径: {JSON_PATH}")

    # 读取
    if not os.path.exists(JSON_PATH):
        print(f"错误: 找不到文件 {JSON_PATH}")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 导出
    os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)
    
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        for metric, higher_is_better, sheet_name in TARGETS:
            # 计算排名并转置
            # 注意这里传入 ascending = not higher_is_better
            # 如果是 SR (higher_is_better=True), 则 ascending=False (降序排名, 值大是第1名)
            df = get_transposed_rank_df(data, metric, SELECTED_ENVS, not higher_is_better)
            
            # 写入 Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"已生成 Sheet: {sheet_name} (指标: {metric})")
            print("   数据结构预览 (行=环境, 列=算法):")
            print(df.to_string(index=False))
            print("-" * 20)

    print(f"\n文件已保存: {OUTPUT_EXCEL}")

if __name__ == '__main__':
    main()