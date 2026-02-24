import json
import os
import pandas as pd

# ================= 配置与常量 =================

# 1. 环境开关: 可选 'search' 或 'capture'
# ENV_TYPE = 'capture'
ENV_TYPE = 'search'

# 2. 全局指标键名 -> 中文表头映射字典 (包含所有可能出现的指标)
GLOBAL_METRIC_MAPPING = {
    "SR": "成功率",
    "avg_step": "平均步数",
    "avg_reward": "平均奖励",
    "avg_collision": "平均碰撞次数",
    "avg_evasion": "平均规避次数"
}

# 3. 根据环境类型加载不同的配置
if ENV_TYPE == 'search':
    # Search 环境配置
    METRICS_NAME = ["SR", "avg_step", "avg_collision", "avg_reward"]
    HUMAN_WEIGHTS = {
        "SR": 0.4,
        "avg_step": 0.1,
        "avg_collision": 0.15,
        "avg_reward": 0.35
    }
    BASE_DIR = 'results/robustness_score/search'
    OUTPUT_FILE = 'results/excel/search/weights_distribution_table.xlsx'

else:
    # Capture 环境配置 (默认为 capture)
    METRICS_NAME = ["SR", "avg_step", "avg_collision", "avg_evasion", "avg_reward"]
    HUMAN_WEIGHTS = {
        "SR": 0.35,
        "avg_step": 0.1,
        "avg_collision": 0.15,
        "avg_evasion": 0.1,
        "avg_reward": 0.3
    }
    BASE_DIR = 'results/robustness_score/capture'
    OUTPUT_FILE = 'results/excel/capture/weights_distribution_table.xlsx'

def main():
    print(f"当前环境: {ENV_TYPE}")
    print(f"工作目录: {BASE_DIR}")

    # 1. 读取 JSON 数据
    game_weights_path = os.path.join(BASE_DIR, 'game_weights.json')
    final_weights_path = os.path.join(BASE_DIR, 'final_combined_weights.json')

    # 检查文件是否存在，给出更友好的报错
    if not os.path.exists(game_weights_path) or not os.path.exists(final_weights_path):
        print(f"错误: 在目录 {BASE_DIR} 下未找到权重 JSON 文件。")
        print("请先运行对应的权重计算/演化脚本生成这些文件。")
        return

    with open(game_weights_path, 'r', encoding='utf-8') as f:
        game_w = json.load(f)
    with open(final_weights_path, 'r', encoding='utf-8') as f:
        final_w = json.load(f)

    # 2. 构建表格数据 (Strict Mode)
    # 根据当前环境定义的 METRICS_NAME 顺序生成行，忽略当前环境不需要的指标
    table_data = []
    for key in METRICS_NAME:
        # 获取中文名称
        cn_name = GLOBAL_METRIC_MAPPING.get(key, key)
        
        table_data.append({
            "指标名称": cn_name,
            "基于博弈的权重": game_w[key],
            "人工先验权重": HUMAN_WEIGHTS[key],
            "总权重": final_w[key]
        })

    # 3. 创建 DataFrame 并处理格式
    df = pd.DataFrame(table_data)
    
    # 定义列顺序
    columns_order = ["指标名称", "基于博弈的权重", "人工先验权重", "总权重"]
    df = df[columns_order]

    # 数值列保留三位小数
    numeric_cols = columns_order[1:]
    df[numeric_cols] = df[numeric_cols].round(3)

    # 4. 导出 Excel
    # 确保输出目录存在 (防止因目录不存在报错)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    df.to_excel(OUTPUT_FILE, index=False)
    
    print(f"\n成功导出至: {OUTPUT_FILE}")
    print("预览数据:")
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()