import json
import pandas as pd
import os

# ================= 配置与常量 =================

# 1. 环境开关: 可选 'search' 或 'capture'
ENV_TYPE = 'capture'
# ENV_TYPE = 'search'

# 2. 全局映射字典：JSON Key -> Excel Header
# 包含所有环境可能用到的指标
GLOBAL_COLUMN_MAPPING = {
    "SR": "成功率",
    "avg_step": "平均步数",
    "avg_reward": "平均奖励",
    "avg_collision": "平均碰撞次数",
    "avg_evasion": "平均逃逸次数"
}

# 3. 根据环境类型加载不同的配置
if ENV_TYPE == 'search':
    # Search 环境配置
    BASE_DIR = 'results/robustness_score/search'
    OUTPUT_FILE = 'results/excel/search/RS_CV_table.xlsx'
    
    # Search 环境需要的指标顺序
    TARGET_METRICS = ["SR", "avg_step", "avg_reward", "avg_collision"]

else:
    # Capture 环境配置
    BASE_DIR = 'results/robustness_score/capture'
    OUTPUT_FILE = 'results/excel/capture/RS_CV_table.xlsx'
    
    # Capture 环境需要的指标顺序 (包含 avg_evasion)
    TARGET_METRICS = ["SR", "avg_step", "avg_reward", "avg_collision", "avg_evasion"]

# 输入文件路径 (基于 BASE_DIR)
CV_FILE = os.path.join(BASE_DIR, 'cv_details.json')
ROBUSTNESS_FILE = os.path.join(BASE_DIR, 'robustness_main.json')

def main():
    print(f"当前环境: {ENV_TYPE}")
    print(f"读取目录 (Input): {BASE_DIR}")
    print(f"输出路径 (Output): {OUTPUT_FILE}")

    # 1. 检查文件是否存在
    if not os.path.exists(CV_FILE) or not os.path.exists(ROBUSTNESS_FILE):
        print(f"错误: 找不到输入文件。")
        print(f"请检查目录 {BASE_DIR} 下是否有 cv_details.json 和 robustness_main.json")
        return

    # 2. 读取 JSON 数据
    print("正在读取数据...")
    with open(CV_FILE, 'r', encoding='utf-8') as f:
        cv_data = json.load(f)
    
    with open(ROBUSTNESS_FILE, 'r', encoding='utf-8') as f:
        robustness_data = json.load(f)

    # 3. 数据处理：将数据重组为列表字典格式
    rows = []

    # 遍历所有算法
    for algo_name, metrics in cv_data.items():
        row = {"算法": algo_name}
        
        # 仅填充当前环境定义在 TARGET_METRICS 中的指标
        for metric_key in TARGET_METRICS:
            # 获取对应的中文列名
            chinese_header = GLOBAL_COLUMN_MAPPING.get(metric_key, metric_key)
            
            # 从 cv_data 中获取值，若缺失则设为 None
            # 注意：原逻辑是遍历 metrics.items()，这里改为遍历 TARGET_METRICS 以保证顺序和筛选
            value = metrics.get(metric_key, None)
            row[chinese_header] = value
        
        # 填充鲁棒性值 (从另一个文件获取)
        row["鲁棒性值"] = robustness_data.get(algo_name, None)
        
        rows.append(row)

    # 4. 创建 DataFrame
    df = pd.DataFrame(rows)

    # 5. 格式调整
    # 动态构建最终 Excel 的列顺序
    # 格式: [算法] + [环境相关指标中文名] + [鲁棒性值]
    final_columns_order = ["算法"] + \
                          [GLOBAL_COLUMN_MAPPING.get(m, m) for m in TARGET_METRICS] + \
                          ["鲁棒性值"]
    
    # 确保列存在 (防止数据完全缺失时的报错)
    available_cols = [col for col in final_columns_order if col in df.columns]
    df = df[available_cols]

    # 保留三位小数
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].round(3)

    # 6. 导出 Excel
    try:
        # 关键步骤：创建输出目录
        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")

        df.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
        print(f"成功！表格已导出至: {OUTPUT_FILE}")
        print("\n预览前几行数据：")
        # 使用 to_string 防止打印时换行混乱
        print(df.to_string(index=False))
    except Exception as e:
        print(f"导出失败: {e}")

if __name__ == '__main__':
    main()