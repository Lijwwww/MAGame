import pandas as pd
import json
import os
import numpy as np

# ================= 配置与常量 =================

# 1. 环境开关: 可选 'search' 或 'capture'
ENV_TYPE = 'search'

# 2. 全局指标键名 -> 中文表头映射
GLOBAL_METRICS_MAP = {
    "SR": "成功率",
    "avg_step": "平均步数",
    "avg_time": "平均时间",
    "avg_distance": "平均距离",
    "avg_gap": "平均间隔",
    "avg_collision": "平均碰撞次数",
    "avg_evasion": "平均规避次数",
    "avg_reward": "平均奖励"
}

# 3. 根据环境类型加载不同的配置
if ENV_TYPE == 'search':
    # Search 环境配置
    BASE_DIR = 'results/eval/search'
    OUTPUT_FILE = 'results/excel/search/total_results.xlsx'
    
    # Search 特有的算法列表
    ALGORITHMS = ['TD3', 'PPO', 'DDPG']
    
    # Search 特有的场景列表 (无 speed, poses)
    SCENARIOS = ['default', 'obstacle0', 'obstacle1', 'friction0', 'friction1']
    
    # Search 特有的指标 (无 evasion)
    TARGET_METRICS = ["SR", "avg_step", "avg_collision", "avg_reward"]

else:
    # Capture 环境配置
    BASE_DIR = 'results/eval/capture'
    OUTPUT_FILE = 'results/excel/capture/total_results.xlsx'
    
    # Capture 特有的算法列表
    ALGORITHMS = ['TD3', 'SAC', 'PPO', 'DDPG', 'CrossQ', 'TRPO', 'TQC']
    
    # Capture 特有的场景列表
    SCENARIOS = [
        "default", "obstacle0", "obstacle1", "friction0", "friction1",
        "poses0", "poses1", "speed0", "speed1"
    ]
    
    # Capture 特有的指标
    TARGET_METRICS = ["SR", "avg_step", "avg_collision", "avg_evasion", "avg_reward"]

# 输入文件路径
JSON_FILE_PATH = os.path.join(BASE_DIR, 'merged_results.json')

def main():
    print(f"当前环境: {ENV_TYPE}")
    print(f"读取文件: {JSON_FILE_PATH}")
    print(f"输出文件: {OUTPUT_FILE}")

    # 1. 读取 JSON 数据
    if not os.path.exists(JSON_FILE_PATH):
        print(f"错误: 找不到输入文件 {JSON_FILE_PATH}")
        return

    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 构建数据行
    rows = []
    
    for algo_name in ALGORITHMS:
        # 如果 JSON 中没有该算法数据，跳过
        if algo_name not in data:
            print(f"警告: 数据中未找到算法 {algo_name}，跳过。")
            continue
            
        algo_data = data[algo_name]
        
        # -- Row 1: 算法名称 (作为小标题) --
        # 为了美观，后面填充空字符串，长度等于 列数(指标列+场景列数)
        # 这里的长度控制主要是为了后续如果手动合并单元格方便，或者保持对齐
        row_title = [algo_name] + [""] * len(SCENARIOS)
        rows.append(row_title)
        
        # -- Row 2: 表头 (指标 + 各个场景名) --
        row_header = ["指标"] + SCENARIOS
        rows.append(row_header)
        
        # -- Rows 3-N: 各个指标的数值 --
        for metric_key in TARGET_METRICS:
            # 获取中文名
            metric_name = GLOBAL_METRICS_MAP.get(metric_key, metric_key)
            row_data = [metric_name]
            
            for scenario in SCENARIOS:
                # 获取该场景下的指标字典
                scenario_data = algo_data.get(scenario, {})
                
                # 获取具体数值，默认为 0
                val = scenario_data.get(metric_key, 0)
                
                # 处理列表类型 (取平均值)
                if isinstance(val, list):
                    val = np.mean(val) if len(val) > 0 else 0
                
                # 保留4位小数
                try:
                    val = round(float(val), 4)
                except (ValueError, TypeError):
                    val = 0 # 异常处理
                
                row_data.append(val)
            
            rows.append(row_data)
            
        # -- Row N+1: 空行作为分隔符 --
        rows.append([""] * (len(SCENARIOS) + 1))

    # 3. 创建 DataFrame
    df = pd.DataFrame(rows)

    # 4. 导出 Excel
    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # header=False, index=False 因为我们自己构建了包含标题的行结构
    try:
        df.to_excel(OUTPUT_FILE, index=False, header=False)
        print(f"\nExcel 文件生成成功: {OUTPUT_FILE}")
        
        # 简单预览前几行
        print("数据预览 (前 10 行):")
        # 临时设置 pandas 显示选项以避免换行混乱
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.width', 1000)
        print(df.head(10))
        
    except Exception as e:
        print(f"导出失败: {e}")

if __name__ == '__main__':
    main()