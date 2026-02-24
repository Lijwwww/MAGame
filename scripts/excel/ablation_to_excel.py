import json
import os
import pandas as pd

# ================= 配置 =================
# 1. 环境开关: 可选 'search' 或 'capture'
# ENV_TYPE = 'capture'
ENV_TYPE = 'search'

if ENV_TYPE == 'search':
    # Search 环境配置
    BASE_DIR = 'results/robustness_score/search'
    OUTPUT_FILE = 'results/excel/search/ablation_comparison.xlsx'
else:
    # Capture 环境配置
    BASE_DIR = 'results/robustness_score/capture'
    OUTPUT_FILE = 'results/excel/capture/ablation_comparison.xlsx'

# 定义行名与文件名的对应关系 (根据 BASE_DIR 拼接)
FILES_MAP = {
    "Ours": "robustness_main.json",
    "Equal-W": "ablation_equal_w.json",
    "SR-Only": "ablation_single_sr.json"
}

def main():
    print(f"当前环境: {ENV_TYPE}")
    print(f"读取目录: {BASE_DIR}")
    
    combined_data = {}

    # 1. 读取数据
    for row_name, filename in FILES_MAP.items():
        filepath = os.path.join(BASE_DIR, filename)
        if not os.path.exists(filepath):
            print(f"文件缺失: {filename} (将跳过)")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            # 读取 {"TD3": score, ...}
            combined_data[row_name] = json.load(f)

    # 2. 转换为 DataFrame
    # orient='index' 会将 keys (Ours, w/o weights...) 设为行索引
    # columns 自动变为算法名 (TD3, SAC...)
    df = pd.DataFrame.from_dict(combined_data, orient='index')

    # 3. 导出 Excel
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # index=True 保留行索引 (即 Ours, w/o weights 等作为第一列)
    df.to_excel(OUTPUT_FILE, index=True)
    
    print(f"消融实验对比表已导出至:\n   {OUTPUT_FILE}")
    print("\n预览:")
    print(df)

if __name__ == '__main__':
    main()