# 合并各算法测试结果，并邪修使内层一个环境一行让格式样式好看

import os
import json

# ================= 配置 =================
ENV_TYPE = 'search'
SOURCE_FOLDER = f'results/eval/{ENV_TYPE}'
OUTPUT_FILE = os.path.join(SOURCE_FOLDER, 'merged_results.json')
ALGORITHMS = ['TD3', 'SAC', 'PPO', 'DDPG', 'CrossQ', 'TRPO', 'TQC']
# =======================================

def merge_json(source_folder, output_file, algorithms):
    all_data = {}

    # --- 第一步：读取并合并数据 ---
    print("正在读取各个算法的日志文件...")
    for algo in algorithms:
        filename = f"log_{algo}.json"
        file_path = os.path.join(source_folder, algo, filename)
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                # 假设单个文件结构是 { "TD3": { "default": {...} } }
                algo_data = json.load(f)
                all_data.update(algo_data)
        else:
            print(f" [x] 未找到文件: {filename}")

    # --- 第二步：自定义格式写入 ---
    if all_data:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("{\n")
            
            # 遍历算法 (TD3, PPO...)
            algos = list(all_data.keys())
            for i, algo in enumerate(algos):
                f.write(f'    "{algo}": {{\n')
                
                # 遍历环境 (default, obstacle0...)
                envs = all_data[algo]
                env_names = list(envs.keys())
                for j, env_name in enumerate(env_names):
                    if ENV_TYPE == 'search':
                        if env_name == 'speed0' or env_name == 'speed1' or env_name == 'poses1' or env_name == 'poses0':
                            continue
                    metrics = envs[env_name]
                    
                    # 【核心技巧】将最内层的字典 dump 成不换行的字符串
                    # ensure_ascii=False 保证中文或其他字符正常显示
                    metrics_str = json.dumps(metrics, ensure_ascii=False)
                    
                    # 判断是否需要加逗号（最后一个元素不加）
                    comma = "," if j < len(env_names) - 1 else ""
                    
                    # 写入一行： "环境名": {数据},
                    f.write(f'        "{env_name}": {metrics_str}{comma}\n')
                
                # 算法块结束的大括号
                algo_comma = "," if i < len(algos) - 1 else ""
                f.write(f'    }}{algo_comma}\n')
            
            f.write("}")
            
        print(f"\n[成功] 合并完成 {output_file}")
    else:
        print("未合并任何数据。")

if __name__ == '__main__':
    merge_json(SOURCE_FOLDER, OUTPUT_FILE, ALGORITHMS)