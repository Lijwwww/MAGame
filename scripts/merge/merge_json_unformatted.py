import os
import json

# ================= 配置区域 =================
SOURCE_FOLDER = 'results/eval/search'  # 'results/eval/capture'
OUTPUT_FILE = os.path.join(SOURCE_FOLDER, 'merged_results.json')
ALGORITHMS = ['TD3', 'SAC', 'PPO', 'DDPG', 'CrossQ', 'TRPO', 'TQC']
# ===========================================

def main():
    all_data = {}

    print(f"开始从 {SOURCE_FOLDER} 读取并合并文件...")

    for algo in ALGORITHMS:
        filename = f"log_{algo}.json"
        file_path = os.path.join(SOURCE_FOLDER, algo, filename)
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    algo_data = json.load(f)
                    
                    # 将其更新到总字典中
                    all_data.update(algo_data)
                    print(f" -> 已合并: {filename}")
            except json.JSONDecodeError:
                print(f" [!] 跳过: {filename} 格式错误，无法解析")
        else:
            print(f" [x] 跳过: 未找到 {filename}")

    # 保存合并后的结果
    if all_data:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # indent=4 保证输出格式整齐
            json.dump(all_data, f, indent=4, ensure_ascii=False)
            
        print(f"\n合并完成！文件已保存至: {OUTPUT_FILE}")
    else:
        print("\n没有合并任何数据，请检查文件路径或算法名称。")

if __name__ == '__main__':
    main()