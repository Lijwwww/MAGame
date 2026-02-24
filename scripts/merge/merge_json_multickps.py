import os
import json

# ================= 配置区域 =================
ENV_TYPE = 'capture'
# ENV_TYPE = 'search'
SOURCE_FOLDER = f'/workspace/omniisaacgymenvs/PursuitSim3D/results/eval_multi_ckps/{ENV_TYPE}'
OUTPUT_FILE = f'/workspace/omniisaacgymenvs/PursuitSim3D/results/eval_multi_ckps/{ENV_TYPE}/merged_results.json'
# ALGORITHMS = ['TD3', 'SAC', 'PPO', 'DDPG', 'CrossQ', 'TRPO', 'TQC']
ALGORITHMS = ['TD3', 'PPO']
# ===========================================

def get_step_int(step_str):
    """辅助函数：从 '20000_steps' 中提取数字 20000 用于排序"""
    try:
        return int(step_str.split('_')[0])
    except:
        return 0

def merge_json_multickps(source_folder, output_file, algorithms):
    # 最终结构: all_data[algo][step][env] = metrics
    all_data = {}

    print(f"正在扫描目录: {source_folder} ...")

    # --- 第一步：遍历目录读取数据 ---
    for algo in algorithms:
        # 算法目录，例如 .../TD3
        algo_path = os.path.join(source_folder, algo)
        
        if not os.path.exists(algo_path):
            # print(f" [x] 跳过不存在的算法目录: {algo}")
            continue

        # 确保字典里有这个算法的key
        if algo not in all_data:
            all_data[algo] = {}

        # 遍历算法目录下的子文件夹，例如 TD3_default, TD3_friction0 ...
        for sub_folder_name in os.listdir(algo_path):
            sub_folder_path = os.path.join(algo_path, sub_folder_name)
            
            # 只处理文件夹，且名称符合 {Algorithm}_{Env} 格式
            if not os.path.isdir(sub_folder_path) or not sub_folder_name.startswith(algo + "_"):
                continue
            
            # 解析环境名：从 "TD3_friction0" 中去掉 "TD3_" 得到 "friction0"
            # len(algo) + 1 是为了跳过中间的下划线
            env_name = sub_folder_name[len(algo)+1:]
            
            # 构造目标JSON文件名：log_TD3_friction0.json
            json_filename = f"log_{sub_folder_name}.json"
            json_path = os.path.join(sub_folder_path, json_filename)
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        # 读取原始文件
                        # 结构: { "TD3_friction0": { "20000_steps": {...}, "40000_steps": {...} } }
                        raw_content = json.load(f)
                        
                        # 获取核心数据块
                        # 尝试直接用文件夹名作为key获取，如果失败则取第一个value
                        if sub_folder_name in raw_content:
                            steps_data = raw_content[sub_folder_name]
                        else:
                            steps_data = list(raw_content.values())[0]

                        # --- 核心数据重组 (Pivot) ---
                        # 原始: 步数 -> 指标
                        # 目标: 算法 -> 步数 -> 环境 -> 指标
                        for step_key, metrics in steps_data.items():
                            # 如果该步数在all_data[algo]里还没记录，初始化它
                            if step_key not in all_data[algo]:
                                all_data[algo][step_key] = {}
                            
                            # 填入数据
                            all_data[algo][step_key][env_name] = metrics
                            
                except json.JSONDecodeError:
                    print(f" [!] 跳过损坏文件: {json_path}")
                except Exception as e:
                    print(f" [!] 处理文件出错 {json_path}: {e}")

    # --- 第二步：按指定格式写入 ---
    if all_data:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f"正在写入合并结果至: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("{\n")
            
            # 1. 遍历算法 (排序)
            sorted_algos = sorted(all_data.keys())
            for i, algo in enumerate(sorted_algos):
                f.write(f'    "{algo}": {{\n')
                
                # 2. 遍历步数 (按数值大小排序: 2万, 4万, 10万...)
                # all_data[algo] 的 key 是 "20000_steps" 等
                sorted_steps = sorted(all_data[algo].keys(), key=get_step_int)
                
                for j, step in enumerate(sorted_steps):
                    f.write(f'        "{step}": {{\n')
                    
                    # 3. 遍历环境 (排序: default, friction0, obstacle0...)
                    envs_dict = all_data[algo][step]
                    sorted_envs = sorted(envs_dict.keys())
                    
                    for k, env in enumerate(sorted_envs):
                        metrics = envs_dict[env]
                        
                        # 【格式化核心】将指标字典转为不换行的字符串
                        metrics_str = json.dumps(metrics, ensure_ascii=False)
                        
                        # 逗号处理
                        comma = "," if k < len(sorted_envs) - 1 else ""
                        
                        # 写入: "friction0": {...},
                        f.write(f'            "{env}": {metrics_str}{comma}\n')
                    
                    # 步数块结束
                    step_comma = "," if j < len(sorted_steps) - 1 else ""
                    f.write(f'        }}{step_comma}\n')
                
                # 算法块结束
                algo_comma = "," if i < len(sorted_algos) - 1 else ""
                f.write(f'    }}{algo_comma}\n')
            
            f.write("}")
            
        print("[成功] 合并完成。")
    else:
        print("[警告] 未提取到任何数据，请检查路径配置。")

if __name__ == '__main__':
    merge_json_multickps(SOURCE_FOLDER, OUTPUT_FILE, ALGORITHMS)