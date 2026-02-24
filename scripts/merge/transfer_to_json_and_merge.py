import os
import re
import json

# ================= 配置区域 =================
# 原始txt文件所在的文件夹路径 (当前目录用 '.')
SOURCE_FOLDER = 'logs' 

# 生成的json文件存放的文件夹名称
OUTPUT_FOLDER = 'logs/json_output'

# 需要转换的算法名称列表 (只有在此列表中的算法文件才会被处理)
TARGET_ALGORITHMS = ['TD3', 'SAC', 'PPO', 'DDPG', 'CrossQ', 'TRPO', 'TQC', 'DDPG'] 

# ===========================================

def parse_list(list_content_str):
    """将 '[1.2 3.4 5.6]' 格式的内容转换为浮点数列表"""
    # 去除首尾空白，按空格分割
    items = list_content_str.strip().split()
    return [parse_value(x) for x in items]

def get_avg(value_list):
    """计算列表平均值，保留4位小数"""
    if not value_list:
        return 0.0
    return round(sum(value_list) / len(value_list), 4)

def parse_value(value_str):
    """转浮点数，保留4位小数"""
    try:
        return round(float(value_str), 4)
    except ValueError:
        return 0.0

def process_log_file(file_path):
    """解析单个日志文件，返回字典格式数据 (值也是字典)"""
    data = {}
    current_env = None
    
    # 正则表达式 (增加 avg_time)
    patterns = {
        'SR': re.compile(r'SR:([\d\.]+)'),
        'step': re.compile(r'avg_step:([\d\.]+)'),
        'time': re.compile(r'avg_time:([\d\.]+)'),
        'distance': re.compile(r'avg_distance:\[([\d\.\s]+)\]'),
        'gap': re.compile(r'avg_gap:\[([\d\.\s]+)\]'),
        'collision': re.compile(r'avg_collision:([\d\.]+)'),
        'evasion': re.compile(r'avg_evasion:([\d\.]+)'),
        'reward': re.compile(r'avg_reward:(-?[\d\.]+)')
    }

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        
        if 'nolidar' in line:
            break
        if not line:
            continue

        if line.endswith(':') and not line.startswith('SR:'):
            current_env = line[:-1]
            if 'speeed' in current_env:
                current_env = current_env.replace('speeed', 'speed')
            continue

        if line.startswith('SR:') and current_env:
            try:
                # 提取列表原始数据
                dist_match = patterns['distance'].search(line)
                gap_match = patterns['gap'].search(line)
                dist_list = parse_list(dist_match.group(1)) if dist_match else []
                gap_list = parse_list(gap_match.group(1)) if gap_match else []

                record = {
                    "SR": parse_value(patterns['SR'].search(line).group(1)),
                    "avg_step": parse_value(patterns['step'].search(line).group(1)),
                    "avg_time": parse_value(patterns['time'].search(line).group(1)) if patterns['time'].search(line) else 0.0,
                    
                    # --- 这里决定了是存列表还是存均值 ---
                    # "avg_distance": get_avg(dist_list), # 如果想存列表，改成 dist_list
                    # "avg_gap": get_avg(gap_list),       # 如果想存列表，改成 gap_list
                    "avg_distance": dist_list, # 如果想存列表，改成 dist_list
                    "avg_gap": gap_list,       # 如果想存列表，改成 gap_list
                    # --------------------------------
                    
                    "avg_collision": parse_value(patterns['collision'].search(line).group(1)),
                    "avg_evasion": parse_value(patterns['evasion'].search(line).group(1)),
                    "avg_reward": parse_value(patterns['reward'].search(line).group(1)) if patterns['reward'].search(line) else 0.0
                }
                
                data[current_env] = record
                current_env = None
                
            except AttributeError as e:
                print(f"Warning: 解析行时出错 (环境: {current_env}): {e}")
                continue

    return data

def save_custom_json(data, output_path):
    """
    自定义保存JSON格式：
    外层保持缩进，最内层的字典数据强制保持在一行。
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("{\n")
        
        alg_list = list(data.keys())
        for i, alg in enumerate(alg_list):
            f.write(f'    "{alg}": {{\n')
            
            envs = data[alg]
            env_keys = list(envs.keys())
            for j, env_name in enumerate(env_keys):
                # 将内部字典 dump 成字符串，不换行
                val_str = json.dumps(envs[env_name], ensure_ascii=False)
                
                comma = "," if j < len(env_keys) - 1 else ""
                f.write(f'        "{env_name}": {val_str}{comma}\n')
            
            alg_comma = "," if i < len(alg_list) - 1 else ""
            f.write(f'    }}{alg_comma}\n')
            
        f.write("}")

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"已创建输出目录: {OUTPUT_FOLDER}")

    # 用于最后合并所有结果的大字典
    all_data = {}

    for alg in TARGET_ALGORITHMS:
        filename = f"log_{alg}.txt"
        file_path = os.path.join(SOURCE_FOLDER, filename)
        
        if os.path.exists(file_path):
            print(f"正在处理: {filename} ...")
            
            # 解析内容
            env_data = process_log_file(file_path)
            
            # 构造成目标结构
            final_json = {alg: env_data}
            
            # 1. 保存单个文件 (使用新的自定义保存函数)
            output_filename = f"log_{alg}.json"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            save_custom_json(final_json, output_path)
            print(f"  -> 单个文件已保存: {output_path}")

            # 2. 将数据加入总字典，用于最后合并
            all_data.update(final_json)

        else:
            print(f"跳过: 未找到 {filename}")

    # ==========================================
    # 最后保存合并后的总文件
    # ==========================================
    if all_data:
        merged_output_path = os.path.join(OUTPUT_FOLDER, "all_algorithms_merged.json")
        save_custom_json(all_data, merged_output_path)
        print(f"\n[完成] 所有结果已合并保存至: {merged_output_path}")

if __name__ == '__main__':
    main()