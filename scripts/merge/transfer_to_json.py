import os
import re
import json

# ================= 配置区域 =================
# 指定输入的单个 TXT 文件路径
INPUT_FILE = '/workspace/omniisaacgymenvs/PursuitSim3D/logs/TD3_history/default/TD3_default.txt'

# 指定输出的 JSON 文件路径
OUTPUT_FILE = '/workspace/omniisaacgymenvs/PursuitSim3D/logs/TD3_history/default/TD3_default.json'

# 指定该文件对应的算法名称 (将作为 JSON 的一级 Key)
# 如果留空 None，脚本会自动根据文件名猜测 (如 log_TD3.txt -> TD3)
ALGO_NAME = "TD3_default" 
# ===========================================

def parse_list(list_content_str):
    """将 '[1.2 3.4 5.6]' 格式的内容转换为浮点数列表"""
    items = list_content_str.strip().split()
    return [parse_value(x) for x in items]

def get_avg(value_list):
    """计算列表平均值，保留4位小数"""
    if not value_list:
        return 0.0
    # return round(sum(value_list) / len(value_list), 4)
    return sum(value_list) / len(value_list)

def parse_value(value_str):
    """转浮点数，保留4位小数"""
    try:
        # return round(float(value_str), 4)
        return float(value_str)
    except ValueError:
        return 0.0

def process_log_file(file_path):
    """解析日志文件核心逻辑"""
    data = {}
    current_env = None
    
    # 正则表达式
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

    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return {}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if 'nolidar' in line: break
        if not line: continue

        # 识别环境名行 (以冒号结尾且不是SR开头)
        if line.endswith(':') and not line.startswith('SR:'):
            current_env = line[:-1]
            if 'speeed' in current_env: current_env = current_env.replace('speeed', 'speed')
            continue

        # 识别数据行
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
            except AttributeError:
                continue
    return data

def save_custom_json(data, output_path):
    """自定义保存：每个环境一行"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("{\n")
        alg_list = list(data.keys())
        for i, alg in enumerate(alg_list):
            f.write(f'    "{alg}": {{\n')
            
            envs = data[alg]
            env_keys = list(envs.keys())
            for j, env_name in enumerate(env_keys):
                # 将字典转为单行字符串
                val_str = json.dumps(envs[env_name], ensure_ascii=False)
                comma = "," if j < len(env_keys) - 1 else ""
                f.write(f'        "{env_name}": {val_str}{comma}\n')
            
            alg_comma = "," if i < len(alg_list) - 1 else ""
            f.write(f'    }}{alg_comma}\n')
        f.write("}")

def main():
    print(f"正在处理单个文件: {INPUT_FILE}")
    
    # 1. 解析数据
    env_data = process_log_file(INPUT_FILE)
    
    if not env_data:
        print("未解析到任何数据，请检查文件路径或内容格式。")
        return

    # 2. 确定算法名称 Key
    if ALGO_NAME:
        final_alg_name = ALGO_NAME
    else:
        # 自动推导: "logs/log_TD3.txt" -> "TD3"
        base = os.path.basename(INPUT_FILE)
        name_no_ext = os.path.splitext(base)[0]
        final_alg_name = name_no_ext.replace("log_", "")

    # 3. 构造最终结构
    final_json = {final_alg_name: env_data}

    # 4. 保存
    save_custom_json(final_json, OUTPUT_FILE)
    print(f"转换成功！已保存至: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()