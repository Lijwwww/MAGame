# 计算两个环境类型下算法鲁棒性值、权重、消融实验值等，并画出算法鲁棒性值随训练步数的曲线图
# 已集成合并原始数据和计算博弈权重两个脚本的逻辑

import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kendalltau

from merge.merge_json_multickps import merge_json_multickps

# ================= 配置区域 =================
# 选择了读取最后一个训练节点的博弈权重，因此下面的参数大部分没用了
LAMBDA = 0.5  # 权重融合系数
METRIC_SIGNS = {
    "SR": 1, "avg_step": -1, "avg_collision": -1, "avg_evasion": -1, "avg_reward": 1
}

# 统一定义输出目录
FINAL_OUTPUT_DIR = 'results/robustness_score_multickps/combined'

# 两种环境的独立配置
CONFIGS = {
    'search': {
        'ALGORITHMS': ['TD3'],
        'ENVIRONMENTS': ['default', 'obstacle0', 'obstacle1', 'friction0', 'friction1'],
        'METRICS_NAME': ["SR", "avg_step", "avg_collision", "avg_reward"],
        'HUMAN_WEIGHTS': {
            "SR": 0.45, "avg_step": 0.1, "avg_collision": 0.1, "avg_reward": 0.35
        },
        'RESULT_PATH': 'results/eval_multi_ckps/search/merged_results.json',
        'GAME_WEIGHTS_PATH': 'results/robustness_score/search/game_weights.json',
        'SOURCE_FOLDER': f'results/eval/search',
        'ALPHA': 2.0
    },
    'capture': {
        'ALGORITHMS': ['TD3', 'PPO'],
        'ENVIRONMENTS': ['default', 'speed0', 'speed1', 'obstacle0', 'obstacle1', 
                         'poses0', 'poses1', 'friction0', 'friction1'],
        'METRICS_NAME': ["SR", "avg_step", "avg_collision", "avg_evasion", "avg_reward"],
        'HUMAN_WEIGHTS': {
            "SR": 0.4, "avg_step": 0.1, "avg_collision": 0.1, 
            "avg_evasion": 0.1, "avg_reward": 0.3
        },
        'RESULT_PATH': 'results/eval_multi_ckps/capture/merged_results.json',
        'GAME_WEIGHTS_PATH': 'results/robustness_score/capture/game_weights.json',
        'SOURCE_FOLDER': f'results/eval/capture',
        'ALPHA': 5.0
    }
}

# ===========================================

def get_step_int(step_str):
    """辅助函数：提取步数数字用于排序"""
    try:
        return int(step_str.split('_')[0])
    except:
        return 0

# -------------------------------------------------------------------------
# 1. 权重计算模块 
# -------------------------------------------------------------------------
def calculate_game_weights(snapshot):
    pass

# -------------------------------------------------------------------------
# 2. 鲁棒性计算模块 
# -------------------------------------------------------------------------
def get_conditional_cv(values):
    mu = np.mean(values)
    sigma = np.std(values)
    min_val = np.min(values)
    if min_val < 0:
        return sigma / (mu - min_val)
    else:
        return sigma / mu if mu != 0 else 0

def calculate_combined_weights(game_weights_dict, human_weights):
    """融合人类权重和博弈权重 (增加 human_weights 参数)"""
    combined_weights = {}
    total_weight = 0
    for metric, h_w in human_weights.items():
        g_w = game_weights_dict.get(metric, 0)
        w = LAMBDA * h_w + (1 - LAMBDA) * g_w
        combined_weights[metric] = w
        total_weight += w
    return {k: v / total_weight for k, v in combined_weights.items()}

def calculate_step_metrics(step_data, weights, env_list):
    """
    计算单步的鲁棒性指标 (增加 env_list 参数)
    """
    metrics_values = {k: [] for k in weights.keys()}
    
    # 提取数据
    for env_name in env_list:
        if env_name not in step_data: continue 
        env_vals = step_data[env_name]
        for metric in weights.keys():
            val = env_vals[metric]
            metrics_values[metric].append(val)
            
    # 计算统计量
    stats = {}
    for metric, values in metrics_values.items():
        if not values: values = [0]
        stats[metric] = {
            'cv': get_conditional_cv(values),
            'sigma': np.std(values)
        }

    # 1. RS Score
    rs_score = 0
    for m in weights:
        cv = stats[m]['cv']
        w = weights[m]
        if cv > 1e-9:
            rs_score += w * (-math.log(cv))
        else:
            rs_score += w * (-math.log(1e-9))

    # 2. Sigma Score
    sigma_score = 0
    for m in weights:
        s = stats[m]['sigma']
        if s > 1e-9: 
            sigma_score += weights[m] * (-math.log(s))
        else:
            sigma_score += weights[m] * (-math.log(1e-9))

    # 3. Single SR
    cv_sr = stats['SR']['cv']
    single_sr_score = -math.log(cv_sr) if cv_sr > 1e-9 else -math.log(1e-9)
    
    return rs_score, sigma_score, single_sr_score

# -------------------------------------------------------------------------
# 3. 绘图函数
# -------------------------------------------------------------------------
def plot_robustness_curves(all_results, output_dir):
    """
    绘制各算法随训练步数的鲁棒性变化曲线
    """
    import matplotlib.font_manager as fm
    # 1. 将字体文件注册到 Matplotlib 的字体管理器中
    # 建议使用变量统一路径，避免大小写或路径不一致
    font_abs_path = '/workspace/omniisaacgymenvs/PursuitSim3D/assets/Fonts/SIMSUN.TTC'
    times_abs_path = '/workspace/omniisaacgymenvs/PursuitSim3D/assets/Fonts/TIMES.TTF'

    # 1. 注册字体
    fm.fontManager.addfont(font_abs_path)
    fm.fontManager.addfont(times_abs_path)
    
    # 2. 获取准确的内部名称 (务必也用绝对路径获取)
    name_simsun = fm.FontProperties(fname=font_abs_path).get_name()
    name_times = fm.FontProperties(fname=times_abs_path).get_name()
    
    # 打印一下，确保获取到了正确名称（通常是 'SimSun'）
    # print(f"检测到字体名: {name_simsun}, {name_times}")

    # 3. 强制全局设置
    # 直接把 font.family 设为具体的字体名列表，不要只依赖 'serif' 别名
    plt.rcParams['font.family'] = [name_times, name_simsun]
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 8))
    
    # 设置风格 (可选)
    # plt.style.use('seaborn-v0_8-paper') 
    
    # 定义颜色映射 (保证同一个算法在不同环境下颜色一致)
    unique_algos = set()
    for env_res in all_results.values():
        unique_algos.update(env_res.keys())
    unique_algos = sorted(list(unique_algos))
    
    # 使用 colormap
    colors = plt.cm.get_cmap('tab10')
    # colors = plt.get_cmap('Dark2', len(unique_algos))
    algo_color_map = {algo: colors(i) for i, algo in enumerate(unique_algos)}

    colors = [
        '#4363D8', '#3CB44B','#E6194B',  '#FFE119', '#911EB4', 
        '#46F0F0', '#F032E6', '#BCF60C', '#FABEBE', '#008080'
    ]
    algo_color_map = {algo: colors[i % len(colors)] for i, algo in enumerate(unique_algos)}

    # 定义线型映射
    line_styles = {'search': '--', 'capture': '-'}
    env_markers = {'search': 'o', 'capture': 's'} # 不同的环境也可以用不同的点区分

    for env_name, env_results in all_results.items():
        style = line_styles.get(env_name, '-')
        
        for algo, data in env_results.items():
            steps = data['steps']
            scores = data['rs_scores']
            
            color = algo_color_map.get(algo, 'black')
            marker = env_markers.get(env_name, 'o')
            
            label_str = f"{algo} ({env_name})"
            
            plt.plot(steps, scores, marker=marker, linestyle=style, 
                     color=color, label=label_str, linewidth=2, markersize=6)

    plt.xlabel('训练步数', fontsize=24)
    plt.ylabel('鲁棒性评分', fontsize=24)
    plt.legend(fontsize=16)
    # plt.legend(fontsize=16, ncol=2) # 分两列显示图例
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_path = os.path.join(output_dir, 'robustness_trend_combined.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"趋势图已保存至: {output_path}")


def save_excel_for(all_results, output_dir):
    """
    将结果整理为 DataFrame 并导出 Excel
    Columns: Search_TD3, Capture_TD3, ...
    """
    print(f"正在生成 Origin 绘图数据...")
    
    data_dict = {}
    
    for env_name, env_results in all_results.items():
        for algo, res in env_results.items():
            # 列名格式：Env_Algo
            col_name = f"{env_name}_{algo}"
            series = pd.Series(data=res['rs_scores'], index=res['steps'], name=col_name)
            data_dict[col_name] = series

    df = pd.DataFrame(data_dict)
    
    df.sort_index(inplace=True)
    df.index.name = 'Step'

    output_path = os.path.join(output_dir, 'robustness_curves_combined.xlsx')
    df.to_excel(output_path)
    print(f"Origin 绘图数据已保存至: {output_path}")

# -------------------------------------------------------------------------
# 4. 主流程
# -------------------------------------------------------------------------
def compute_env_results(env_type, config):
    """
    计算特定环境类型下的所有算法鲁棒性结果
    """
    print(f"\n>>> 开始计算环境: {env_type}")
    result_path = config['RESULT_PATH']
    game_weights_path = config['GAME_WEIGHTS_PATH']
    algorithms = config['ALGORITHMS']
    human_weights = config['HUMAN_WEIGHTS']
    environments = config['ENVIRONMENTS']
    source_folder = config['SOURCE_FOLDER']

    # 可选：调用merge_json脚本，先合并原始测试数据
    merge_json_multickps(source_folder, result_path, algorithms)

    print(f"读取数据: {result_path}")
    if not os.path.exists(result_path):
        print(f"错误：文件 {result_path} 不存在，跳过该环境。")
        return {}

    with open(result_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    # 读取权重 (只读一次，假设所有step共用同一套最终权重)
    with open(game_weights_path, 'r', encoding='utf-8') as f:
        game_weights = json.load(f)
    final_weights = calculate_combined_weights(game_weights, human_weights)
    
    env_results = {}

    for algo in algorithms:
        if algo not in full_data: continue
        
        print(f"处理算法: {algo} ...")
        steps = sorted(full_data[algo].keys(), key=get_step_int)
        
        algo_res = {
            "steps": [], "rs_scores": [], "sigma_scores": [], "single_sr_scores": []
        }
        
        for step in steps:
            step_int = get_step_int(step)
            step_data_for_metric = full_data[algo][step]
            
            # 计算鲁棒性 (注意传入了 environments 列表)
            rs, sigma, sr = calculate_step_metrics(step_data_for_metric, final_weights, environments)
            
            algo_res["steps"].append(step_int)
            algo_res["rs_scores"].append(rs)
            algo_res["sigma_scores"].append(sigma)
            algo_res["single_sr_scores"].append(sr)
            
            # print(f"  Step {step}: RS={rs:.4f}")
            
        env_results[algo] = algo_res
        
    return env_results

def main():
    # 确保输出目录存在
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

    # 容器：{ 'search': {...}, 'capture': {...} }
    all_env_results = {}

    # 1. 分别计算两种环境
    for env_type in ['search', 'capture']:
        config = CONFIGS[env_type]
        results = compute_env_results(env_type, config)
        if results:
            all_env_results[env_type] = results

    # 2. 保存总的 JSON 数据
    output_json_path = os.path.join(FINAL_OUTPUT_DIR, "robustness_evolution_combined.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_env_results, f, indent=4)
    print(f"\n所有计算完成！合并结果已保存至: {output_json_path}")

    # 3. 统一画图
    plot_robustness_curves(all_env_results, FINAL_OUTPUT_DIR)

    # 4. 统一导出 Excel
    save_excel_for(all_env_results, FINAL_OUTPUT_DIR)

if __name__ == '__main__':
    main()
