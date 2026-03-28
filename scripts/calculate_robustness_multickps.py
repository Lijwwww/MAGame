# 计算所有步数下算法鲁棒性值、权重、消融实验值等，并画出算法鲁棒性值随训练步数的曲线图
# 目前方案是读取最后一个训练节点的博弈权重，因此权重计算相关的代码全都用不到，但这里仍做保留
# 已集成合并原始数据（每步数据合并成一个文件）和计算博弈权重两个脚本的逻辑
# 若已有合并后文件，则可直接通过RESULT_PATH读数据画图

import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kendalltau
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm

from merge.merge_json_multickps import merge_json_multickps

# ================= Linux 字体配置 =================
# 1. 将字体文件注册到 Matplotlib 的字体管理器中
# 建议使用变量统一路径，避免大小写或路径不一致
font_abs_path = '/workspace/omniisaacgymenvs/PursuitSim3D/assets/Fonts/SIMSUN.TTC'
times_abs_path = '/workspace/omniisaacgymenvs/PursuitSim3D/assets/Fonts/TIMES.TTF'

# 注册字体
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

# ================= Windows 字体配置 =================
# # 1. 字体设置
# # 逻辑：优先使用 'Times New Roman'，遇到它无法显示的字符（如汉字），
# # 就会自动去用列表里的第二个字体 'SimSun' (宋体)
# plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']

# # 2. 解决负号显示为方块的问题
# plt.rcParams['axes.unicode_minus'] = False

# # 3. (可选) 设置数学公式字体，使其与 Times 风格一致
# plt.rcParams['mathtext.fontset'] = 'stix'

# ================= 配置区域 =================
# 选择了读取最后一个训练节点的博弈权重，因此不做权重计算，下面的参数除了main函数涉及到的，都没用了
LAMBDA = 0.5  # 权重融合系数

METRIC_SIGNS = {
    "SR": 1,
    "avg_step": -1,
    "avg_collision": -1,
    "avg_evasion": -1,
    "avg_reward": 1
}

ENV_TYPE = 'capture'
# ENV_TYPE = 'search'

if ENV_TYPE == 'search':
    ALGORITHMS = ['TD3', 'PPO']

    ENVIRONMENTS = ['default', 'obstacle0', 'obstacle1', 
                'friction0', 'friction1']

    METRICS_NAME = ["SR", "avg_step", "avg_collision", "avg_reward"]
    HUMAN_WEIGHTS = {
        "SR": 0.45,
        "avg_step": 0.1,
        "avg_collision": 0.1,
        "avg_reward": 0.35
    }
    RESULT_PATH = 'results/eval_multi_ckps/search/merged_results.json'
    OUTPUT_DIR = 'results/robustness_score_multickps/search'
    GAME_WEIGHTS_PATH = 'results/robustness_score/search/game_weights.json'
    SOURCE_FOLDER = f'results/eval_multi_ckps/search'
    ALPHA = 2.0  # 演化强度参数
else:
    ALGORITHMS = ['TD3', 'PPO']
    ENVIRONMENTS = ['default', 'speed0', 'speed1', 'obstacle0', 'obstacle1', 
            'poses0', 'poses1', 'friction0', 'friction1']
    METRICS_NAME = ["SR", "avg_step", "avg_collision", "avg_evasion", "avg_reward"]
    HUMAN_WEIGHTS = {
        "SR": 0.4,
        "avg_step": 0.1,
        "avg_collision": 0.1,
        "avg_evasion": 0.1,
        "avg_reward": 0.3
    }
    RESULT_PATH = 'results/eval_multi_ckps/capture/merged_results.json'
    OUTPUT_DIR = 'results/robustness_score_multickps/capture'
    SOURCE_FOLDER = f'results/eval_multi_ckps/capture'
    GAME_WEIGHTS_PATH = 'results/robustness_score/capture/game_weights.json'
    ALPHA = 5.0  # 演化强度参数
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
    """
    计算博弈论权重，并直接返回字典 {metric: weight}
    """
    # 1. 构造矩阵
    env_data = []
    for env in ENVIRONMENTS:
        env_matrix = []
        for algo in ALGORITHMS:
            metrics = []
            for metric in METRICS_NAME:
                # 获取数据 (snapshot 已经是当前step的数据)
                val = snapshot[algo][env][metric]
                metrics.append(np.mean(val) if isinstance(val, list) else val)
            env_matrix.append(metrics)
        env_data.append(np.array(env_matrix))

    n = len(METRICS_NAME)

    # 2. Kendall Tau Payoff
    payoff_avg = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            taus = []
            for env in env_data:
                # 获取指标名称
                metric_i_name = METRICS_NAME[i]
                metric_j_name = METRICS_NAME[j]
                
                # 获取数据并应用符号修正
                data_i = env[:, i] * METRIC_SIGNS[metric_i_name]
                data_j = env[:, j] * METRIC_SIGNS[metric_j_name]

                tau, _ = kendalltau(data_i, data_j)
                # tau, _ = kendalltau(env[:, i], env[:, j])
                taus.append(tau)
            payoff_avg[i, j] = np.mean(taus)

    # 3. 演化博弈
    num_states = n * n
    T = np.zeros((num_states, num_states))
    def state_index(i, j): return i * n + j
    alpha = 5.0

    for i in range(n):
        for j in range(n):
            s = state_index(i, j)
            for i_prime in range(n):
                if i_prime != i:
                    delta = payoff_avg[i_prime, j] - payoff_avg[i, j]
                    T[s, state_index(i_prime, j)] = np.exp(alpha * delta)
            for j_prime in range(n):
                if j_prime != j:
                    delta = payoff_avg[i, j_prime] - payoff_avg[i, j]
                    T[s, state_index(i, j_prime)] = np.exp(-alpha * delta)

    # 4. 稳态分布
    row_sums = T.sum(axis=1, keepdims=True)
    T = np.where(row_sums > 0, T / row_sums, np.eye(num_states))
    eigvals, eigvecs = np.linalg.eig(T.T)
    stat_dist = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    if stat_dist.ndim > 1: stat_dist = stat_dist[:, 0]
    stat_dist = stat_dist / stat_dist.sum()

    # 5. 提取权重
    indicator_weights = np.zeros(n)
    for i in range(n):
        for j in range(n):
            indicator_weights[i] += stat_dist[state_index(i, j)]
    
    weights_normalized = indicator_weights / np.sum(indicator_weights)
    
    return dict(zip(METRICS_NAME, weights_normalized))

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

def calculate_combined_weights(game_weights_dict):
    """融合人类权重和博弈权重"""
    combined_weights = {}
    total_weight = 0
    for metric, h_w in HUMAN_WEIGHTS.items():
        g_w = game_weights_dict.get(metric, 0) # 容错
        w = LAMBDA * h_w + (1 - LAMBDA) * g_w
        combined_weights[metric] = w
        total_weight += w
    return {k: v / total_weight for k, v in combined_weights.items()}

def calculate_step_metrics(step_data, weights):
    """
    计算单步的鲁棒性指标
    step_data: { "default": {...}, "speed0": {...} } (即某个算法在某一步的所有环境数据)
    """
    metrics_values = {k: [] for k in weights.keys()}
    
    # 提取数据
    for env_name in ENVIRONMENTS:
        if env_name not in step_data: continue # 容错
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

    # 1. RS Score (Log inside)
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
def plot_robustness_curves(final_results, output_dir):
    """
    绘制各算法随训练步数的鲁棒性变化曲线
    """

    plt.figure(figsize=(12, 8))

    # 设置风格 (可选)
    # plt.style.use('seaborn-v0_8-paper') 
    
    marker_list = ['o', 's', '^', 'D', '*', 'v', 'p']

    for i, (algo, data) in enumerate(final_results.items()):
        steps = data['steps']
        scores = data['rs_scores']
        current_marker = marker_list[i % len(marker_list)]
        
        # 绘制曲线，添加标记点
        plt.plot(steps, scores, marker=current_marker, linestyle='-', label=algo, linewidth=2)

    # plt.title('Robustness Score Evolution over Training Steps', fontsize=14)
    plt.xlabel('训练步数/千步', fontsize=24)
    plt.ylabel('鲁棒性评分', fontsize=24)
    plt.legend(fontsize=22)
    # 在 plt.xticks() 前添加以下代码
    def format_thousands(x, pos):
        """将步数转换为千单位"""
        return f'{int(x/1000)}'
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_thousands))
    plt.xticks(fontsize=18) # 设置 X 轴刻度字体大小
    plt.yticks(fontsize=18) # 设置 Y 轴刻度字体大小
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图片
    output_path = os.path.join(output_dir, 'robustness_trend.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"趋势图已保存至: {output_path}")


def save_excel_for(final_results, output_dir):
    """
    将结果整理为 DataFrame 并导出 Excel，方便 Origin 画图
    Index: Step
    Columns: 各算法的 Robustness Score
    """
    print(f"正在生成 Origin 绘图数据...")
    
    # 使用 Pandas 的字典合并功能，自动对齐不同算法的 Step
    data_dict = {}
    for algo, res in final_results.items():
        # 创建 Series，索引是 steps，值是 rs_scores
        # 这样即使不同算法的 checkpoint 步数不完全一致，也能正确对齐
        series = pd.Series(data=res['rs_scores'], index=res['steps'], name=algo)
        data_dict[algo] = series

    # 合并为一个 DataFrame
    df = pd.DataFrame(data_dict)
    
    # 按步数排序
    df.sort_index(inplace=True)
    df.index.name = 'Step' # X轴名称

    # 保存
    output_path = os.path.join(output_dir, 'robustness_curves.xlsx')
    df.to_excel(output_path)
    print(f"Origin 绘图数据已保存至: {output_path}")

# -------------------------------------------------------------------------
# 4. 主流程
# -------------------------------------------------------------------------
def main():
    # 可选：调用merge_json脚本，先合并原始测试数据
    # merge_json_multickps(SOURCE_FOLDER, RESULT_PATH, ALGORITHMS)

    print(f"读取数据: {RESULT_PATH}")
    with open(RESULT_PATH, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    # 准备结果容器
    final_results = {}

    for algo in ALGORITHMS:
        if algo not in full_data: continue
        
        print(f"\n处理算法: {algo} ...")
        
        # 获取并排序步数
        steps = sorted(full_data[algo].keys(), key=get_step_int)
        
        algo_results = {
            "steps": [], # 存储步数 (int)
            "rs_scores": [],
            "sigma_scores": [],
            "single_sr_scores": [],
            "combined_weights": [], 
            "game_weights": []      
        }
        
        for step in steps:
            step_int = get_step_int(step)
            
            # 1. 获取当前步的快照 (用于计算权重)
            snapshot_for_weight = {algo: full_data[algo][step]} 
            
            # 2. 计算当前步的动态权重
            # game_weights = calculate_game_weights(snapshot_for_weight)  # 每个节点重算一遍（舍弃）
            with open(GAME_WEIGHTS_PATH, 'r', encoding='utf-8') as f:     # 读取最后一个节点权重
                game_weights = json.load(f)
            final_weights = calculate_combined_weights(game_weights)
            
            # 3. 获取当前步的详细数据 (用于计算鲁棒性)
            step_data_for_metric = full_data[algo][step]
            
            # 4. 计算鲁棒性
            rs, sigma, sr = calculate_step_metrics(step_data_for_metric, final_weights)
            
            # 5. 记录
            algo_results["steps"].append(step_int)
            algo_results["rs_scores"].append(rs)
            algo_results["sigma_scores"].append(sigma)
            algo_results["single_sr_scores"].append(sr)
            algo_results["combined_weights"].append(final_weights)
            algo_results["game_weights"].append(game_weights)
            
            print(f"  Step {step}: RS={rs:.4f}")
            # print(f"    Game Weights: {game_weights}")
            # print(f"    Combined Weights: {final_weights}")
            
        final_results[algo] = algo_results

    # 保存结果
    '''
    {
        "TD3": {
            "steps": [20000, 40000, ...],
            "rs_scores": [1.5, 1.8, ...],
            "sigma_scores": [...],
            ...
        }
    }
    '''
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    output_path = os.path.join(OUTPUT_DIR, "robustness_evolution.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\n计算完成！结果已保存至: {output_path}")

    # 调用绘图函数
    plot_robustness_curves(final_results, OUTPUT_DIR)

    save_excel_for(final_results, OUTPUT_DIR)

if __name__ == '__main__':
    main()