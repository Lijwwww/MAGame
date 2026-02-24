# 计算一个环境类型下算法鲁棒性值、权重、消融实验值等
# 已集成合并原始数据和计算博弈权重两个脚本的逻辑

import json
import math
import numpy as np
import os
from scipy.stats import kendalltau

from merge.merge_json import merge_json

# ================= 配置区域 =================
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
    ALGORITHMS = ['TD3', 'PPO', 'DDPG', 'SAC']

    ENVIRONMENTS = ['default', 'obstacle0', 'obstacle1', 
                'friction0', 'friction1']

    METRICS_NAME = ["SR", "avg_step", "avg_collision", "avg_reward"]
    HUMAN_WEIGHTS = {
        "SR": 0.4,
        "avg_step": 0.1,
        "avg_collision": 0.15,
        "avg_reward": 0.35
    }
    RESULT_PATH = 'results/eval/search/merged_results.json'
    OUTPUT_DIR = 'results/robustness_score/search'
    SOURCE_FOLDER = f'results/eval/search'
    ALPHA = 2.0  # 演化强度参数
else:
    ALGORITHMS = ['TD3', 'SAC', 'PPO', 'DDPG', 'CrossQ', 'TRPO', 'TQC']
    ENVIRONMENTS = ['default', 'speed0', 'speed1', 'obstacle0', 'obstacle1', 
            'poses0', 'poses1', 'friction0', 'friction1']
    METRICS_NAME = ["SR", "avg_step", "avg_collision", "avg_evasion", "avg_reward"]
    HUMAN_WEIGHTS = {
        "SR": 0.35,
        "avg_step": 0.1,
        "avg_collision": 0.15,
        "avg_evasion": 0.1,
        "avg_reward": 0.3
    }
    RESULT_PATH = 'results/eval/capture/merged_results.json'
    OUTPUT_DIR = 'results/robustness_score/capture'
    SOURCE_FOLDER = f'results/eval/capture'
    ALPHA = 3.0  # 演化强度参数

# ================= 核心逻辑 =================

def calculate_game_weights(data):

    # Function to filter out list-type metrics and flatten the data
    def filter_metrics(metrics_list):
        filtered = []
        for item in metrics_list:
            if not isinstance(item, list):
                filtered.append(item)
        return filtered

    # Create environment matrices
    env_data = []
    for env in ENVIRONMENTS:
        env_matrix = []
        for algo in ALGORITHMS:
            # Get metrics and filter out list-type ones
            metrics = []
            for metric in METRICS_NAME:
                val = data[algo][env][metric]
                
                # --- 如果是列表，取平均值；否则直接使用 ---
                if isinstance(val, list):
                    metrics.append(np.mean(val))
                else:
                    metrics.append(val)

            env_matrix.append(metrics)
        
        # Only add if we have data for all 3 algorithms in this environment
        # if len(env_matrix) == 3:
        env_data.append(np.array(env_matrix))

    # print(env_data)
    n = env_data[0].shape[1]  # Number of metrics after filtering

    # -----------------------------
    # 2. 计算 payoff 矩阵（Kendall tau 平均）
    # -----------------------------
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
    np.fill_diagonal(payoff_avg, 0)

    # -----------------------------
    # 3. 构建博弈状态空间与转移概率矩阵
    # 每个状态为 (i,j)：排序者为 M_i，评价者为 M_j
    # 总共 n^2 个状态
    # -----------------------------
    num_states = n * n
    T = np.zeros((num_states, num_states))  # 转移矩阵 T

    # 状态映射函数：(i,j) → index
    def state_index(i, j):
        return i * n + j

    for i in range(n):
        for j in range(n):
            s = state_index(i, j)

            # 排序者突变 (i → i')
            for i_prime in range(n):
                if i_prime != i:
                    s_prime = state_index(i_prime, j)
                    delta = payoff_avg[i_prime, j] - payoff_avg[i, j]
                    T[s, s_prime] = np.exp(ALPHA * delta)

            # 评价者突变 (j → j')
            for j_prime in range(n):
                if j_prime != j:
                    s_prime = state_index(i, j_prime)
                    delta = payoff_avg[i, j_prime] - payoff_avg[i, j]
                    T[s, s_prime] = np.exp(-ALPHA * delta)

    # -----------------------------
    # 4. 转移矩阵归一化（每行表示状态的跳转概率分布）
    # -----------------------------
    row_sums = T.sum(axis=1, keepdims=True)
    T = np.where(row_sums > 0, T / row_sums, np.eye(num_states))  # 若无出边，则自环

    # -----------------------------
    # 5. 求稳态分布 π（解 πT = π）
    # 表示每个状态在长期演化中的频率
    # -----------------------------
    eigvals, eigvecs = np.linalg.eig(T.T)
    stat_dist = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    stat_dist = stat_dist[:, 0]
    stat_dist = stat_dist / stat_dist.sum()

    # -----------------------------
    # 6. 提取排序者指标的边缘概率作为权重
    # 即 w_i = sum_j π(i,j)
    # -----------------------------
    indicator_weights = np.zeros(n)
    for i in range(n):
        for j in range(n):
            idx = state_index(i, j)
            indicator_weights[i] += stat_dist[idx]

    # 归一化权重
    weights_normalized = indicator_weights / np.sum(indicator_weights)

    game_weights = dict(zip(METRICS_NAME, weights_normalized))

    # 输出主要结果
    print(
        {
            "payoff_matrix": payoff_avg,
            # "transition_matrix": T,
            # "stationary_distribution": stat_dist.reshape((n, n)),
            "game_weights": game_weights
        }
    )
    # print(f'game_weights: {game_weights}')

    return game_weights

def get_conditional_cv(values):
    """计算条件变异系数 (Conditional Coefficient of Variation)"""
    mu = np.mean(values)
    sigma = np.std(values)
    min_val = np.min(values)
    
    # 保持原有逻辑：处理负值均值的情况
    if min_val < 0:
        return sigma / (mu - min_val)
    else:
        return sigma / mu if mu != 0 else 0

def calculate_combined_weights(game_weights):
    """
    计算最终权重：融合人类先验权重与博弈论计算权重
    """
    combined_weights = {}
    total_weight = 0
    
    # 遍历 HUMAN_WEIGHTS 的键
    for metric, h_w in HUMAN_WEIGHTS.items():
        g_w = game_weights[metric]
        
        # 融合公式
        w = LAMBDA * h_w + (1 - LAMBDA) * g_w
        combined_weights[metric] = w
        total_weight += w
    
    # 归一化
    normalized_weights = {k: v / total_weight for k, v in combined_weights.items()}
    
    print(f'Total normalized weights: {normalized_weights}')
    return normalized_weights

def calculate_algorithm_metrics(data, algorithm, weights):
    """
    计算单个算法的三种鲁棒性指标，并返回详细的统计数据(CV, Sigma)
    """
    algo_data = data[algorithm]
    metrics_values = {k: [] for k in weights.keys()}
    
    # 提取数据
    for env_name, env_data in algo_data.items():
        if env_name not in ENVIRONMENTS:
            continue
        for metric in weights.keys():
            val = env_data[metric]
            metrics_values[metric].append(val)
            
    # 计算中间统计量
    stats = {}
    for metric, values in metrics_values.items():
        stats[metric] = {
            'cv': get_conditional_cv(values),
            'sigma': np.std(values)
        }

    # 1. 计算主鲁棒性评分 (RS) 
    # 公式: - sum(w_i * log(CondCV_i))  =>  sum(w_i * -log(CondCV_i))
    rs_score = 0
    for m in weights:
        cv = stats[m]['cv']
        w = weights[m]
        # 添加极小值保护防止 log(0) 报错，逻辑上 CV=0 代表极度鲁棒
        if cv > 1e-9:
            rs_score += w * (-math.log(cv))
        else:
            rs_score += w * (-math.log(1e-9))

    # 2. 消融实验 B: 基于标准差 (Sigma)
    # Logic: sum(w * -log(sigma))
    sigma_score = 0
    for m in weights:
        s = stats[m]['sigma']
        if s > 1e-9: 
            sigma_score += weights[m] * (-math.log(s))
        else:
            sigma_score += weights[m] * (-math.log(1e-9))

    # 3. 消融实验 C: 仅 SR
    # Logic: -log(CV_SR)
    cv_sr = stats['SR']['cv']
    single_sr_score = -math.log(cv_sr) if cv_sr > 1e-9 else -math.log(1e-9)

    print(f"Algorithm: {algorithm} | RS: {rs_score:.4f} | Sigma: {sigma_score:.4f} | Single(SR): {single_sr_score:.4f}")


    # 4. 消融实验 D: Equal Weights (等权重)
    # 逻辑: 权重全部变成 1/N
    ablation_equal_w = 0
    n_metrics = len(weights)
    for m in weights:
        cv = stats[m]['cv']
        val = -math.log(cv) if cv > 1e-9 else -math.log(1e-9)
        ablation_equal_w += (1.0 / n_metrics) * val

    # 5. 消融实验 E: No Log (线性加权)
    # 逻辑: 直接加权求和 CondCV
    ablation_no_log = 0
    for m in weights:
        ablation_no_log += weights[m] * stats[m]['cv']
    ablation_no_log = 1 - ablation_no_log # 变成越大越好
    
    # 构造并返回结果字典，包含详细统计信息
    return {
        "robustness_score": rs_score,
        "ablation_sigma": sigma_score,
        "ablation_single_sr": single_sr_score,
        "ablation_equal_weights": ablation_equal_w,
        "ablation_no_log": ablation_no_log,
        "details_cv": {m: stats[m]['cv'] for m in weights},    # 仅提取CV
        "details_sigma": {m: stats[m]['sigma'] for m in weights} # 仅提取Sigma
    }

def main():
    # 可选：调用merge_json脚本，先合并原始测试数据
    # merge_json(SOURCE_FOLDER, RESULT_PATH, ALGORITHMS)

    # 1. 读取数据
    if not os.path.exists(RESULT_PATH):
        print(f"Error: File not found at {RESULT_PATH}")
        return

    with open(RESULT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 计算权重
    game_weights = calculate_game_weights(data)
    weights = calculate_combined_weights(game_weights)

    # 3. 计算所有算法的指标
    all_results = {}
    for alg in data:
        all_results[alg] = calculate_algorithm_metrics(data, alg, weights)

    
    # 4. 保存权重信息
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 结果文件映射表
    file_map = {
        "robustness_score": "robustness_main.json",
        "ablation_sigma": "ablation_sigma.json",
        "ablation_single_sr": "ablation_single_sr.json",
        "ablation_equal_weights": "ablation_equal_w.json",
        "ablation_no_log": "ablation_no_log.json",
        "details_cv": "cv_details.json", 
        "details_sigma": "sigma_details.json" 
    }

    for key, filename in file_map.items():
        # 提取特定指标构建新字典
        metric_data = {alg: res[key] for alg, res in all_results.items()}
        
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metric_data, f, indent=4, ensure_ascii=False)
        
        print(f"Saved {key} to {output_path}")

    # 5. 保存权重信息
    # 定义权重文件映射
    weights_map = {
        "game_weights.json": game_weights,
        "final_combined_weights.json": weights
    }

    for filename, w_data in weights_map.items():
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(w_data, f, indent=4, ensure_ascii=False)
        print(f"Saved weights to {output_path}")

    # 额外打印一下详细数据方便查看
    print("\n=== Detailed Statistics Saved ===")
    # print("CV Details:", json.dumps({alg: res['details_cv'] for alg, res in all_results.items()}, indent=2))
    # print("Sigma Details:", json.dumps({alg: res['details_sigma'] for alg, res in all_results.items()}, indent=2))

if __name__ == '__main__':
    main()