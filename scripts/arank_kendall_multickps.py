import json
import os
import numpy as np
from scipy.stats import kendalltau

# ================= 配置区域 =================
ALGORITHMS = ['TD3']
ENVIRONMENTS = [
    'default', 'speed0', 'speed1', 'obstacle0', 'obstacle1', 
    'poses0', 'poses1', 'friction0', 'friction1'
]
METRICS_NAME = ["SR", "avg_step", "avg_collision", "avg_evasion", "avg_reward"]
JSON_PATH = '/workspace/omniisaacgymenvs/PursuitSim3D/results/eval_multi_ckps/merged_results.json'
# ===========================================

def get_step_int(step_str):
    return int(step_str.split('_')[0])

def calculate_game_weights(snapshot):
    """
    复用博弈算法逻辑
    """
    # -----------------------------
    # 1. 构造不同环境的算法评分数据 (env_data)
    # -----------------------------
    env_data = []
    for env in ENVIRONMENTS:
        env_matrix = []
        for algo in ALGORITHMS:
            # 获取指标，列表取均值，数值直接用
            metrics = []
            for metric in METRICS_NAME:
                val = snapshot[algo][env][metric]
                if isinstance(val, list):
                    metrics.append(np.mean(val))
                else:
                    metrics.append(val)
            env_matrix.append(metrics)
        env_data.append(np.array(env_matrix))

    n = env_data[0].shape[1]  # 指标数量

    # -----------------------------
    # 2. 计算 payoff 矩阵（Kendall tau 平均）
    # -----------------------------
    payoff_avg = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            taus = []
            for env in env_data:
                tau, _ = kendalltau(env[:, i], env[:, j])
                taus.append(tau)
            payoff_avg[i, j] = np.mean(taus)

    # -----------------------------
    # 3. 构建博弈状态空间与转移概率矩阵
    # -----------------------------
    num_states = n * n
    T = np.zeros((num_states, num_states))

    # 状态映射函数：(i,j) -> index
    def state_index(i, j):
        return i * n + j

    alpha = 5.0  # 演化强度参数

    for i in range(n):
        for j in range(n):
            s = state_index(i, j)

            # 排序者突变 (i -> i')
            for i_prime in range(n):
                if i_prime != i:
                    s_prime = state_index(i_prime, j)
                    delta = payoff_avg[i_prime, j] - payoff_avg[i, j]
                    T[s, s_prime] = np.exp(alpha * delta)

            # 评价者突变 (j -> j')
            for j_prime in range(n):
                if j_prime != j:
                    s_prime = state_index(i, j_prime)
                    delta = payoff_avg[i, j_prime] - payoff_avg[i, j]
                    T[s, s_prime] = np.exp(-alpha * delta)

    # -----------------------------
    # 4. 转移矩阵归一化
    # -----------------------------
    row_sums = T.sum(axis=1, keepdims=True)
    T = np.where(row_sums > 0, T / row_sums, np.eye(num_states))

    # -----------------------------
    # 5. 求稳态分布 π
    # -----------------------------
    eigvals, eigvecs = np.linalg.eig(T.T)
    # 提取特征值为1对应的特征向量
    stat_dist = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    # 容错：如果结果是二维的取第一列，否则保持
    if stat_dist.ndim > 1:
        stat_dist = stat_dist[:, 0]
    stat_dist = stat_dist / stat_dist.sum()

    # -----------------------------
    # 6. 提取排序者指标的边缘概率作为权重
    # -----------------------------
    indicator_weights = np.zeros(n)
    for i in range(n):
        for j in range(n):
            idx = state_index(i, j)
            indicator_weights[i] += stat_dist[idx]

    # 归一化权重
    weights_normalized = indicator_weights / np.sum(indicator_weights)
    
    return weights_normalized


if __name__ == '__main__':
    print(f"读取数据: {JSON_PATH}")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    # 假设第一个算法的数据包含所有steps，不再做交集检查
    base_algo = ALGORITHMS[0]
    steps = sorted(full_data[base_algo].keys(), key=get_step_int)
    print(f"检测到 {len(steps)} 个Step，开始计算...")

    # 输出文件路径
    output_txt = os.path.join(os.path.dirname(JSON_PATH), "weights_results.txt")
    
    with open(output_txt, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("Step\t" + "\t".join(METRICS_NAME) + "\n")
        
        for step in steps:
            # 构造快照：从full_data中提取当前step的所有算法数据
            # 这里假设所有算法和环境数据都是齐全的，不做 try-except
            snapshot = {algo: full_data[algo][step] for algo in ALGORITHMS}
            
            # 计算权重
            w = calculate_game_weights(snapshot)
            
            # 写入结果
            row_str = "\t".join([f"{x:.4f}" for x in w])
            f.write(f"{step}\t{row_str}\n")
            print(f"Step {step}: 完成")

    print(f"结果已保存至: {output_txt}")