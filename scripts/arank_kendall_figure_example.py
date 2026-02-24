import numpy as np

def cal_weight_custom():
    # -----------------------------
    # 1. 自定义 Payoff 矩阵 (根据提供的图片)
    # -----------------------------
    # 矩阵结构:
    # [ 0, -1,  2]
    # [ 1,  0,  1]
    # [-2,  1,  0]
    payoff_avg = np.array([
        [0.0, -1.0, 2.0],
        [1.0,  0.0, 1.0],
        [-2.0, 1.0, 0.0]
    ])

    # 获取指标数量 (n=3)
    n = payoff_avg.shape[0]
    
    # 为了演示，生成对应的虚拟指标名称
    metrics_name = [f"Metric_{i+1}" for i in range(n)]

    # -----------------------------
    # 2. 构建博弈状态空间与转移概率矩阵
    # 每个状态为 (i,j)：排序者为 M_i，评价者为 M_j
    # 总共 n^2 个状态
    # -----------------------------
    num_states = n * n
    T = np.zeros((num_states, num_states))  # 转移矩阵 T

    # 状态映射函数：(i,j) → index
    def state_index(i, j):
        return i * n + j

    alpha = 3.0  # 演化强度参数

    for i in range(n):
        for j in range(n):
            s = state_index(i, j)

            # 排序者突变 (i → i')
            for i_prime in range(n):
                if i_prime != i:
                    s_prime = state_index(i_prime, j)
                    delta = payoff_avg[i_prime, j] - payoff_avg[i, j]
                    T[s, s_prime] = np.exp(alpha * delta)

            # 评价者突变 (j → j')
            for j_prime in range(n):
                if j_prime != j:
                    s_prime = state_index(i, j_prime)
                    delta = payoff_avg[i, j_prime] - payoff_avg[i, j]
                    T[s, s_prime] = np.exp(-alpha * delta)

    # -----------------------------
    # 3. 转移矩阵归一化（每行表示状态的跳转概率分布）
    # -----------------------------
    row_sums = T.sum(axis=1, keepdims=True)
    T = np.where(row_sums > 0, T / row_sums, np.eye(num_states))  # 若无出边，则自环

    # -----------------------------
    # 4. 求稳态分布 π（解 πT = π）
    # 表示每个状态在长期演化中的频率
    # -----------------------------
    eigvals, eigvecs = np.linalg.eig(T.T)
    # 找到特征值为 1 对应的特征向量
    stat_dist = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    
    # 这里的 stat_dist 可能是多维的，取第一列并展平
    stat_dist = stat_dist[:, 0]
    stat_dist = stat_dist / stat_dist.sum()

    # -----------------------------
    # 5. 提取排序者指标的边缘概率作为权重
    # 即 w_i = sum_j π(i,j)
    # -----------------------------
    indicator_weights = np.zeros(n)
    for i in range(n):
        for j in range(n):
            idx = state_index(i, j)
            indicator_weights[i] += stat_dist[idx]

    # 归一化权重
    weights_normalized = indicator_weights / np.sum(indicator_weights)

    # -----------------------------
    # 6. 打印结果
    # -----------------------------
    # 这里将 numpy array 转换为 list 以便更清晰地打印，或者直接打印 array 也可以
    print("Calculation Results:")
    print(
        {
            "payoff_matrix": payoff_avg,
            "transition_matrix": T, # 如果需要打印巨大的转移矩阵，可以取消注释
            "stationary_distribution": stat_dist.reshape((n, n)),
            "indicator_weights_normalized": weights_normalized
        }
    )
    
    print(f'\nFinal Weights List: {weights_normalized}')

    return weights_normalized, metrics_name

if __name__ == '__main__':
    # 不需要读取文件，直接运行计算
    cal_weight_custom()