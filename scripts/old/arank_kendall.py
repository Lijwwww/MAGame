import json
import numpy as np
from scipy.stats import kendalltau


# data = {
#     "TD3": {
#         "default": {"SR": 0.9286, "avg_step": 23.9709, "avg_time": 7.5456, "avg_collision": 1.0847, "avg_evasion": 1.6601, "avg_reward": 7.9363},
#         "obstacle0": {"SR": 0.7526, "avg_step": 30.9937, "avg_time": 10.1394, "avg_collision": 1.1090, "avg_evasion": 2.5430, "avg_reward": 1.0624},
#         "obstacle1": {"SR": 0.9492, "avg_step": 23.6921, "avg_time": 7.4596, "avg_collision": 0.9880, "avg_evasion": 1.6816, "avg_reward": 7.9786},
#         "friction0": {"SR": 0.9742, "avg_step": 21.4992, "avg_time": 6.6699, "avg_collision": 0.9354, "avg_evasion": 1.4281, "avg_reward": 9.2014},
#         "friction1": {"SR": 0.9183, "avg_step": 24.8990, "avg_time": 8.4668, "avg_collision": 0.9423, "avg_evasion": 1.7572, "avg_reward": 7.6875},
#         "poses0": {"SR": 0.9433, "avg_step": 28.9422, "avg_time": 9.1757, "avg_collision": 1.7731, "avg_evasion": 1.0956, "avg_reward": -3.2637},
#         "poses1": {"SR": 0.9916, "avg_step": 10.9748, "avg_time": 3.9729, "avg_collision": 0.6076, "avg_evasion": 0.7324, "avg_reward": 39.8794},
#         "speed0": {"SR": 0.9422, "avg_step": 24.0828, "avg_time": 8.0678, "avg_collision": 1.0000, "avg_evasion": 2.1025, "avg_reward": 7.0013},
#         "speed1": {"SR": 0.9516, "avg_step": 24.1111, "avg_time": 8.0506, "avg_collision": 0.9869, "avg_evasion": 1.3830, "avg_reward": 7.8562}
#     },
#     "SAC": {
#         "default": {"SR": 0.8962, "avg_step": 25.2990, "avg_time": 7.8754, "avg_collision": 1.0290, "avg_evasion": 1.7181, "avg_reward": 6.3574},
#         "obstacle0": {"SR": 0.7917, "avg_step": 28.8565, "avg_time": 9.5501, "avg_collision": 1.0255, "avg_evasion": 2.0162, "avg_reward": 3.9161},
#         "obstacle1": {"SR": 0.9646, "avg_step": 22.0919, "avg_time": 7.0073, "avg_collision": 0.7567, "avg_evasion": 1.3380, "avg_reward": 10.7572},
#         "friction0": {"SR": 0.9840, "avg_step": 19.9537, "avg_time": 6.2426, "avg_collision": 0.6981, "avg_evasion": 1.1965, "avg_reward": 11.9062},
#         "friction1": {"SR": 0.9634, "avg_step": 23.1274, "avg_time": 7.9369, "avg_collision": 0.8290, "avg_evasion": 1.3691, "avg_reward": 9.8835},
#         "poses0": {"SR": 0.9694, "avg_step": 27.8196, "avg_time": 8.8943, "avg_collision": 1.9146, "avg_evasion": 0.5148, "avg_reward": -1.9305},
#         "poses1": {"SR": 0.9884, "avg_step": 11.7353, "avg_time": 4.2220, "avg_collision": 0.6082, "avg_evasion": 0.4811, "avg_reward": 36.7285},
#         "speed0": {"SR": 0.9635, "avg_step": 22.3477, "avg_time": 7.5787, "avg_collision": 0.7799, "avg_evasion": 1.6836, "avg_reward": 10.3312},
#         "speed1": {"SR": 0.9770, "avg_step": 21.6692, "avg_time": 7.3313, "avg_collision": 0.7522, "avg_evasion": 0.9987, "avg_reward": 10.6003}
#     },
#     "PPO": {
#         "default": {"SR": 0.7621, "avg_step": 32.3790, "avg_time": 9.7390, "avg_collision": 1.9866, "avg_evasion": 2.5161, "avg_reward": -0.6926},
#         "obstacle0": {"SR": 0.4388, "avg_step": 40.7038, "avg_time": 12.5585, "avg_collision": 1.7261, "avg_evasion": 3.2829, "avg_reward": -10.7284},
#         "obstacle1": {"SR": 0.7512, "avg_step": 32.2934, "avg_time": 9.3127, "avg_collision": 1.9631, "avg_evasion": 2.4716, "avg_reward": -1.0622},
#         "friction0": {"SR": 0.8425, "avg_step": 28.6301, "avg_time": 8.0717, "avg_collision": 1.7858, "avg_evasion": 2.1611, "avg_reward": 1.6530},
#         "friction1": {"SR": 0.6952, "avg_step": 34.8710, "avg_time": 10.8304, "avg_collision": 1.6788, "avg_evasion": 2.6811, "avg_reward": -3.0204},
#         "poses0": {"SR": 0.7272, "avg_step": 36.1998, "avg_time": 10.6512, "avg_collision": 2.5693, "avg_evasion": 1.9882, "avg_reward": -7.9113},
#         "poses1": {"SR": 0.9176, "avg_step": 17.4186, "avg_time": 5.6002, "avg_collision": 1.1221, "avg_evasion": 1.8319, "avg_reward": 26.7384},
#         "speed0": {"SR": 0.7898, "avg_step": 31.0148, "avg_time": 9.5830, "avg_collision": 1.7116, "avg_evasion": 2.7601, "avg_reward": -0.4908},
#         "speed1": {"SR": 0.6449, "avg_step": 36.2476, "avg_time": 10.5129, "avg_collision": 2.3007, "avg_evasion": 2.3918, "avg_reward": -4.9954}
#     },
#     'DDPG': {
#         "default": { "SR": 0.9213, "avg_step": 24.2373, "avg_time": 6.1017, "avg_collision": 0.8867, "avg_evasion": 1.5453, "avg_reward": 8.3422},
#         "obstacle0": { "SR": 0.8086, "avg_step": 29.7815, "avg_time": 9.9603, "avg_collision": 0.9572, "avg_evasion": 2.2613, "avg_reward": 2.9913},
#         "obstacle1": { "SR": 0.9632, "avg_step": 22.8395, "avg_time": 8.0088, "avg_collision": 1.1605, "avg_evasion": 1.4566, "avg_reward": 9.0974},
#         "friction0": { "SR": 0.9758, "avg_step": 20.8583, "avg_time": 6.5168, "avg_collision": 0.9211, "avg_evasion": 1.3623, "avg_reward": 10.6308},
#         "friction1": { "SR": 0.9394, "avg_step": 24.6594, "avg_time": 8.2874, "avg_collision": 0.6970, "avg_evasion": 1.5842, "avg_reward": 8.1574},
#         "poses0": { "SR": 0.9484, "avg_step": 31.4290, "avg_time": 10.2664, "avg_collision": 1.2817, "avg_evasion": 1.0183, "avg_reward": -2.7765},
#         "poses1": { "SR": 0.9937, "avg_step": 13.0953, "avg_time": 4.6069, "avg_collision": 0.6168, "avg_evasion": 0.9853, "avg_reward": 30.3624},
#         "speed0": { "SR": 0.9595, "avg_step": 23.4530, "avg_time": 7.8778, "avg_collision": 0.9791, "avg_evasion": 1.8668, "avg_reward": 9.0029},
#         "speed1": { "SR": 0.9579, "avg_step": 23.8138, "avg_time": 7.9385, "avg_collision": 0.8686, "avg_evasion": 1.2066, "avg_reward": 8.6578}
#     },
#     'CrossQ': {
#         "default": {"SR": 0.8670, "avg_step": 28.3890, "avg_time": 8.8835, "avg_collision": 1.1614, "avg_evasion": 2.2541, "avg_reward": 3.1212},
#         "obstacle1": {"SR": 0.9372, "avg_step": 24.8146, "avg_time": 7.9283, "avg_collision": 0.9591, "avg_evasion": 1.8978, "avg_reward": 6.5443},
#         "obstacle0": {"SR": 0.7377, "avg_step": 31.9461, "avg_time": 10.3786, "avg_collision": 1.0796, "avg_evasion": 2.5293, "avg_reward": -0.4772},
#         "friction0": {"SR": 0.9630, "avg_step": 23.4784, "avg_time": 7.0465, "avg_collision": 1.0926, "avg_evasion": 1.7716, "avg_reward": 7.1628},
#         "friction1": {"SR": 0.9251, "avg_step": 26.9145, "avg_time": 9.0314, "avg_collision": 0.8934, "avg_evasion": 1.9836, "avg_reward": 5.1345},
#         "poses0": {"SR": 0.9101, "avg_step": 30.9725, "avg_time": 10.2080, "avg_collision": 1.7376, "avg_evasion": 1.2402, "avg_reward": -4.5467},
#         "poses1": {"SR": 0.9727, "avg_step": 17.4679, "avg_time": 5.8521, "avg_collision": 0.8475, "avg_evasion": 1.4238, "avg_reward": 26.6299},
#         "speed0": {"SR": 0.9186, "avg_step": 25.6081, "avg_time": 7.7638, "avg_collision": 0.9835, "avg_evasion": 2.3155, "avg_reward": 5.6037},
#         "speed1": {"SR": 0.9347, "avg_step": 27.3790, "avg_time": 8.9406, "avg_collision": 1.0653, "avg_evasion": 1.7772, "avg_reward": 4.6383}
#     },
#     'TRPO': {
#         "default": {"SR": 0.8714, "avg_step": 28.7743, "avg_time": 8.8304, "avg_collision": 1.4068, "avg_evasion": 2.1050, "avg_reward": 1.2199},
#         "obstacle0": {"SR": 0.5659, "avg_step": 37.4574, "avg_time": 12.3207, "avg_collision": 1.5504, "avg_evasion": 2.9535, "avg_reward": -6.8517},
#         "obstacle1": {"SR": 0.8538, "avg_step": 28.7646, "avg_time": 9.4867, "avg_collision": 1.4123, "avg_evasion": 1.9354, "avg_reward": 1.3095},
#         "friction0": {"SR": 0.9307, "avg_step": 25.2078, "avg_time": 8.5023, "avg_collision": 1.2433, "avg_evasion": 1.8153, "avg_reward": 4.0239},
#         "friction1": {"SR": 0.8299, "avg_step": 30.6598, "avg_time": 9.6026, "avg_collision": 1.3536, "avg_evasion": 2.1057, "avg_reward": 0.0389},
#         "poses0": {"SR": 0.8209, "avg_step": 34.1190, "avg_time": 10.3974, "avg_collision": 2.4037, "avg_evasion": 1.5186, "avg_reward": -6.3588},
#         "poses1": {"SR": 0.9620, "avg_step": 14.4984, "avg_time": 4.9759, "avg_collision": 0.7698, "avg_evasion": 1.1616, "avg_reward": 30.9652},
#         "speed0": {"SR": 0.8825, "avg_step": 26.9098, "avg_time": 8.8003, "avg_collision": 1.2951, "avg_evasion": 2.2842, "avg_reward": 3.0404},
#         "speed1": {"SR": 0.7848, "avg_step": 32.8102, "avg_time": 10.0465, "avg_collision": 1.7179, "avg_evasion": 1.9746, "avg_reward": -2.1667}
#     },
#     'TQC': {
#         "default": {"SR": 0.8335, "avg_step": 26.8064, "avg_time": 8.0883, "avg_collision": 0.9711, "avg_evasion": 1.9130, "avg_reward": 5.4150},
#         "obstacle0": {"SR": 0.8575, "avg_step": 25.6713, "avg_time": 8.6084, "avg_collision": 0.9747, "avg_evasion": 1.6644, "avg_reward": 8.2188},
#         "obstacle1": {"SR": 0.9671, "avg_step": 20.9700, "avg_time": 7.3927, "avg_collision": 0.8000, "avg_evasion": 1.1800, "avg_reward": 11.5667},
#         "friction0": {"SR": 0.9864, "avg_step": 19.6636, "avg_time": 6.2140, "avg_collision": 0.8485, "avg_evasion": 1.1833, "avg_reward": 12.9017},
#         "friction1": {"SR": 0.9692, "avg_step": 22.3314, "avg_time": 7.6469, "avg_collision": 0.7515, "avg_evasion": 1.1527, "avg_reward": 11.5788},
#         "poses0": {"SR": 0.9687, "avg_step": 27.0898, "avg_time": 9.0217, "avg_collision": 1.9353, "avg_evasion": 0.6827, "avg_reward": -2.1267},
#         "poses1": {"SR": 0.9979, "avg_step": 10.4580, "avg_time": 3.7683, "avg_collision": 0.6134, "avg_evasion": 0.4181, "avg_reward": 37.9413},
#         "speed0": {"SR": 0.9623, "avg_step": 21.5423, "avg_time": 6.9651, "avg_collision": 0.9311, "avg_evasion": 1.6632, "avg_reward": 11.3038},
#         "speed1": {"SR": 0.9720, "avg_step": 21.6438, "avg_time": 7.3131, "avg_collision": 0.8384, "avg_evasion": 1.0191, "avg_reward": 11.4671}
#     }
# }

def calculate_game_weights(data):
    # -----------------------------
    # 1. 构造不同环境的算法评分数据
    # -----------------------------
    # algorithms = ['TD3', 'SAC', 'PPO', 'DDPG', 'CrossQ', 'TRPO', 'TQC']
    algorithms = ['TD3', 'PPO', 'DDPG']
    # environments = ['default', 'speed0', 'speed1', 'obstacle0', 'obstacle1', 
    #             'poses0', 'poses1', 'friction0', 'friction1']
    environments = ['default', 'obstacle0', 'obstacle1', 
                'poses0', 'poses1', 'friction0', 'friction1']
    # metrics_name = ["SR", "avg_step", "avg_collision", "avg_evasion", "avg_reward"]
    metrics_name = ["SR", "avg_step", "avg_collision", "avg_reward"]

    # Function to filter out list-type metrics and flatten the data
    def filter_metrics(metrics_list):
        filtered = []
        for item in metrics_list:
            if not isinstance(item, list):
                filtered.append(item)
        return filtered

    # Create environment matrices
    env_data = []
    for env in environments:
        env_matrix = []
        for algo in algorithms:
            # Get metrics and filter out list-type ones
            metrics = []
            for metric in metrics_name:
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
                tau, _ = kendalltau(env[:, i], env[:, j])
                taus.append(tau)
            payoff_avg[i, j] = np.mean(taus)

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

    alpha = 5.0  # 演化强度参数

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

    game_weights = dict(zip(metrics_name, weights_normalized))

    # # 输出主要结果
    # print(
    #     {
    #         "payoff_matrix": payoff_avg,
    #         "transition_matrix": T,
    #         "stationary_distribution": stat_dist.reshape((n, n)),
    #         "game_weights": game_weights
    #     }
    # )
    print(f'game_weights: {game_weights}')

    return game_weights


if __name__ == '__main__':
    json_file_path = '/workspace/omniisaacgymenvs/PursuitSim3D/results/eval/merged_results.json'

    # 读取文件并加载到变量 data 中
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    calculate_game_weights(data)