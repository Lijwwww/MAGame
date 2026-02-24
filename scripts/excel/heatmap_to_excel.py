import json
import pandas as pd
import os
import numpy as np
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

# ================= 配置区域 =================
ENV_TYPE = 'capture' # 可选 'search' 或 'capture'

# ================= Linux 字体配置 =================
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


# ================= Windows 字体配置 =================
# # 1. 字体设置
# # 逻辑：优先使用 'Times New Roman'，遇到它无法显示的字符（如汉字），
# # 就会自动去用列表里的第二个字体 'SimSun' (宋体)
# plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']

# # 2. 解决负号显示为方块的问题
# plt.rcParams['axes.unicode_minus'] = False

# # 3. (可选) 设置数学公式字体，使其与 Times 风格一致
# plt.rcParams['mathtext.fontset'] = 'stix'


if ENV_TYPE == 'search':
    JSON_PATH = 'results/eval/search/merged_results.json'
    OUTPUT_EXCEL = 'results/excel/search/final_kendall_heatmap.xlsx'
    OUTPUT_IMG = 'results/excel/search/final_kendall_heatmap.png'
    
    ALGORITHMS = ['TD3', 'PPO', 'DDPG']
    ENVIRONMENTS = ['default', 'obstacle0', 'obstacle1', 'friction0', 'friction1']
    
    # 指标名称汉化
    METRIC_CONFIG = {
        "成功率":     ("SR", 1),
        "平均奖励":   ("avg_reward", 1),
        "平均步数":   ("avg_step", -1),
        "平均碰撞次数":   ("avg_collision", -1)
    }

else: # Capture
    JSON_PATH = 'results/eval/capture/merged_results.json'
    OUTPUT_EXCEL = 'results/excel/capture/final_kendall_heatmap.xlsx'
    OUTPUT_IMG = 'results/excel/capture/final_kendall_heatmap.png'
    
    ALGORITHMS = ['TD3', 'SAC', 'PPO', 'DDPG', 'CrossQ', 'TRPO', 'TQC']
    
    ENVIRONMENTS = ['default', 'speed0', 'speed1', 'obstacle0', 'obstacle1', 
                    'poses0', 'poses1', 'friction0', 'friction1']
    
    # 指标名称汉化 (保持符号逻辑不变)
    METRIC_CONFIG = {
        "成功率":       ("SR", 1),              # 越大越好
        "平均奖励":     ("avg_reward", 1),      # 越大越好
        "平均步数":     ("avg_step", -1),       # 越小越好
        "平均碰撞次数": ("avg_collision", -1),  # 越小越好
        "平均逃逸次数": ("avg_evasion", -1)     # 越小越好 (与Payoff Matrix一致)
    }

# ================= 核心逻辑 (保持不变) =================

def calculate_exact_payoff_matrix(data):
    """
    计算完全一致的 Payoff Matrix (先局部 Tau 再平均)
    """
    metrics_list = list(METRIC_CONFIG.keys())
    n = len(metrics_list)
    
    sum_tau_matrix = np.zeros((n, n))
    env_count = 0

    print(f"正在计算全局一致性矩阵 (环境数量: {len(ENVIRONMENTS)})...")

    for env in ENVIRONMENTS:
        env_vectors = {}
        valid_env = True
        
        # 提取数据
        for m_display, (m_key, sign) in METRIC_CONFIG.items():
            vals = []
            for alg in ALGORITHMS:
                try:
                    # 兼容不同数据结构
                    if alg in data and env in data[alg]:
                        raw_val = data[alg][env].get(m_key, 0)
                        if isinstance(raw_val, list):
                            raw_val = np.mean(raw_val)
                        vals.append(raw_val * sign) # 应用符号修正
                    else:
                        vals.append(0)
                except Exception:
                    valid_env = False
                    break
            env_vectors[m_display] = vals
        
        if not valid_env:
            continue

        env_count += 1
        
        # 计算 Kendall Tau
        for i in range(n):
            for j in range(n):
                name_i = metrics_list[i]
                name_j = metrics_list[j]
                
                tau, _ = kendalltau(env_vectors[name_i], env_vectors[name_j])
                if np.isnan(tau): tau = 0 
                
                sum_tau_matrix[i, j] += tau

    # 取平均
    if env_count > 0:
        avg_tau_matrix = sum_tau_matrix / env_count
    else:
        avg_tau_matrix = sum_tau_matrix
    
    # 转 DataFrame
    df = pd.DataFrame(avg_tau_matrix, index=metrics_list, columns=metrics_list)
    return df

def plot_heatmap(df, save_path):
    """
    使用 matplotlib 绘制热力图 (中文版 + 大字号)
    """
    data = df.values
    labels = df.columns
    n = len(labels)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制热力图矩阵
    im = ax.imshow(data, cmap='RdBu_r', vmin=0, vmax=1)
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("排序一致性", rotation=-90, va="bottom", fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    # 设置刻度标签
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    
    # 调大坐标轴标签字号
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_yticklabels(labels, fontsize=18)

    # 调整 x 轴标签位置到顶部，并旋转
    # ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # 调整 x 轴标签位置到底部，并旋转
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True) # 底部
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")


    # 在每个格子里添加数值文本
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            # 根据背景色深浅自动调整字体颜色
            text_color = "white" if abs(val) > 0.5 else "black"
            
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, 
                    fontsize=18, fontweight='medium')

    # ax.set_title(f"多指标排序一致性热力图 ({ENV_TYPE})", y=1.1, fontsize=18)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[绘图完成] 图片已保存至: {save_path}")

def main():
    if not os.path.exists(JSON_PATH):
        print(f"错误: 找不到数据文件 {JSON_PATH}")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 计算矩阵
    df = calculate_exact_payoff_matrix(data)

    # 2. 导出 Excel (中文表头)
    os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)
    df.to_excel(OUTPUT_EXCEL, sheet_name="一致性矩阵")
    print(f"\n[数据保存] 中文Excel已保存: {OUTPUT_EXCEL}")

    # 3. 绘制热力图 (中文 + 大字号)
    plot_heatmap(df, OUTPUT_IMG)

    print("\n数据预览:")
    print(df.round(3))

if __name__ == '__main__':
    main()