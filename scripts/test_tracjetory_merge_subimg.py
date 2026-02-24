# 将test_tracjetory.py生成的任意四个图拼成一个图

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# ================= 配置区域 (请修改这里) =================

# 1. 地图图片路径
MAP_IMAGE_PATH = "/workspace/omniisaacgymenvs/PursuitSim3D/utils/pathplan/map_simple.png"

# 2. 地图物理范围 (必须与训练环境一致)
MAP_BOUNDS = {
    'x_min': -10, 'x_max': 10,
    'y_min': -5,  'y_max': 5,
}

# 3. 四个数据文件的路径 (请填入你的 .npy 文件路径)
# 格式: ("子图标题", "文件路径")
PLOTS_CONFIG = [
    ("TD3",    "/path/to/your/TD3_trajectory.npy"),    # 左上
    ("SAC",    "/path/to/your/SAC_trajectory.npy"),    # 右上
    ("PPO",    "/path/to/your/PPO_trajectory.npy"),    # 左下
    ("CrossQ", "/path/to/your/CrossQ_trajectory.npy")  # 右下
]

# 4. 保存路径
SAVE_PATH = "results/imgs/Comparison_2x2_Plot.png"

# =======================================================

def draw_single_subplot(ax, trajectory_data, map_img, bounds, title):
    """
    在给定的 ax 上绘制单个算法的轨迹
    """
    # --- 1. 绘制地图背景 ---
    if map_img is not None:
        extent = [
            bounds['x_min'], bounds['x_max'], 
            bounds['y_min'], bounds['y_max']
        ]
        # 按照参考代码的配置：origin='upper', cmap='gray'
        ax.imshow(map_img, extent=extent, cmap='gray', origin='upper', alpha=1.0)
    else:
        ax.set_facecolor('white')

    # --- 2. 设置范围和比例 ---
    ax.set_xlim(bounds['x_min'], bounds['x_max'])
    ax.set_ylim(bounds['y_min'], bounds['y_max'])
    ax.set_aspect('equal')
    
    # 去掉坐标轴和黑框 (参考代码风格)
    ax.axis('off')
    # 如果想要保留黑框但去掉刻度，请注释上一行，使用下面两行：
    # ax.set_xticks([])
    # ax.set_yticks([])

    # 设置子图标题
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    # --- 3. 解析轨迹数据 ---
    # 假设数据形状: (Entities, Steps, 3)
    num_entities = trajectory_data.shape[0]
    num_agents = num_entities - 1
    
    # 使用参考代码的配色: Plasma
    colors = plt.cm.plasma(np.linspace(0, 0.9, num_agents))

    # --- 4. 绘制 Agents ---
    for i in range(num_agents):
        xs = trajectory_data[i, :, 0]
        ys = trajectory_data[i, :, 1]
        
        # 线条
        ax.plot(xs, ys, color=colors[i], linewidth=2.0, alpha=0.9, label=f'Agent {i}')
        # 起点 (实心圆)
        ax.scatter(xs[0], ys[0], color=colors[i], marker='o', s=50, 
                   edgecolors='white', linewidth=0.5, zorder=10)
        # 终点 (X)
        ax.scatter(xs[-1], ys[-1], color=colors[i], marker='x', s=60, 
                   linewidth=2, zorder=10)

    # --- 5. 绘制 Target ---
    target_idx = -1
    txs = trajectory_data[target_idx, :, 0]
    tys = trajectory_data[target_idx, :, 1]
    
    # 目标 (鲜红色虚线)
    ax.plot(txs, tys, color='red', linestyle='--', linewidth=2.5, alpha=0.9, label='Target')
    # 目标起点 (*)
    ax.scatter(txs[0], tys[0], color='red', marker='*', s=120, 
               edgecolors='black', zorder=10, label='Start')
    # 目标终点 (P)
    ax.scatter(txs[-1], tys[-1], color='darkred', marker='P', s=120, 
               edgecolors='white', zorder=10, label='End')


def main():
    # 1. 准备地图图片 (只读一次)
    map_img = None
    if os.path.exists(MAP_IMAGE_PATH):
        try:
            # 读取原始图片
            raw_img = mpimg.imread(MAP_IMAGE_PATH)
            # 如果是RGB图，为了配合 cmap='gray'，建议转单通道 (参考逻辑)
            if raw_img.ndim == 3:
                map_img = raw_img[:, :, 0] 
            else:
                map_img = raw_img
        except Exception as e:
            print(f"Error loading map: {e}")
    else:
        print(f"Warning: Map not found at {MAP_IMAGE_PATH}")

    # 2. 创建画布 (2行2列)
    # 物理比例是 2:1，所以figsize设为 (16, 9) 左右比较合适，留出空间给标题和图例
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=300)
    axes = axes.flatten() # 展平方便循环

    print("--- Generating 2x2 Trajectory Plot ---")

    # 3. 循环绘制每个子图
    for i, (algo_name, file_path) in enumerate(PLOTS_CONFIG):
        ax = axes[i]
        
        if os.path.exists(file_path):
            print(f"Plotting {algo_name} ...")
            try:
                # 加载 .npy 数据
                # 注意：如果原本存的是 object array，需要 allow_pickle=True
                data = np.load(file_path, allow_pickle=True)
                
                # 兼容性处理：如果读取出来是 object array (里面是list)，需要转标准 numpy
                if data.dtype == object:
                    # 尝试转换，假设每个轨迹长度一致可以直接 stack，如果不一致需特殊处理
                    # 这里假设存的时候已经是 (M, T, 3) 格式
                    pass 
                
                draw_single_subplot(ax, data, map_img, MAP_BOUNDS, algo_name)
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                ax.text(0.5, 0.5, f"Data Error: {algo_name}", ha='center')
        else:
            print(f"File not found: {file_path}")
            ax.text(0.5, 0.5, f"Missing File: {algo_name}", ha='center')
            ax.axis('off')

    # 4. 添加全局共享图例 (从第一个子图中提取句柄)
    # 这样比每个子图旁边都挂一个图例要整洁得多
    handles, labels = axes[0].get_legend_handles_labels()
    
    # 筛选图例项 (去重 & 排序)
    # 想要的顺序: Agent 0, Agent 1, Agent 2, Target, Start, End
    desired_order = ['Agent 0', 'Agent 1', 'Agent 2', 'Target', 'Start', 'End']
    unique_labels = dict(zip(labels, handles))
    
    final_handles = []
    final_labels = []
    
    for lbl in desired_order:
        if lbl in unique_labels:
            final_handles.append(unique_labels[lbl])
            final_labels.append(lbl)
    
    # 将图例放在底部
    fig.legend(final_handles, final_labels, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), # 居中底部
               ncol=len(final_labels),     # 横向排列
               frameon=False, 
               fontsize=12)

    # 5. 调整布局
    plt.tight_layout()
    # 底部留白给图例 (根据需要微调)
    plt.subplots_adjust(bottom=0.1) 
    
    # 6. 保存
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    plt.savefig(SAVE_PATH, bbox_inches='tight', dpi=300)
    print(f"Done! Combined plot saved to: {SAVE_PATH}")
    plt.close()

if __name__ == "__main__":
    main()