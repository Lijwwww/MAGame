import scipy.ndimage  # 需要引入 scipy 进行图像形态学操作
import json
from matplotlib.colors import ListedColormap
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import os
from matplotlib.lines import Line2D

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


def plot_trajectory_2d(trajectory_data, map_bounds, map_image_path, save_path, title):
    """
    绘制 2D 轨迹图：黑白地图(障碍物腐蚀处理) + 统一的起终点标记
    """
    # 1. 设置画布 (保持 2:1 比例)
    fig, ax = plt.subplots(figsize=(12, 6))

    # 2. 处理并绘制地图
    if map_image_path and os.path.exists(map_image_path):
        # 读取图片并转为二值 (0:障碍物/黑, 1:地面/白)
        raw_img = mpimg.imread(map_image_path)
        if raw_img.ndim == 3:
            raw_img = raw_img[:, :, 0]
        
        # 二值化阈值处理
        threshold = 0.5 if raw_img.max() <= 1.0 else 127
        binary_img = (raw_img > threshold).astype(int)

        # --- 核心修改：腐蚀黑色障碍物 ---
        # 逻辑：让黑色区域(0)变小 = 让白色区域(1)变大 => 使用膨胀(dilation)操作
        # iterations=2 控制腐蚀程度，数字越大障碍物越瘦
        processed_img = scipy.ndimage.binary_dilation(binary_img, structure=np.ones((3,3)), iterations=1).astype(int)
        # processed_img = scipy.ndimage.binary_dilation(binary_img, structure=np.ones((3,3)), iterations=2).astype(int)
        # processed_img = binary_img

        # 映射坐标范围
        extent = [map_bounds['x_min'], map_bounds['x_max'], map_bounds['y_min'], map_bounds['y_max']]
        
        # 绘制地图 (cmap='gray': 0黑, 1白)
        ax.imshow(processed_img, extent=extent, cmap='gray', origin='upper', alpha=1.0, zorder=0)
    else:
        print(f"Warning: Map not found at {map_image_path}")
        ax.set_facecolor('white')

    # 3. 设置坐标轴与网格
    ax.set_xlim(map_bounds['x_min'], map_bounds['x_max'])
    ax.set_ylim(map_bounds['y_min'], map_bounds['y_max'])
    ax.set_aspect('equal')
    
    # 仅保留图像内容，去除坐标轴刻度但保留边框
    ax.set_xticks([])
    ax.set_yticks([])
    # 增加淡网格提升可读性
    ax.grid(True, which='major', color='#CCCCCC', linestyle='--', linewidth=0.5, alpha=0.3, zorder=1)

    # 4. 解析数据
    num_entities = trajectory_data.shape[0]
    num_agents = num_entities - 1
    
    # 配色方案
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # --- 绘制围捕者 (Agents) ---
    for i in range(num_agents):
        xs = trajectory_data[i, :, 0]
        ys = trajectory_data[i, :, 1]
        
        c = colors[i % 10]
        ax.plot(xs, ys, color=c, linewidth=2.5, alpha=0.9, label=f'智能体 {i}', zorder=10)
        ax.scatter(xs[0], ys[0], color=c, marker='o', s=60, edgecolors='white', linewidth=0.8, zorder=11)
        ax.scatter(xs[-1], ys[-1], color=c, marker='x', s=70, linewidth=2, zorder=11)

    # --- 绘制逃逸者 (Target) ---
    target_idx = -1
    txs = trajectory_data[target_idx, :, 0]
    tys = trajectory_data[target_idx, :, 1]
    
    t_color = '#D62728' # 鲜红色
    ax.plot(txs, tys, color=t_color, linestyle='--', linewidth=2.5, alpha=1.0, label='目标', zorder=10)
    # 起点 (圆)
    ax.scatter(txs[0], tys[0], color=t_color, marker='o', s=80, edgecolors='white', linewidth=1.0, zorder=11, label='起点')
    # 终点 (叉)
    ax.scatter(txs[-1], tys[-1], color=t_color, marker='x', s=90, linewidth=2.5, zorder=11, label='终点')

    # 5. 图例与保存
    # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., frameon=False, fontsize=10) # 画在图外
    # plt.legend(  # 画在图里
    #     loc='upper right',       # 图例的右上角对齐
    #     bbox_to_anchor=(1, 1),   # 对齐到图表的(1,1)位置（即图表右上角）
    #     borderaxespad=0.3,        # 距离坐标轴的内边距（防止贴太紧，可设为0-1之间）
    #     frameon=False,           # 去掉边框（透明背景）
    #     fontsize=10
    # )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Plot saved: {os.path.basename(save_path)}")


def plot_legend_only(num_agents, save_path, vertical=False):
    """
    导出单独的图例图片
    :param num_agents: 智能体数量 (用于生成正确的颜色)
    :param save_path: 保存路径
    :param vertical: True为竖排图例，False为横排图例
    """
    # 1. 生成与主图一致的颜色
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # 2. 创建自定义图例句柄 (Proxy Artists)
    legend_elements = []
    
    # --- 添加 Agents ---
    # for i in range(num_agents):
    #     # 创建线段句柄
    #     line = Line2D([0], [0], color=colors[i], lw=2, label=f'Agent {i}')
    #     legend_elements.append(line)
    # 智能体共只画一个图例的版本
    line = Line2D([0], [0], color=colors[0], lw=2, label=f'智能体')
    legend_elements.append(line)
        
    # --- 添加 Target ---
    target_line = Line2D([0], [0], color='red', linestyle='--', lw=2.5, label='目标')
    legend_elements.append(target_line)
    
    # (可选) 如果你想解释起点和终点，可以解开下面的注释
    legend_elements.append(Line2D([0], [0], color='#D62728', marker='o', linestyle='None', label='起点', markeredgewidth=0.8, markersize=8))
    legend_elements.append(Line2D([0], [0], color='#D62728', marker='x', linestyle='None', label='终点', markeredgewidth=2, markersize=8.5))

    # 3. 创建一个空画布
    fig_size = (2, 3) if vertical else (8, 0.5)
    fig, ax = plt.subplots(figsize=fig_size)
    
    # 4. 移除坐标轴和背景
    ax.axis('off')
    
    # 5. 生成图例
    if vertical:
        # 竖排
        ax.legend(handles=legend_elements, loc='center', frameon=False, fontsize=12)
    else:
        # 横排 (ncol=列数，根据元素数量自动调整)
        ax.legend(handles=legend_elements, loc='center', ncol=len(legend_elements), frameon=False, fontsize=12)
    
    # 6. 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  -> Shared legend saved: {os.path.basename(save_path)}")


def evaluation_with_plotting():
    """
    绘制轨迹
    """
    # --- 1. 准备地图参数 ---
    # 根据你的环境 env_name 设置地图边界
    # map_img_path = "/workspace/omniisaacgymenvs/PursuitSim3D/utils/pathplan/map_simple.png"
    map_img_path = "/workspace/omniisaacgymenvs/PursuitSim3D/utils/pathplan/map_difficult.png"
    map_bounds = {
        'x_min': -10, 'x_max': 10,
        'y_min': -5, 'y_max': 5,
    }

    save_dir = "results/test_tracjetory/trajectory_imgs"
    os.makedirs(save_dir, exist_ok=True)

    # data_save_path = 'results/test_tracjetory/search/DDPG_default/trajectory_imgs/Random_Success_Idx8_Step24.npy'
    data_save_path = 'results/test_tracjetory/search/SAC_default/trajectory_imgs/Random_Success_Idx17_Step24.npy'
    # data_save_path = 'results/test_tracjetory/capture/PPO_default/trajectory_imgs/Random_Success_Idx0_Step44.npy'
    # data_save_path = 'results/test_tracjetory/capture/TD3_default/trajectory_imgs/Median_Performance_Step20.npy'

    traj_data = np.load(data_save_path)

    file_name = os.path.splitext(os.path.basename(data_save_path))[0]
    parent_dir = data_save_path.split(os.sep)[-3]
    base_filename = parent_dir + '_' + file_name

    plot_trajectory_2d(
        traj_data, map_bounds, map_img_path,
        os.path.join(save_dir, f"{base_filename}.png"),
        f"Fastest Success - {parent_dir}"
    )
        
    # 保存一个通用的图例文件，方便后期拼图
    is_vertical = False
    legend_base_name = 'vertical' if is_vertical else 'horizontal'
    legend_save_path = os.path.join(save_dir, f"Shared_Legend_{legend_base_name}.png")
    
    # 从 task 获取智能体数量
    num_agents = traj_data.shape[0] - 1
    
    # 调用函数 (默认横排，适合放在拼图底部)
    plot_legend_only(num_agents, legend_save_path, vertical=is_vertical)


if __name__ == '__main__':
    evaluation_with_plotting()