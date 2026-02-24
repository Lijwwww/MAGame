import json
from matplotlib.colors import ListedColormap
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import os

def _distance(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos2-pos1)))

def total_distance(trajectory):
    total = 0
    trajectory = np.array(trajectory)
    for i in range(trajectory.shape[0]-1):
        total += _distance(trajectory[i], trajectory[i+1])
    return total

def mean_gap(trajectory1, trajectory2):
    trajectory1 = np.array(trajectory1)
    trajectory2 = np.array(trajectory2)
    total = 0
    num = trajectory1.shape[0]
    for i in range(num):
        total +=  _distance(trajectory1[i], trajectory2[i])
    return total / num

def collision_number(trajectory, dist_min=0.5): # A1, A2, A3
    trajectory = np.array(trajectory)
    total = 0
    for i in range(trajectory.shape[1]):
        is_collision = False
        for j in range(trajectory.shape[0]-1):
            for k in range(j+1, trajectory.shape[0]):
                if _distance(trajectory[j,i], trajectory[k,i]) < dist_min:
                    total += 1
                    is_collision = True
                    break
            if is_collision:
                break
    return total

def evasion_number(trajectory, dist_min=1): # A1, A2, A3, B1
    trajectory = np.array(trajectory)
    total = 0
    is_pursuit = False
    # print(trajectory)
    for i in range(trajectory.shape[1]):
        for j in range(trajectory.shape[0]-1):
            if _distance(trajectory[j, i], trajectory[-1, i]) < dist_min:
                is_pursuit = True
                break
        else:
            if is_pursuit:
                total += 1
                is_pursuit = False
    return total


def evaluation(env, model, env_name, checkpoint_name, save_dir, n_eval_episodes=1000):
    env._task.game_record = []
    evaluate_policy(model, env, n_eval_episodes)
    record = np.array(env._task.game_record)
    '''
    形状 (Shape): (N, 5)
    N: 评估的总回合数（例如 n_eval_episodes=1000）。
    5: 每一行包含 5 个特定的统计指标。

    每一列的含义:
    Col 0: Success (bool) - 是否成功抓捕（True/False）。
    Col 1: Steps (int) - 该回合消耗的步数（时间步长）。
    Col 2: Time (float) - 物理仿真时间或挂钟时间。
    Col 3: Trajectory (List of Lists) - 原始轨迹数据。
        形状 (Shape): (M, T, 3)
        M: 实体数量 (Num_Entities)。通常 = 智能体数量 (Agents) + 1 个目标 (Target)。
        T: 时间步数 (Num_Steps)。这个数字对应 record 中的第 1 列（Steps）。
        3: 坐标 (x, y, z)。
    Col 4: Rewards (List/Array) - 该回合每一步的奖励记录。
    '''
    total = record[:,:3].sum(0)
    num = record.shape[0]
    distance_list = []
    gap_list = []
    collision_list = []
    evasion_list = []
    reward_list = np.array([np.mean(r) for r in record[:, 4]])
    for i in range(num):
        dist = []
        gap = []
        for j in range(env._task._num_agents):
            dist.append(total_distance(record[i,3][j]))
            gap.append(mean_gap(record[i,3][j], record[i,3][env._task._num_agents]))
        distance_list.append(dist)
        gap_list.append(gap)
        collision_list.append(collision_number(record[i,3][:env._task._num_agents]))
        evasion_list.append(evasion_number(record[i,3]))
    
    # 1. 打印结果到命令行
    print("SR:{} avg_step:{} avg_time:{} avg_distance:{} avg_gap:{} avg_collision:{} avg_evasion:{} avg_reward:{}".format(
          total[0]/num, total[1]/num, total[2]/num,
          np.mean(distance_list, 0),
          np.mean(gap_list, 0),
          np.mean(collision_list, 0),
          np.mean(evasion_list, 0),
          np.mean(reward_list, 0)
      ))
    # with open(save_path, 'a') as f:
    #     f.write("{}:\nSR:{} avg_step:{} avg_time:{} avg_distance:{} avg_gap:{} avg_collision:{} avg_evasion:{} avg_reward:{}\n".format(
    #           env_name,
    #           total[0]/num, total[1]/num, total[2]/num,
    #           np.mean(distance_list, 0),
    #           np.mean(gap_list, 0),
    #           np.mean(collision_list, 0),
    #           np.mean(evasion_list, 0),
    #           np.mean(reward_list, 0)
    #       ))
    
    # 2. 存储json结果
    log_save_dir = os.path.join(save_dir, checkpoint_name)
    os.makedirs(log_save_dir, exist_ok=True)
    save_path = os.path.join(log_save_dir, f"log_{checkpoint_name}.json")
    metrics = {
        "SR": (total[0] / num),
        "avg_step": (total[1] / num),
        "avg_time": (total[2] / num),
        "avg_distance": np.mean(distance_list, 0).tolist(),  # 列表
        "avg_gap": np.mean(gap_list, 0).tolist(),            # 列表
        "avg_collision": np.mean(collision_list, 0).item(), 
        "avg_evasion": np.mean(evasion_list, 0).item(),     
        "avg_reward": np.mean(reward_list, 0).item()        
    }
    
    # 读取现有数据
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            data = json.load(f)
    else:
        # 文件不存在：创建一个新的空字典
        data = {}
    
    # 更新数据
    if checkpoint_name not in data:
        data[checkpoint_name] = {}
    
    # 将当前环境的数据放入字典
    data[checkpoint_name][env_name] = metrics
    
    # 覆盖写入
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
            
    # 3. 存储record原始数据    
    npy_filename = f"raw_data_{checkpoint_name}_{env_name}.npy"
    record_save_dir = os.path.join(save_dir, checkpoint_name, 'raw_records')
    os.makedirs(record_save_dir, exist_ok=True)
    record_save_path = os.path.join(record_save_dir, npy_filename)
    np.save(record_save_path, record)

    # --- 以后如何读取 ---
    # data = np.load(npy_path, allow_pickle=True)
    # trajectory = data[0, 3] # 读取第一个回合的轨迹
    


def plot_success_rate(success_rates, name, save_dir):
    plt.figure()
    # plt.plot(range(1, len(success_rates) + 1), success_rates, label="Success Rate")
    plt.plot(range(100, len(success_rates) + 100), success_rates, label="Success Rate")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Average Success Rate")
    plt.title("Success Rate Over Time")
    plt.legend()
    plt.grid()
    filename = f"success_rate_plot_{name}.png"
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path)
    plt.close()

def plot_avg_reward(cumulative_avg_reward, name, save_dir):
    plt.figure()
    # plt.plot(range(1, len(cumulative_avg_reward) + 1), cumulative_avg_reward, label="Average Reward")
    plt.plot(range(100, len(cumulative_avg_reward) + 100), cumulative_avg_reward, label="Average Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Average Reward")
    plt.title("Average Reward Over Time")
    plt.legend()
    plt.grid()
    filename = f"average_reward_plot_{name}.png"
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path)
    plt.close()
    


def evaluation_for_training(env, train_save_dir, env_name, checkpoint_name):
    record = np.array(env._task.game_record)
    total = record[:, :3].sum(0)
    num = record.shape[0]

    print(f'Episodes num: {num}')
    
    distance_list = []
    gap_list = []
    collision_list = []
    evasion_list = []
    reward_list = np.array([np.mean(r) for r in record[:, 4]])
    
    for i in range(num):
        dist = []
        gap = []
        for j in range(env._task._num_agents):
            dist.append(total_distance(record[i, 3][j]))
            gap.append(mean_gap(record[i, 3][j], record[i, 3][env._task._num_agents]))
        distance_list.append(dist)
        gap_list.append(gap)
        collision_list.append(collision_number(record[i, 3][:env._task._num_agents]))
        evasion_list.append(evasion_number(record[i, 3]))
    
    # # 累计平均
    # success_rates = np.cumsum(record[:, 0]) / np.arange(1, num + 1)
    # cumulative_avg_reward = np.cumsum(reward_list) / np.arange(1, num + 1)
    #
    # plot_success_rate(success_rates, name)
    # plot_avg_reward(cumulative_avg_reward, name)

    # 1. 绘图并存储
    window = 100 # 滑动窗口平均
    success_array = record[:, 0]
    success_rates = []
    avg_rewards = []

    for i in range(num):
        start_idx = max(0, i - window + 1)
        success_rates.append(np.mean(success_array[start_idx:i + 1]))
        avg_rewards.append(np.mean(reward_list[start_idx:i + 1]))

    success_rates = np.array(success_rates)
    avg_rewards = np.array(avg_rewards)

    name_tag = f"{checkpoint_name}_{env_name}" # 图片后缀
    
    plot_success_rate(success_rates[window - 1:], name_tag, train_save_dir)
    plot_avg_reward(avg_rewards[window - 1:], name_tag, train_save_dir)

    # 2. 控制台打印详细结果
    print("Train Eval Result -> SR:{} avg_step:{} avg_time:{} avg_distance:{} avg_gap:{} avg_collision:{} avg_evasion:{} avg_reward:{}".format(
          total[0]/num, total[1]/num, total[2]/num,
          np.mean(distance_list, 0),
          np.mean(gap_list, 0),
          np.mean(collision_list, 0),
          np.mean(evasion_list, 0),
          np.mean(reward_list, 0)
      ))

    # 3. 存储json结果
    log_save_path = os.path.join(train_save_dir, f"log_{checkpoint_name}.json")
    metrics = {
        "SR": (total[0] / num),
        "avg_step": (total[1] / num),
        "avg_time": (total[2] / num),
        "avg_distance": np.mean(distance_list, 0).tolist(),  # 列表
        "avg_gap": np.mean(gap_list, 0).tolist(),            # 列表
        "avg_collision": np.mean(collision_list, 0).item(), 
        "avg_evasion": np.mean(evasion_list, 0).item(),     
        "avg_reward": np.mean(reward_list, 0).item()        
    }
    
    # 读取现有数据
    if os.path.exists(log_save_path):
        with open(log_save_path, 'r') as f:
            data = json.load(f)
    else:
        # 文件不存在：创建一个新的空字典
        data = {}
    
    # 更新数据
    if checkpoint_name not in data:
        data[checkpoint_name] = {}
    
    # 将当前环境的数据放入字典
    data[checkpoint_name][env_name] = metrics
    
    # 覆盖写入
    with open(log_save_path, 'w') as f:
        json.dump(data, f, indent=4)
            
    # 3. 存储record原始数据    
    npy_filename = f"raw_data_{checkpoint_name}_{env_name}.npy"
    record_save_path = os.path.join(train_save_dir, npy_filename)
    np.save(record_save_path, record)
    

# from models import NPG
# from models import TD3_tianshou
# def evaluation_tianshou(env):
#     NPG.evaluate(env, "checkpoints/NPGv2.pth")
#     record = np.array(env._task.game_record)
#     total = record[:,:3].sum(0)
#     num = record.shape[0]
#     distance_list = []
#     gap_list = []
#     collision_list = []
#     evasion_list = []
#     reward_list = np.array([np.mean(r) for r in record[:, 4]])
#     for i in range(num):
#         dist = []
#         gap = []
#         for j in range(env._task._num_agents):
#             dist.append(total_distance(record[i,3][j]))
#             gap.append(mean_gap(record[i,3][j], record[i,3][env._task._num_agents]))
#         distance_list.append(dist)
#         gap_list.append(gap)
#         collision_list.append(collision_number(record[i,3][:env._task._num_agents]))
#         evasion_list.append(evasion_number(record[i,3]))

#     print("SR:{} avg_step:{} avg_time:{} avg_distance:{} avg_gap:{} avg_collision:{} avg_evasion:{} avg_reward:{}".format(
#           total[0]/num, total[1]/num, total[2]/num,
#           np.mean(distance_list, 0),
#           np.mean(gap_list, 0),
#           np.mean(collision_list, 0),
#           np.mean(evasion_list, 0),
#           np.mean(reward_list, 0)
#       ))
#     with open('/workspace/omniisaacgymenvs/PursuitSim3D/results/log_NPG.txt', 'a') as f:
#         f.write("v2:\nSR:{} avg_step:{} avg_time:{} avg_distance:{} avg_gap:{} avg_collision:{} avg_evasion:{} avg_reward:{}\n".format(
#               total[0]/num, total[1]/num, total[2]/num,
#               np.mean(distance_list, 0),
#               np.mean(gap_list, 0),
#               np.mean(collision_list, 0),
#               np.mean(evasion_list, 0),
#               np.mean(reward_list, 0)
#           ))


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


def plot_trajectory_2d_gray_obsatcle(trajectory_data, map_bounds, map_image_path, save_path, title):
    """
    绘制单次轨迹的 2D 投影图 (带地图背景)
    """
    # 1. 设置画布比例
    # 地图物理尺寸: 宽=20 (10 - -10), 高=10 (5 - -5). 比例 2:1
    # 我们设置 figsize 为 (12, 6) 保持这个比例，分辨率更高
    fig, ax = plt.subplots(figsize=(12, 6)) 
    
    # 2. 绘制地图背景 (如果图片存在)
    if map_image_path and os.path.exists(map_image_path):
        # 读取图片
        raw_img = mpimg.imread(map_image_path)
        
        if raw_img.ndim == 3:
            img_2d = raw_img[:, :, 0] 
        else:
            img_2d = raw_img

        # --- [修复核心] 强制二值化为 0 和 1 ---
        # 有时候读取的图片可能是 0~255 的整数，也可能是 0.0~1.0 的浮点数
        # 我们设定一个阈值（比如 0.5），强行把像素分为两类：
        # 0 (障碍物/黑), 1 (地面/白)
        # 这样就能完美对应 ListedColormap 里的两个颜色
        threshold = 0.5
        if img_2d.max() > 1.0: # 如果是 0-255 的格式
            threshold = 127
            
        # 生成最终的二值索引图 (int 类型)
        # 这里的 0 对应 cmap 的第一个颜色 (#3B3B3B)
        # 这里的 1 对应 cmap 的第二个颜色 (#FFFFFF)
        img = (img_2d > threshold).astype(int)

        # extent 参数极其重要：它将图片的像素范围映射到物理坐标系
        # [x_min, x_max, y_min, y_max]
        extent = [
            map_bounds['x_min'], map_bounds['x_max'], 
            map_bounds['y_min'], map_bounds['y_max']
        ]
        
        # 原本是 cmap='gray' (0是黑, 1是白)
        # 现在我们要: 0(障碍物) -> 深灰(#3B3B3B), 1(地面) -> 纯白(#FFFFFF)
        custom_cmap = ListedColormap(['#3B3B3B', '#FFFFFF'])
        # 绘制图片
        # origin='upper' 是图片的默认模式 (0,0在左上角)，对应物理坐标的 y_max
        # cmap='gray' 确保二值图正确显示黑白
        ax.imshow(img, extent=extent, cmap=custom_cmap, origin='upper', alpha=1.0, zorder=0)
    else:
        print(f"Warning: Map image not found at {map_image_path}, using blank background.")
        ax.set_facecolor('white') # 如果没图，背景设白

    # 3. 设置范围和比例
    ax.set_xlim(map_bounds['x_min'], map_bounds['x_max'])
    ax.set_ylim(map_bounds['y_min'], map_bounds['y_max'])
    ax.set_aspect('equal') # 锁定比例，确保车不会画扁

    # 增加淡网格 (提升科学感) 
    # zorder=1 保证网格在地图上面，但在轨迹下面
    ax.grid(True, which='major', color='#CCCCCC', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)

    # 4. 去掉坐标轴 (只保留图像内容)
    # ax.axis('off') 
    # 如果你想保留边缘刻度但去掉黑框，可以用下面这几行代替 ax.axis('off'):
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # 或者不要完全去掉 axis('off')，而是保留一个干净的边框
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333') # 边框也用深灰，不要纯黑
        spine.set_linewidth(1.5)       # 边框稍微加粗一点点，更有定界感
    
    # 隐藏刻度数字（如果只想要纯轨迹展示）
    ax.set_xticks([])
    ax.set_yticks([])

    # 5. 解析轨迹数据
    num_entities = trajectory_data.shape[0]
    num_agents = num_entities - 1
    
    # 使用高对比度的颜色 (因为地图可能有黑有白，鲜艳的颜色比较显眼)
    # viridis, plasma, jet 等 colormap 比较亮
    # colors = plt.cm.plasma(np.linspace(0, 0.9, num_agents))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # --- 绘制 Agents ---
    for i in range(num_agents):
        xs = trajectory_data[i, :, 0]
        ys = trajectory_data[i, :, 1]
        
        # 线条稍微加粗一点
        # ax.plot(xs, ys, color=colors[i], linewidth=2.0, alpha=0.9, label=f'Agent {i}')
        # # 起点 (实心圆)
        # ax.scatter(xs[0], ys[0], color=colors[i], marker='o', s=50, edgecolors='white', linewidth=0.5, zorder=10)
        # # 终点 (X)
        # ax.scatter(xs[-1], ys[-1], color=colors[i], marker='x', s=60, linewidth=2, zorder=10)
        ax.plot(xs, ys, color=colors[i], linewidth=2.5, alpha=0.9, label=f'Agent {i}', zorder=10)
        ax.scatter(xs[0], ys[0], color=colors[i], marker='o', s=60, edgecolors='white', linewidth=0.8, zorder=11)
        ax.scatter(xs[-1], ys[-1], color=colors[i], marker='x', s=70, linewidth=2, zorder=11)

    # --- 绘制 Target ---
    target_idx = -1
    txs = trajectory_data[target_idx, :, 0]
    tys = trajectory_data[target_idx, :, 1]
    
    # 目标用 鲜红色 虚线
    # ax.plot(txs, tys, color='red', linestyle='--', linewidth=2.5, alpha=0.9, label='Target')
    # ax.scatter(txs[0], tys[0], color='red', marker='*', s=120, edgecolors='black', zorder=10, label='Start')
    # ax.scatter(txs[-1], tys[-1], color='darkred', marker='P', s=120, edgecolors='white', zorder=10, label='End')
    ax.plot(txs, tys, color='#D62728', linestyle='--', linewidth=2.5, alpha=1.0, label='Target', zorder=10)
    ax.scatter(txs[0], tys[0], color='#D62728', marker='*', s=150, edgecolors='white', zorder=11, label='Start')
    ax.scatter(txs[-1], tys[-1], color='#8C1B1B', marker='P', s=150, edgecolors='white', zorder=11, label='End')

    # # 标题放下面或者上面，根据你的喜好
    # ax.set_title(title, fontsize=10, pad=10)
    
    # # 图例放到图外面，不遮挡地图
    # plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., frameon=False)
    
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=200, bbox_inches='tight') # bbox_inches='tight' 去除白边
    # plt.close()
    # print(f"  -> Plot saved: {os.path.basename(save_path)}")
    # 标题优化
    # 使用 serif 字体更像论文，或者保持默认 sans-serif
    # ax.set_title(title, fontsize=12, pad=12, color='#333333', fontweight='bold')
    
    # 图例优化
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
               borderaxespad=0., frameon=False, fontsize=10) # frameon=False 去掉图例边框，更简洁
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight') # 论文建议 dpi 300
    plt.close()
    print(f"  -> Plot saved: {os.path.basename(save_path)}")

def plot_trajectory_2d2(trajectory_data, map_bounds, map_image_path, save_path, title):
    """
    绘制单次轨迹的 2D 投影图 (带地图背景)
    """
    # 1. 设置画布比例
    # 地图物理尺寸: 宽=20 (10 - -10), 高=10 (5 - -5). 比例 2:1
    # 我们设置 figsize 为 (12, 6) 保持这个比例，分辨率更高
    fig, ax = plt.subplots(figsize=(12, 6)) 
    
    # 2. 绘制地图背景 (如果图片存在)
    if map_image_path and os.path.exists(map_image_path):
        # 读取图片
        img = mpimg.imread(map_image_path)
        
        # extent 参数极其重要：它将图片的像素范围映射到物理坐标系
        # [x_min, x_max, y_min, y_max]
        extent = [
            map_bounds['x_min'], map_bounds['x_max'], 
            map_bounds['y_min'], map_bounds['y_max']
        ]
        
        # 绘制图片
        # origin='upper' 是图片的默认模式 (0,0在左上角)，对应物理坐标的 y_max
        # cmap='gray' 确保二值图正确显示黑白
        ax.imshow(img, extent=extent, cmap='gray', origin='upper', alpha=1.0)
    else:
        print(f"Warning: Map image not found at {map_image_path}, using blank background.")
        ax.set_facecolor('white') # 如果没图，背景设白

    # 3. 设置范围和比例
    ax.set_xlim(map_bounds['x_min'], map_bounds['x_max'])
    ax.set_ylim(map_bounds['y_min'], map_bounds['y_max'])
    ax.set_aspect('equal') # 锁定比例，确保车不会画扁

    # 4. 去掉坐标轴 (只保留图像内容)
    ax.axis('off') 
    # 如果你想保留边缘刻度但去掉黑框，可以用下面这几行代替 ax.axis('off'):
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.set_xticks([])
    # ax.set_yticks([])

    # 5. 解析轨迹数据
    num_entities = trajectory_data.shape[0]
    num_agents = num_entities - 1
    
    # 使用高对比度的颜色 (因为地图可能有黑有白，鲜艳的颜色比较显眼)
    # viridis, plasma, jet 等 colormap 比较亮
    colors = plt.cm.plasma(np.linspace(0, 0.9, num_agents))

    # --- 绘制 Agents ---
    for i in range(num_agents):
        xs = trajectory_data[i, :, 0]
        ys = trajectory_data[i, :, 1]
        
        # 线条稍微加粗一点
        ax.plot(xs, ys, color=colors[i], linewidth=2.0, alpha=0.9, label=f'Agent {i}')
        # 起点 (实心圆)
        ax.scatter(xs[0], ys[0], color=colors[i], marker='o', s=50, edgecolors='white', linewidth=0.5, zorder=10)
        # 终点 (X)
        ax.scatter(xs[-1], ys[-1], color=colors[i], marker='x', s=60, linewidth=2, zorder=10)

    # --- 绘制 Target ---
    target_idx = -1
    txs = trajectory_data[target_idx, :, 0]
    tys = trajectory_data[target_idx, :, 1]
    
    # 目标用 鲜红色 虚线
    ax.plot(txs, tys, color='red', linestyle='--', linewidth=2.5, alpha=0.9, label='Target')
    ax.scatter(txs[0], tys[0], color='red', marker='*', s=120, edgecolors='black', zorder=10, label='Start')
    ax.scatter(txs[-1], tys[-1], color='darkred', marker='P', s=120, edgecolors='white', zorder=10, label='End')

    # 标题放下面或者上面，根据你的喜好
    # ax.set_title(title, fontsize=10, pad=10)
    
    # 图例放到图外面，不遮挡地图
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., frameon=False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight') # bbox_inches='tight' 去除白边
    plt.close()
    print(f"  -> Plot saved: {os.path.basename(save_path)}")



import scipy.ndimage  # 需要引入 scipy 进行图像形态学操作

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
        processed_img = scipy.ndimage.binary_dilation(binary_img, structure=np.ones((3,3)), iterations=3).astype(int)

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


from matplotlib.lines import Line2D
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


def evaluation_with_plotting(env, model, env_name, checkpoint_name, save_dir, n_eval_episodes=1000):
    """
    评估并绘制典型轨迹
    """
    # 运行评估
    env._task.game_record = []
    evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    
    # 获取记录
    # record 结构: [ [success, steps, time, trajectory, rewards], ... ]
    record = np.array(env._task.game_record, dtype=object) # object 类型防止 jagged array 报错
    num_episodes = record.shape[0]
    
    print(f"Evaluation finished. Collected {num_episodes} episodes.")

    # 提取列数据
    success_col = record[:, 0].astype(bool)  # 第0列: 是否成功
    steps_col = record[:, 1].astype(int)     # 第1列: 步数
    trajectory_col = record[:, 3]            # 第3列: 轨迹数据 (List of Lists)

    # --- 1. 准备地图参数 ---
    # 根据你的环境 env_name 设置地图边界
    map_img_path = "/workspace/omniisaacgymenvs/PursuitSim3D/utils/pathplan/map_simple.png"
    map_bounds = {
        'x_min': -10, 'x_max': 10,
        'y_min': -5, 'y_max': 5,
    }

    traj_dir = os.path.join(save_dir, "trajectory_imgs")
    os.makedirs(traj_dir, exist_ok=True)

    # --- 2. 筛选并绘图逻辑 ---
    # === A. 最快成功围捕 (Fastest Success) ===
    success_indices = np.where(success_col)[0]
    if len(success_indices) > 0:
        # 在成功的里面找步数最少的
        min_steps_idx = np.argmin(steps_col[success_indices])
        global_idx = success_indices[min_steps_idx]
        steps = steps_col[global_idx]
        
        # 提取轨迹并转为 numpy
        # 原始结构: list(agents) -> list(steps) -> list(xyz)
        # 转换后 shape: (Num_Agents, Num_Steps, 3)
        base_filename = f"Fastest_Success_Step{steps}"

        traj_data = np.array(trajectory_col[global_idx])
        
        plot_trajectory_2d(
            traj_data, map_bounds, map_img_path,
            os.path.join(traj_dir, f"{base_filename}.png"),
            f"Fastest Success - {checkpoint_name}_{env_name}"
        )
        
        data_save_path = os.path.join(traj_dir, f"{base_filename}.npy")
        np.save(data_save_path, traj_data)
        print(f"  -> Trajectory data saved: {base_filename}.npy")
    else:
        print("No successful episodes found.")

    # === B. 最多碰撞 (Most Collisions) ===
    # 初始化变量，防止报错
    max_coll_idx = None 
    max_coll_count = 0
    num_agents_for_collision = env._task._num_agents # 从 task 获取智能体数量
    
    all_collisions = []
    
    # 只有当有回合数据时才计算
    if num_episodes > 0:
        for i in range(num_episodes):
            # 取出前 num_agents 个实体的轨迹
            # trajectory_col[i] 结构是 [A1, A2, ..., Target]
            current_traj = np.array(trajectory_col[i])
            # 安全切片：确保不会越界
            if current_traj.shape[0] >= num_agents_for_collision:
                agents_traj = current_traj[:num_agents_for_collision] 
                n_coll = collision_number(agents_traj)
                all_collisions.append(n_coll)
            else:
                all_collisions.append(0)
    
        all_collisions = np.array(all_collisions)
        
        # 找到最大碰撞次数的索引 (防止空数组报错)
        if len(all_collisions) > 0:
            max_coll_idx = np.argmax(all_collisions)
            max_coll_count = all_collisions[max_coll_idx]
    
    # 绘图逻辑：必须同时满足碰撞数 > 0 且 索引已找到
    if max_coll_count > 0 and max_coll_idx is not None:
        traj_data = np.array(trajectory_col[max_coll_idx])
        base_filename = f"Max_Collision_Count{max_coll_count}"
        
        plot_trajectory_2d(
            traj_data, map_bounds, map_img_path,
            os.path.join(traj_dir, f"{base_filename}.png"),
            f"Max Collisions - {checkpoint_name}_{env_name}"
        )

        data_save_path = os.path.join(traj_dir, f"{base_filename}.npy")
        np.save(data_save_path, traj_data)
        print(f"  -> Trajectory data saved: {base_filename}.npy")
    else:
        print("No collisions found in any episode (or 0 collisions).")

    # === C. 中位数表现 (Median Performance) ===
    # 在成功的案例中找步数中位数，代表“一般水平”
    if len(success_indices) > 5:
        sorted_indices_by_step = success_indices[np.argsort(steps_col[success_indices])]
        median_local_idx = len(sorted_indices_by_step) // 2
        global_median_idx = sorted_indices_by_step[median_local_idx]
        median_steps = steps_col[global_median_idx]
        
        traj_data = np.array(trajectory_col[global_median_idx])
        base_filename = f"Median_Performance_Step{median_steps}"
        plot_trajectory_2d(
            traj_data, map_bounds, map_img_path,
            os.path.join(traj_dir, f"{base_filename}.png"),
            f"Median Performance\n{checkpoint_name} ({env_name})\nSteps: {median_steps}"
        )

        data_save_path = os.path.join(traj_dir, f"{base_filename}.npy")
        np.save(data_save_path, traj_data)
        print(f"  -> Trajectory data saved: {base_filename}.npy")

    # === D. 随机采样 (Random Examples) ===
    # 随机抽取样本（如果总数不够则全取）
    n_random_samples = 8
    if num_episodes > 0:
        #防止采样数大于总数
        sample_size = min(n_random_samples, num_episodes)
        # replace=False 表示不重复采样
        random_indices = np.random.choice(num_episodes, size=sample_size, replace=False)
        
        for idx in random_indices:
            # 获取该回合信息
            steps = steps_col[idx]
            is_success = success_col[idx]
            status_str = "Success" if is_success else "Fail"
            
            traj_data = np.array(trajectory_col[idx])
            
            # 定义文件名 (包含 状态、索引、步数)
            base_filename = f"Random_{status_str}_Idx{idx}_Step{steps}"
            
            # 绘图
            plot_trajectory_2d(
                traj_data, map_bounds, map_img_path,
                os.path.join(traj_dir, f"{base_filename}.png"),
                f"Random Sample ({status_str})\n{checkpoint_name} - Ep {idx}"
            )
            
            # 保存数据
            data_save_path = os.path.join(traj_dir, f"{base_filename}.npy")
            np.save(data_save_path, traj_data)
            print(f"  -> Trajectory data saved: {base_filename}.npy")

    # === E. 导出独立图例 ===
    # 保存一个通用的图例文件，方便后期拼图
    is_vertical = False
    legend_base_name = 'vertical' if is_vertical else 'horizontal'
    legend_save_path = os.path.join(save_dir, f"Shared_Legend_{legend_base_name}.png")
    
    # 从 task 获取智能体数量
    num_agents = traj_data.shape[0] - 1
    
    # 调用函数 (默认横排，适合放在拼图底部)
    plot_legend_only(num_agents, legend_save_path, vertical=False)

    # --- 3. 记录日志 ---
    record = np.array(env._task.game_record)
    total = record[:,:3].sum(0)
    num = record.shape[0]
    distance_list = []
    gap_list = []
    collision_list = []
    evasion_list = []
    reward_list = np.array([np.mean(r) for r in record[:, 4]])
    for i in range(num):
        dist = []
        gap = []
        for j in range(env._task._num_agents):
            dist.append(total_distance(record[i,3][j]))
            gap.append(mean_gap(record[i,3][j], record[i,3][env._task._num_agents]))
        distance_list.append(dist)
        gap_list.append(gap)
        collision_list.append(collision_number(record[i,3][:env._task._num_agents]))
        evasion_list.append(evasion_number(record[i,3]))

    log_save_path = os.path.join(save_dir, f"log_{checkpoint_name}_{env_name}.txt")
    print("SR:{} avg_step:{} avg_time:{} avg_distance:{} avg_gap:{} avg_collision:{} avg_evasion:{} avg_reward:{}".format(
          total[0]/num, total[1]/num, total[2]/num,
          np.mean(distance_list, 0),
          np.mean(gap_list, 0),
          np.mean(collision_list, 0),
          np.mean(evasion_list, 0),
          np.mean(reward_list, 0)
      ))
    with open(log_save_path, 'a') as f:
        f.write("{}:\nSR:{} avg_step:{} avg_time:{} avg_distance:{} avg_gap:{} avg_collision:{} avg_evasion:{} avg_reward:{}\n".format(
              env_name,
              total[0]/num, total[1]/num, total[2]/num,
              np.mean(distance_list, 0),
              np.mean(gap_list, 0),
              np.mean(collision_list, 0),
              np.mean(evasion_list, 0),
              np.mean(reward_list, 0)
          ))
    
    # 直接保存 record 数组 (allow_pickle=True 是必须的，因为数据包含列表对象)
    npy_filename = f"raw_data_{checkpoint_name}_{env_name}.npy"
    npy_path = os.path.join(save_dir, npy_filename)
    np.save(npy_path, record)

    # --- 以后如何读取 ---
    # data = np.load(npy_path, allow_pickle=True)
    # trajectory = data[0, 3] # 读取第一个回合的轨迹