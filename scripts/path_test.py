'''
固定初始点和终点，保留一个智能体，画其运行轨迹，证明规划算法是有效的
启动方式
1. self.pathtest 设为 True
2. 设为固定初始位置并保持与本代码的一致
'''

import os
import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from utils.hydra_cfg.reformat import omegaconf_to_dict
from envs.vec_env_rlgames import VecEnvRLGames
from utils.task_util import initialize_task
from utils.pathplan.pathplan_utils import *
from utils.hydra_cfg.hydra_utils import *
from scipy.spatial.distance import cdist
from matplotlib import font_manager


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    headless = cfg.headless
    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id)

    cfg_dict = omegaconf_to_dict(cfg)
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed

    task = initialize_task(cfg_dict, env)

    from omni.isaac.core.simulation_context import SimulationContext
    simulation_context = SimulationContext()
    
    start_point = [-7.0, -3.5, 1.0]
    # goal_point = [0.0, 0.0, 0.0]
    goal_point = [8.8, 1.5, 0.0]
    
    # 推导理想轨迹（方式一所需）
    ideal_trajectory = []
    start_index = parallel_pos_to_index(torch.tensor(start_point[:2]), task.occupancy_mapsize)
    goal_index = parallel_pos_to_index(torch.tensor(goal_point[:2]), task.occupancy_mapsize)
    
    current_index = start_index
    while current_index != goal_index:
        current_coordinate = index_to_coordinate(current_index, task.occupancy_mapsize[1])
        ideal_trajectory.append(parallel_coordinate_to_pos(torch.tensor(current_coordinate), task.occupancy_mapsize))
        
        neighbors = find_neighbor(current_coordinate, task.occupancy_map)
        next_index = current_index
        min_dist = task.dist_to_goal_map[current_index][goal_index]
        
        for neighbor in neighbors:
            neighbor_index = coordinate_to_index(neighbor, task.occupancy_mapsize[1])
            if task.dist_to_goal_map[neighbor_index][goal_index] < min_dist:
                next_index = neighbor_index
                min_dist = task.dist_to_goal_map[neighbor_index][goal_index]
        
        if next_index == current_index:
            break  # 避免死循环
        current_index = next_index
    
    # 记录实际轨迹
    actual_trajectory = []
    # task.target_point[0, 0, :] = torch.tensor(goal_point, device=task._device)
    task.pathtest_target_point = goal_point
    task.robogame.poses['red0'][0] = start_point
    task.initial_pos_agents[0][0] = torch.tensor(start_point)
    
    for step in range(5):
        # 运行方法一
        # task.path_plan()
        task.path_plan2()
        task.pre_physics_step(None)  # 无动作输入
        simulation_context.step(task._env._world)
        # 运行方法二
        # env.step(None)

        # 测试异步用
        task.planning_thread.start()
        

        robot_position, _ = task._agents[0].get_world_poses()
        actual_trajectory.append(robot_position[0, :2].cpu().numpy())
    

    # print("ideal_trajectory data:", task.pathtest_trajectory)

    # ideal_trajectory = torch.stack(ideal_trajectory).numpy() # 方式一：提前推导
    ideal_trajectory = torch.stack(task.pathtest_trajectory).numpy() # 方式二：实时记录
    actual_trajectory = np.array(actual_trajectory)

    # print("ideal_trajectory data:", ideal_trajectory)
    print("ideal_trajectory shape:", ideal_trajectory.shape)
    print("actual_trajectory shape:", actual_trajectory.shape)

    
    # 绘制轨迹图
    # 创建 FontProperties 对象
    prop = font_manager.FontProperties(fname=os.getcwd() + '/assets/Fonts/SIMSUN.TTC')

    # 设置 matplotlib 使用该字体
    plt.rcParams['font.sans-serif'] = [prop.get_name()]
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plt.rcParams.update({
        'axes.titlesize': 25,    # 标题字体大小
        'axes.labelsize': 20,    # 坐标轴标签（xlabel/ylabel）字体大小
        'legend.fontsize': 20    # 图例字体大小
    })

    plt.figure(figsize=(10, 6))
    plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], 'b.-', label="实际轨迹", alpha=0.03)
    plt.plot(ideal_trajectory[:, 0], ideal_trajectory[:, 1], 'g--', label="理想轨迹")
    plt.plot(goal_point[0], goal_point[1], 'r*', markersize=10, label="目标点")
    plt.xlabel("X 坐标 (m)", fontproperties=prop)
    plt.ylabel("Y 坐标 (m)", fontproperties=prop)
    plt.xlim(-10.0, 10.0)
    plt.ylim(-5.0, 5.0)
    plt.title("理想轨迹 vs 实际轨迹", fontproperties=prop)
    plt.legend(prop=prop)
    plt.grid(True)
    plt.savefig(os.getcwd() + "/screenshots/trajectory9.png")

    # 保存到 TXT 文件
    np.savetxt(os.getcwd() + "/results/path_test/ideal_trajectory.txt", ideal_trajectory, fmt="%.6f", delimiter=",", header="x,y", comments="")
    np.savetxt(os.getcwd() + "/results/path_test/actual_trajectory.txt", actual_trajectory, fmt="%.6f", delimiter=",", header="x,y", comments="")

    print("轨迹已成功保存为 TXT 文件！")

    # 参考轨迹（理想轨迹）与目标轨迹（实际轨迹）
    ref_traj = np.array(ideal_trajectory)  
    target_traj = np.array(actual_trajectory)  

    # 计算最近点的最小距离
    distances = cdist(ref_traj, target_traj, metric='euclidean')  
    min_distances = np.min(distances, axis=1)  

    # 计算平均偏差
    mean_deviation = np.mean(min_distances)
    print(f"平均偏差：{mean_deviation:.4f} 米")

    # 画偏差曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(min_distances)), min_distances, 'b.-', label="偏差曲线")
    plt.axhline(y=mean_deviation, color='r', linestyle='--', label=f"平均偏差: {mean_deviation:.4f}m")
    plt.xlabel("运行步数", fontproperties=prop)
    plt.ylabel("偏差 (m)", fontproperties=prop)
    plt.ylim(0.0, 0.2)
    plt.title("轨迹偏差（理想轨迹 vs 实际轨迹）", fontproperties=prop)
    plt.legend(prop=prop)
    plt.grid(True)
    plt.savefig(os.getcwd() + "/screenshots/deviation_curve9.png")
    plt.show()


if __name__ == '__main__':
    parse_hydra_configs()
