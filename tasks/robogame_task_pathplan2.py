from matplotlib import pyplot as plt
from utils.hydra_cfg.hydra_utils import *
from utils.hydra_cfg.reformat import omegaconf_to_dict
from utils.config_utils.path_utils import retrieve_checkpoint_path

from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.torch.rotations import get_euler_xyz
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrimView

from omni.isaac.range_sensor import _range_sensor

from rl_games.algos_torch import model_builder
from rl_games.algos_torch import torch_ext

from omegaconf import OmegaConf

import numpy as np
import pandas as pd
import torch
import math
import time
import os
import queue
import random

from tasks.base.rl_task import RLTask
from robots.articulations.robogame import RoboGame
from utils.pathplan.pathplan_utils import *

from omni.isaac.core.objects import DynamicCylinder

from utils.pathplan2.trajectory import Point, PathSearcher, TrajectoryOptimizer
from utils.pathplan2.costmap import CostMap
from utils.pathplan2.my_utils import path_plan2_coordinate_map2world, path_plan2_coordinate_world2map, find_nearest_point_on_path
# from skimage import io
from PIL import Image
import threading

INF = 999


# 围捕-逃逸 任务

class RoboGameTask(RLTask):
    def __init__(
            self,
            name,
            sim_config,
            env,
            offset=None
    ) -> None:

        print("RoboGameTask Init")

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._name = name

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._agentname = ["red0", "red1", "red2", "blue0"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        self._clipObservations = self._task_cfg["env"]["clipObservations"]
        self._clipActions = self._task_cfg["env"]["clipActions"]

        self._num_agents = self._task_cfg["env"]["numAgents"]  # used for multi-agent environments
        self._totnum_agents = self._num_agents + 1

        # 我加的部分
        self.env_name = self._cfg["env_name"]

        # 路径测试启动：self.pathtest 改为 True
        self.pathtest = False
        if self.pathtest:
            self._num_agents = self._totnum_agents = 1
            self.pathtest_target_point = [0.0, 0.0, 0.0]
            self.pathtest_trajectory = []

        # 课程学习启动：修改pre_physics_step和path_plan
        self.current_step = 0
        self.max_step = 300000
        self.evasion_mode = 0

        # 规划器
        # self.path_plan2_map_array = np.array(Image.open("./utils/pathplan2/map/711_casia.bmp"))
        self.path_plan2_map_array = np.array(Image.open("./utils/pathplan2/map/grid_map_200x100.bmp"))
        # map_array = np.flipud(self.path_plan2_map_array)   # 垂直镜像翻转
        map_array = self.path_plan2_map_array
        map_array = map_array.astype(np.uint8)
        self.expanded_map_array, self.esdf = CostMap(map_array).map2esdf()

        # 解耦规划器启动：开一个新线程专门规划路径，需解此处和pre_physics_step注释并修改调用处
        self.path_cache = [[None for _ in range(self._num_envs)] for _ in range(self._totnum_agents)]
        self.path_lock = threading.Lock()
        self.planning_thread = threading.Thread(target=self.async_path_planner, daemon=True)
        self.thread_started = False

        self._dt = self._task_cfg["sim"]["dt"]

        self._num_observations = 61
        self._dimension = 2
        self._num_actions = self._num_agents * self._dimension

        print(f"Observation dim: {self.num_observations}\nAction dim: {self.num_actions}")

        self._wheel_radius = 0.07
        self._wheel_base = 0.4
        self._axle_base = 0.414
        self._base = (self._wheel_base + self._axle_base) / 2

        # self.maxLinearSpeedx = 0.8
        # self.maxLinearSpeedy = 0.8
        self.pursuit_maxLinearSpeedx = 0.8
        self.pursuit_maxLinearSpeedy = 0.8
        self.evasion_maxLinearSpeedx = 0.8
        self.evasion_maxLinearSpeedy = 0.8
        if self.env_name == 'speed0':
            self.evasion_maxLinearSpeedx = 1.6
            self.evasion_maxLinearSpeedy = 1.6
            print(f'{self.env_name} config was changed successfully!')
        elif self.env_name == 'speed1':
            self.evasion_maxLinearSpeedx = 0.4
            self.evasion_maxLinearSpeedy = 0.4
            print(f'{self.env_name} config was changed successfully!')
        self.maxAngularSpeed = 1.6
        self.maxWheelSpeed = 12

        self.distance_scale = 3.0
        self.lidar_point_scale = 10.0

        self.scale_linvel = 2.0
        self.scale_angvel = math.pi

        self.shoot_range = 7.0
        self.shoot_tolerance = 0.02
        self.gimbal_pos_vel_scale = 2 / math.pi

        self.decimation_path_plan = 10  # 36

        # self.occupancy_map = get_occupancy_map('./utils/pathplan/map_difficult.png', False)
        self.occupancy_map = get_occupancy_map('./utils/pathplan/map_simple.png', False)
        self.occupancy_mapsize = self.occupancy_map.shape
        self.occupancy_to_goal_scale = 0.5

        # 以场地内边为准
        self.arena_half_length = 10
        self.arena_half_width = 5

        # 发布目标点距场地内边的裕度
        self.goal_pub_margin = 0.5  # 0.25

        # self.df = pd.read_csv('./utils/pathplan/map_difficult.csv')
        self.df = pd.read_csv('./utils/pathplan/map_simple.csv')
        self.dist_to_goal_map = self.df.iloc[:, 1:].to_numpy()
        print(self.occupancy_mapsize)

        # 出界裕度
        self.outside_margin = 0.15
        self.robot_dist_min = 1  # 机器人碰撞距离
        self.now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))

        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface()

        self.dir = os.getcwd()

        self.laser_render = False
        self.score_display = False
        self.laser_render = True
        self.score_display = True

        self._goal_colors = np.array([[1, 0, 0], [1, 0.25, 0], [1, 0, 0.25], [0, 0.75, 1]])
        self.random_vel = [[1, 1], [1, -1], [-1, 1], [-1, -1], [0, 1], [1, 0], [0, -1], [-1, 0]]

        RLTask.__init__(self, name, env)

        self.resolving_matrix = torch.tensor([[1, -1, 1, -1],
                                              [1, 1, -1, -1],
                                              [1, 1, 1, 1]], device=self._device)

        # 机器人速度因子
        # self.vel_factor = torch.tensor([[self.maxLinearSpeedx], [self.maxLinearSpeedy], [self.maxAngularSpeed * self._base]], device=self._device)
        self.pursuit_vel_factor = torch.tensor(
            [[self.pursuit_maxLinearSpeedx], [self.pursuit_maxLinearSpeedy], [self.maxAngularSpeed * self._base]],
            device=self._device)
        self.evasion_vel_factor = torch.tensor(
            [[self.evasion_maxLinearSpeedx], [self.evasion_maxLinearSpeedy], [self.maxAngularSpeed * self._base]],
            device=self._device)

        # self.vel_matrix = self.resolving_matrix * self.vel_factor / self._wheel_radius
        self.pursuit_vel_matrix = self.resolving_matrix * self.pursuit_vel_factor / self._wheel_radius
        self.evasion_vel_matrix = self.resolving_matrix * self.evasion_vel_factor / self._wheel_radius

        # print("vel_matrix", self.vel_matrix)
        print("pursuit_vel_matrix", self.pursuit_vel_matrix)
        print("evasion_vel_matrix", self.evasion_vel_matrix)

        # 每个车的轮子关节的索引
        self.joint_indices = torch.arange(4, dtype=torch.int32, device=self._device)

        self.camp_name = ['red', 'blue']

        self.target_point_trajectory = torch.Tensor([])

        print("RoboGameTask Init Done")
        return

    def set_up_scene(self, scene) -> None:
        print("set_up_scene")
        self._stage = get_current_stage()
        self.get_robogame()
        if self.env_name == 'obstacle0' or self.env_name == 'obstacle1':
            self.get_obstacle()
        self.get_target()
        super().set_up_scene(scene)

        self._agents = [ArticulationView(prim_paths_expr="/World/envs/.*/" + self._agentname[i], \
                                         name=self._agentname[i] + "_view", reset_xform_properties=False) for i in
                        range(self._totnum_agents)]

        self._goal_agents = [RigidPrimView(prim_paths_expr="/World/envs/.*/goal_" + self._agentname[i], \
                                           name="goal_" + self._agentname[i] + "_view", reset_xform_properties=False)
                             for i in range(self._totnum_agents)]

        for i in range(self._totnum_agents):
            scene.add(self._agents[i])
            scene.add(self._goal_agents[i])

        print("set_up_scene done")

        return

    def get_robogame(self):
        self.robogame = RoboGame(prim_path=self.default_zero_env_path, env_name=self.env_name, totnum_agents=self._totnum_agents)
        for i in range(self._totnum_agents):
            self._sim_config.apply_articulation_settings(self._agentname[i], get_prim_at_path(
                self.default_zero_env_path + "/" + self._agentname[i]), self._sim_config.parse_actor_config("robot"))

    def get_obstacle(self):
        if self.env_name == 'obstacle0':
            obstacles = [
                DynamicCylinder(
                    prim_path=self.default_zero_env_path + "/CylinderObstacle_0",
                    radius=0.3,
                    height=1.0,
                    position=(0.0, 0.0, 0.5),
                    name="obstacle_0"
                ),
                DynamicCylinder(
                    prim_path=self.default_zero_env_path + "/CylinderObstacle_1",
                    radius=0.1,
                    height=1.0,
                    position=(2.0, 3.2, 0.5),
                    name="obstacle_1"
                ),
                DynamicCylinder(
                    prim_path=self.default_zero_env_path + "/CylinderObstacle_2",
                    radius=0.1,
                    height=1.0,
                    position=(-2.6, -3.3, 0.5),
                    name="obstacle_2"
                )
            ]
            print(f'{self.env_name} config was changed successfully!')

        elif self.env_name == 'obstacle1':
            obstacles = [
                DynamicCylinder(
                    prim_path=self.default_zero_env_path + "/CylinderObstacle_0",
                    radius=0.3,
                    height=1.0,
                    position=(0.0, 0.0, 0.5),
                    name="obstacle_0"
                )
            ]
            print(f'{self.env_name} config was changed successfully!')

        for i, obstacle in enumerate(obstacles):
            self._sim_config.apply_articulation_settings(f"obstacle_{i}", get_prim_at_path(obstacle.prim_path),
                                                         self._sim_config.parse_actor_config("obstacle"))

            obstacle.set_collision_enabled(True)

    def get_target(self):
        scale = np.array([0.1, 0.1, 2])

        goal_agent = [DynamicCuboid(
            prim_path=self.default_zero_env_path + "/goal_" + self._agentname[i],
            translation=np.array([0.0, 0.0, 0.0]),
            name="goal_" + self._agentname[i],
            scale=scale,
            color=self._goal_colors[i])
            for i in range(self._totnum_agents)]

        for i in range(self._totnum_agents):
            self._sim_config.apply_articulation_settings("goal_" + self._agentname[i],
                                                         get_prim_at_path(goal_agent[i].prim_path),
                                                         self._sim_config.parse_actor_config(
                                                             "goal_" + self._agentname[i]))
            goal_agent[i].set_collision_enabled(False)

    def get_observations(self) -> dict:
        # # 解耦规划器启动，需启动path_plan2
        # self.path_plan2()
        # self.control_velocity_from_path()
        for i in range(self.decimation_path_plan):
            if i != 0:
                SimulationContext.step(self._env._world, render=True)
            # self.control_velocity_from_path()
            self.path_plan()
            # self.path_plan2()
            for j in range(9):
                # self.control_velocity_from_path()
                SimulationContext.step(self._env._world, render=False)

        for i in range(self._totnum_agents):
            self.robot_position[i, ...], self.robot_rot[i, ...] = self._agents[i].get_world_poses(clone=False)

        for i in range(self._totnum_agents):
            self.robot_orientation[i, :, 0], self.robot_orientation[i, :, 1], self.robot_orientation[i, :,
                                                                              2] = get_euler_xyz(self.robot_rot[i, ...])

        self.robot_translation = (self.robot_position - self._env_pos)

        for i in range(self._num_envs):
            self.lidar_depth[0, i, :] = torch.from_numpy(
                self.lidarInterface.get_linear_depth_data(f"/World/envs/env_{i}/red0/chassis/Lidar")[:, 0])
            self.lidar_depth[1, i, :] = torch.from_numpy(
                self.lidarInterface.get_linear_depth_data(f"/World/envs/env_{i}/red1/chassis/Lidar")[:, 0])
            self.lidar_depth[2, i, :] = torch.from_numpy(
                self.lidarInterface.get_linear_depth_data(f"/World/envs/env_{i}/red2/chassis/Lidar")[:, 0])
            
        # self.lidar_depth = torch.full_like(self.lidar_depth, 1.0)

        for i in range(self._totnum_agents):
            self.obs_buf[:, 18 * i:18 * i + 3] = self.robot_translation[i, ...]
            self.obs_buf[:, 18 * i] /= self.arena_half_length
            self.obs_buf[:, 18 * i + 1] /= self.arena_half_width
            self.obs_buf[:, 18 * i + 3:18 * i + 6] = self.robot_orientation[i, ...]
            self.obs_buf[:, 18 * i + 3:18 * i + 6] /= (2 * np.pi)
        for i in range(self.num_agents):
            self.obs_buf[:, 18 * i + 6:18 * i + 18] = self.lidar_depth[i, ...]
            self.obs_buf[:, 18 * i + 6:18 * i + 18] /= 20.0
        self.obs_buf[:, 60] = self.progress_buf / self._max_episode_length
        # self.obs_buf[:, 60] = 0.5

        observations = {
            "robogame": {
                "obs_buf": self.obs_buf
            }
        }

        if self.test:
            for j in range(self.num_envs):
                for i in range(self._totnum_agents):
                    self.trajectory_buf[j][i].append(self.robot_translation[i, j, :].tolist())

        return observations

    def pre_physics_step(self, actions) -> None:
        # # 解耦规划器启动1
        # if not self.thread_started:
        #     self.thread_started = True
        #     print("Starting async planner in pre_physics_step")
        #     self.planning_thread = threading.Thread(target=self.async_path_planner, daemon=True)
        #     self.planning_thread.start()

        # # 课程学习启动：不同阶段敌方策略不同
        # self.current_step += 1
        # if self.current_step > self.max_step / 2:
        #     self.evasion_mode = 1
        # if self.current_step < self.max_step / 3:
        #     self.evasion_mode = 0
        # elif self.current_step < self.max_step / 3 * 2:
        #     self.evasion_mode = 1
        # else:
        #     self.evasion_mode = 2

        if not self._env._world.is_playing():
            print("not playing")
            raise RuntimeError
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            print("RESET!")
            self.reset_idx(reset_env_ids)

        if actions == None and self.pathtest: # pathtest
            # print(self.target_point.shape)
            self.target_point[0, :, 0] = self.pathtest_target_point[0]
            self.target_point[0, :, 1] = self.pathtest_target_point[1]

        else:
            actions = actions.clone().to(self._device).reshape(-1, self._num_actions)

            self.last_target_point = self.target_point

            self.target_point[0:self._num_agents, :, 0] = actions[:, [2 * i for i in range(self._num_agents)]].transpose(0,
                                                                                                                         1) * (
                                                                      self.arena_half_length - self.goal_pub_margin)
            self.target_point[0:self._num_agents, :, 1] = actions[:,
                                                          [2 * i + 1 for i in range(self._num_agents)]].transpose(0, 1) * (
                                                                      self.arena_half_width - self.goal_pub_margin)


            self.target_point[self._num_agents, :, 0] = (2 * torch.rand(self._num_envs, dtype=torch.float32,
                                                                        device=self._device) - 1) * (
                                                                    self.arena_half_length - self.goal_pub_margin)  # prey
            self.target_point[self._num_agents, :, 1] = (2 * torch.rand(self._num_envs, dtype=torch.float32,
                                                                    device=self._device) - 1) * (
                                                                self.arena_half_width - self.goal_pub_margin)

            # self.target_point[self._num_agents, :, 0] = 0.0
            # self.target_point[self._num_agents, :, 1] = 0.0

            # self.target_point = torch.clamp(self.target_point, -self._task.clip_actions, self._task.clip_actions).clone()
            # self.target_point[0:self._num_agents, :, 0] = torch.clamp(self.target_point[0:self._num_agents, :, 0], -5.0, 5.0)
            # self.target_point[0:self._num_agents, :, 1] = torch.clamp(self.target_point[0:self._num_agents, :, 1], -10.0, 10.0)

            # print(f'actions: {actions}')
            # print(self.target_point)

        for ai in range(self._totnum_agents):
            self.target_point_index[ai, :] = parallel_pos_to_index(self.target_point[ai, :, 0:2],
                                                                   self.occupancy_mapsize)
            
        for ai in range(self._totnum_agents):
            for i in range(self._num_envs):
                goal_coordinate = index_to_coordinate(self.target_point_index[ai, i].item(), self.occupancy_mapsize[1])
                if judge_obs(self.occupancy_map, goal_coordinate):
                    goal_coordinate = self.find_nearest_goal(goal_coordinate)
                    # try:
                    self.target_point_index[ai, i] = coordinate_to_index(goal_coordinate, self.occupancy_mapsize[1])
                    # except:
                    #     print(self.target_point)

        for i in range(self._totnum_agents):
            self._goal_agents[i].set_world_poses(self.target_point[i, ...] + self._env_pos)


    def find_nearest_goal(self, goal_coordinate):
        visited = set()  # 使用集合而不是列表
        q = queue.Queue()
        q.put(tuple(goal_coordinate))  # 存储为元组
        while not q.empty():
            curnode = q.get()
            if tuple(curnode) in visited:  # 跳过已访问节点
                continue
            visited.add(tuple(curnode))
            for i in range(-1, 2):
                for j in range(-1, 2):
                    temp = (curnode[0] + i, curnode[1] + j)  # 使用元组
                    # if (0 <= temp[0] < self.occupancy_mapsize[0] and 
                    #     0 <= temp[1] < self.occupancy_mapsize[1]):
                    if not judge_obs(self.occupancy_map, temp):
                        return list(temp)  # 返回列表形式
                    if temp not in visited:
                        q.put(temp)
        return None

    def calc_obstacles_pulse(self, coordinate, k=2):
        pulse = [0.0, 0.0]
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                if i != 0 or j != 0:
                    temp = [coordinate[0] + i, coordinate[1] + j]
                    if judge_obs(self.occupancy_map, temp):
                        pulse[0] += -i * 1.0 / (i * i + j * j)
                        pulse[1] += -j * 1.0 / (i * i + j * j)
        return pulse

    def path_plan(self):
        # print('Path plan')
        for i in range(self._totnum_agents):
            self.robot_position[i, ...], self.robot_rot[i, ...] = self._agents[i].get_world_poses(clone=False)
            self.robot_velocities[i, ...] = self._agents[i].get_velocities(clone=False)
            self.robot_roll[i, :], _, self.robot_yaw[i, :] = get_euler_xyz(self.robot_rot[i, ...])

        self.robot_translation = (self.robot_position - self._env_pos)
        robot_translation_index = torch.zeros((self._totnum_agents, self._num_envs), dtype=torch.int32,
                                              device=self._device)

        start_t = time.time()
        for ai in range(self._totnum_agents):
            robot_translation_index[ai, :] = parallel_pos_to_index(self.robot_translation[ai, :, 0:2],
                                                                   self.occupancy_mapsize)
            for i in range(self._num_envs):
                red0_coordinate = index_to_coordinate(robot_translation_index[ai, i].item(), self.occupancy_mapsize[1])
                min_dis_direct = None
                min_dis = INF
                min_dis_node = None
                if judge_obs(self.occupancy_map, red0_coordinate):
                    neighbors = find_neighbor(red0_coordinate, self.occupancy_map)
                    for node in neighbors:
                        temp = self.dist_to_goal_map[coordinate_to_index(node, self.occupancy_mapsize[1])][
                            self.target_point_index[ai, i]]
                        if temp < min_dis:
                            min_dis = temp
                            min_dis_direct = [node[0] - red0_coordinate[0], node[1] - red0_coordinate[1]]
                            min_dis_node = node
                else:
                    neighbors = find_neighbor(red0_coordinate, self.occupancy_map)
                    for node in neighbors:
                        temp = self.dist_to_goal_map[coordinate_to_index(node, self.occupancy_mapsize[1])][
                            self.target_point_index[ai, i]]
                        if temp < min_dis:
                            min_dis = temp
                            min_dis_direct = [node[0] - red0_coordinate[0], node[1] - red0_coordinate[1]]
                            min_dis_node = node

                if self.pathtest:
                    if min_dis_node == None:
                        min_dis_node = random.choice(neighbors)
                    self.pathtest_trajectory.append(parallel_coordinate_to_pos(torch.tensor(min_dis_node), self.occupancy_mapsize))

                pulse_vel = self.calc_obstacles_pulse(red0_coordinate)
                if min_dis_direct == None:
                    min_dis_direct = torch.Tensor(random.choice(self.random_vel))
                world_xspeed = -min_dis_direct[1] - pulse_vel[1] * 0.2
                world_yspeed = min_dis_direct[0] + pulse_vel[0] * 0.2
                self.robot_vel[ai, i, 0] = torch.cos(self.robot_yaw[ai, i]) * world_xspeed + torch.sin(
                    self.robot_yaw[ai, i]) * world_yspeed
                self.robot_vel[ai, i, 1] = -torch.sin(self.robot_yaw[ai, i]) * world_xspeed + torch.cos(
                    self.robot_yaw[ai, i]) * world_yspeed
                self.robot_vel[ai, i, 0:2] *= 0.7

                # orientation control
                desired_yaw = torch.atan2(torch.tensor(world_yspeed), torch.tensor(world_xspeed))
                yaw_diff = desired_yaw - self.robot_yaw[ai, i]
                yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
                angular_velocity = 0.5 * yaw_diff
                self.robot_vel[ai, i, 2] = angular_velocity

        end_t = time.time()
        # print((f'Time: {end_t - start_t}'))

        # self.robot_vel[:,:, 2] = (2 * torch.rand((self._totnum_agents,self._num_envs)) - 1)*0.0

        # if self.evasion_mode == 1:
        #     self.robot_vel[self._totnum_agents-1, :, :] = 0.0
        # self.robot_vel[self._totnum_agents-1, :, :] = 0.0

        # self.robot_vel[:, :, 0] = 0.0
        # self.robot_vel[:, :, 2] = 0.0
        # if self.progress_buf[0] > 25:
        #     self.robot_vel[:, :, 1] = -1.0
        # else:
        #     self.robot_vel[:, :, 1] = 1.0

        # todo, forward kinematics
        # self.wheel_dof_vel = torch.matmul(self.robot_vel, self.vel_matrix)
        # self.wheel_dof_vel = torch.clamp(self.wheel_dof_vel, -self.maxWheelSpeed, self.maxWheelSpeed)
        #
        # for i in range(self._totnum_agents):
        #     self._agents[i].set_joint_velocity_targets(self.wheel_dof_vel[i, ...], joint_indices=self.joint_indices)

        self.pursuit_wheel_dof_vel = torch.matmul(self.robot_vel, self.pursuit_vel_matrix)
        self.pursuit_wheel_dof_vel = torch.clamp(self.pursuit_wheel_dof_vel, -self.maxWheelSpeed, self.maxWheelSpeed)
        self.evasion_wheel_dof_vel = torch.matmul(self.robot_vel, self.evasion_vel_matrix)
        self.evasion_wheel_dof_vel = torch.clamp(self.evasion_wheel_dof_vel, -self.maxWheelSpeed, self.maxWheelSpeed)

        for i in range(self._num_agents):
            self._agents[i].set_joint_velocity_targets(self.pursuit_wheel_dof_vel[i, ...],
                                                       joint_indices=self.joint_indices)
        self._agents[self._totnum_agents-1].set_joint_velocity_targets(self.evasion_wheel_dof_vel[self._totnum_agents-1, ...],
                                                                  joint_indices=self.joint_indices)
        

    # 解耦规划器对应函数
    def path_plan2(self):
        expanded_map_array = self.expanded_map_array
        esdf = self.esdf
    
        for i in range(self._totnum_agents):
            self.robot_position[i, ...], self.robot_rot[i, ...] = self._agents[i].get_world_poses(clone=False)
            self.robot_velocities[i, ...] = self._agents[i].get_velocities(clone=False)
            self.robot_roll[i, :], _, self.robot_yaw[i, :] = get_euler_xyz(self.robot_rot[i, ...])

        self.robot_translation = (self.robot_position - self._env_pos)

        start_t = time.time()
        for ai in range(self._totnum_agents):
            for i in range(self._num_envs):
                x, y = self.robot_translation[ai, i, 0], self.robot_translation[ai, i, 1]
                goal_x, goal_y = self.target_point[ai, i, 0], self.target_point[ai, i, 1]
    
                trans_x, trans_y = path_plan2_coordinate_world2map(x, y)
                trans_goal_x, trans_goal_y = path_plan2_coordinate_world2map(goal_x, goal_y)
    
                current_pt = Point(int(trans_x), int(trans_y))
                goal = Point(int(trans_goal_x), int(trans_goal_y))
    
                pth = PathSearcher().plan(expanded_map_array, current_pt, goal)
                pth = TrajectoryOptimizer(esdf).plan(expanded_map_array, pth)
    
                p = find_nearest_point_on_path(trans_x, trans_y, pth)
                print(pth, x, y, current_pt, goal_x, goal_y, goal)
                if p == len(pth) - 1:
                    self.robot_vel[ai, i, :] = 0.0
                    continue
                next_goal = pth[p+1]

                next_goal_x, next_goal_y = path_plan2_coordinate_map2world(next_goal.x, next_goal.y)
                # self.naive_velocity_controller(velocity, watch_on, [dx, dy], [x, y, theta])
                vel_direct = [next_goal_x - x, next_goal_y - y]

                
                # plt.figure(figsize=(12.0, 6.0))
                # plt.imshow(expanded_map_array, cmap="gray")
                # path_x = [pt.x for pt in pth]
                # path_y = [pt.y for pt in pth]
                # plt.plot(path_x, path_y, label=f"Path: Agent {i}", linestyle='--')  # 绘制路径
                # plt.scatter(next_goal.x, next_goal.y, marker='o', s=30, label=f"Subgoal: Agent {i}")
                # plt.plot(
                #     trans_goal_x, trans_goal_y,
                #     marker='x', markersize=20, label=f"Goal: Agent {i}"
                # )
                # plt.legend()
                # plt.savefig(os.getcwd() + "/utils/pathplan2/result/result_opt_5.png")
                # plt.close()

                if self.pathtest:
                    self.pathtest_trajectory.append(parallel_coordinate_to_pos(torch.tensor([next_goal.x, next_goal.y]), self.occupancy_mapsize))

                # calculate speed
                world_xspeed = vel_direct[0] * 2
                world_yspeed = vel_direct[1] * 2
                self.robot_vel[ai, i, 0] = torch.cos(self.robot_yaw[ai, i]) * world_xspeed + torch.sin(
                    self.robot_yaw[ai, i]) * world_yspeed
                self.robot_vel[ai, i, 1] = -torch.sin(self.robot_yaw[ai, i]) * world_xspeed + torch.cos(
                    self.robot_yaw[ai, i]) * world_yspeed
                self.robot_vel[ai, i, 0:2] *= 0.7

                # orientation control
                desired_yaw = torch.atan2(torch.tensor(world_yspeed), torch.tensor(world_xspeed))
                yaw_diff = desired_yaw - self.robot_yaw[ai, i]
                yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
                angular_velocity = 0.5 * yaw_diff
                self.robot_vel[ai, i, 2] = angular_velocity

        end_t = time.time()
        # print(f'Time:{end_t - start_t}')

        # self.robot_vel[self._totnum_agents-1, :, :] = 0.0

        self.pursuit_wheel_dof_vel = torch.matmul(self.robot_vel, self.pursuit_vel_matrix)
        self.pursuit_wheel_dof_vel = torch.clamp(self.pursuit_wheel_dof_vel, -self.maxWheelSpeed, self.maxWheelSpeed)
        self.evasion_wheel_dof_vel = torch.matmul(self.robot_vel, self.evasion_vel_matrix)
        self.evasion_wheel_dof_vel = torch.clamp(self.evasion_wheel_dof_vel, -self.maxWheelSpeed, self.maxWheelSpeed)

        for i in range(self._num_agents):
            self._agents[i].set_joint_velocity_targets(self.pursuit_wheel_dof_vel[i, ...],
                                                       joint_indices=self.joint_indices)
        self._agents[self._totnum_agents-1].set_joint_velocity_targets(self.evasion_wheel_dof_vel[self._totnum_agents-1, ...],
                                                                  joint_indices=self.joint_indices)
            

    def async_path_planner(self):
        print("Thread START")
        for i in range(self._totnum_agents):
            self.robot_position[i, ...], self.robot_rot[i, ...] = self._agents[i].get_world_poses(clone=False)
            self.robot_velocities[i, ...] = self._agents[i].get_velocities(clone=False)
            self.robot_roll[i, :], _, self.robot_yaw[i, :] = get_euler_xyz(self.robot_rot[i, ...])

        self.robot_translation = (self.robot_position - self._env_pos)

        while True:
            for ai in range(self._totnum_agents):
                for i in range(self._num_envs):
                    x, y = self.robot_translation[ai, i, 0], self.robot_translation[ai, i, 1]
                    goal_x, goal_y = self.target_point[ai, i, 0], self.target_point[ai, i, 1]

                    trans_x, trans_y = path_plan2_coordinate_world2map(x, y)
                    trans_goal_x, trans_goal_y = path_plan2_coordinate_world2map(goal_x, goal_y)

                    current_pt = Point(int(trans_x), int(trans_y))
                    goal = Point(int(trans_goal_x), int(trans_goal_y))

                    try:
                        pth = PathSearcher().plan(self.expanded_map_array, current_pt, goal)
                        pth = TrajectoryOptimizer(self.esdf).plan(self.expanded_map_array, pth)

                        with self.path_lock:
                            self.path_cache[ai][i] = pth

                        # print(f'Robot{ai}: ' ,pth, x, y, current_pt, goal_x, goal_y, goal)
                            
                    except Exception as e:
                        print(f"[Planner Error] Agent {ai}-{i}: {e}")

            time.sleep(0.5)

    def control_velocity_from_path(self):
        for ai in range(self._totnum_agents):
            for i in range(self._num_envs):
                with self.path_lock:
                    pth = self.path_cache[ai][i]

                if pth is None or len(pth) < 2:
                    self.robot_vel[ai, i, :] = 0.0
                    continue

                x, y = self.robot_translation[ai, i, 0], self.robot_translation[ai, i, 1]
                trans_x, trans_y = path_plan2_coordinate_world2map(x, y)
                p = find_nearest_point_on_path(trans_x, trans_y, pth)

                if p >= len(pth) - 1:
                    self.robot_vel[ai, i, :] = 0.0
                    continue

                next_goal = pth[p + 1]
                next_goal_x, next_goal_y = path_plan2_coordinate_map2world(next_goal.x, next_goal.y)

                if ai == 0:
                    print(f'Robot{ai}: ' , trans_x, trans_y, next_goal, x, y, next_goal_x, next_goal_y)

                # 设置速度
                # vel_direct = [next_goal_x - x, next_goal_y - y]
                # world_xspeed = vel_direct[0]
                # world_yspeed = vel_direct[1]

                cur_coordinate = index_to_coordinate(parallel_pos_to_index(self.robot_translation[ai, :, 0:2],
                                                                   self.occupancy_mapsize), self.occupancy_mapsize[1])
                next_goal_coordinate = index_to_coordinate(parallel_pos_to_index(torch.Tensor([[next_goal_x, next_goal_y]]),
                                                                   self.occupancy_mapsize), self.occupancy_mapsize[1])
                min_dis_direct = [next_goal_coordinate[0] - cur_coordinate[0], next_goal_coordinate[1] - cur_coordinate[1]]
                pulse_vel = self.calc_obstacles_pulse(cur_coordinate)
                world_xspeed = -min_dis_direct[1] - pulse_vel[1] * 0.2
                world_yspeed = min_dis_direct[0] + pulse_vel[0] * 0.2

                # 方向控制
                self.robot_vel[ai, i, 0] = torch.cos(self.robot_yaw[ai, i]) * world_xspeed + torch.sin(self.robot_yaw[ai, i]) * world_yspeed
                self.robot_vel[ai, i, 1] = -torch.sin(self.robot_yaw[ai, i]) * world_xspeed + torch.cos(self.robot_yaw[ai, i]) * world_yspeed
                self.robot_vel[ai, i, 0:2] *= 0.1

                # 角速度
                desired_yaw = torch.atan2(torch.tensor(world_yspeed), torch.tensor(world_xspeed))
                yaw_diff = desired_yaw - self.robot_yaw[ai, i]
                yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
                angular_velocity = 0.5 * yaw_diff
                self.robot_vel[ai, i, 2] = angular_velocity

        # 写入关节速度
        self.pursuit_wheel_dof_vel = torch.matmul(self.robot_vel, self.pursuit_vel_matrix)
        self.pursuit_wheel_dof_vel = torch.clamp(self.pursuit_wheel_dof_vel, -self.maxWheelSpeed, self.maxWheelSpeed)

        self.evasion_wheel_dof_vel = torch.matmul(self.robot_vel, self.evasion_vel_matrix)
        self.evasion_wheel_dof_vel = torch.clamp(self.evasion_wheel_dof_vel, -self.maxWheelSpeed, self.maxWheelSpeed)

        for i in range(self._num_agents):
            self._agents[i].set_joint_velocity_targets(self.pursuit_wheel_dof_vel[i, ...],
                                                       joint_indices=self.joint_indices)
        self._agents[self._totnum_agents - 1].set_joint_velocity_targets(
            self.evasion_wheel_dof_vel[self._totnum_agents - 1, ...],
            joint_indices=self.joint_indices)


    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        indices = env_ids.to(dtype=torch.int32)  # env_ids.dtype: torch.int64

        # reset joint velocities
        self.dof_pos[env_ids, :] = 0.0
        self.dof_vel[env_ids, :] = 0.0

        for i in range(self._totnum_agents):
            self._agents[i].set_joint_positions(self.dof_pos[env_ids], indices=indices)

        velocities_default = torch.zeros((num_resets, 6), device=self._device)
        random_respawn_position = 2 * torch.rand((self._totnum_agents, num_resets, 3), device=self._device) - 1
        random_respawn_position[..., 0] = random_respawn_position[..., 0] * (
                    self.arena_half_length - self.goal_pub_margin)
        random_respawn_position[..., 1] = random_respawn_position[..., 1] * (
                    self.arena_half_width - self.goal_pub_margin)
        random_respawn_position[..., 2] = 0.7
        random_index = torch.zeros((self._totnum_agents, num_resets), dtype=torch.int32, device=self._device)
        robot_coordinate = torch.zeros((num_resets, 2), dtype=torch.int32, device=self._device)
        for ai in range(self._totnum_agents):
            random_index[ai, :] = parallel_pos_to_index(random_respawn_position[ai, :, 0:2], self.occupancy_mapsize)
            for i in range(num_resets):
                temp_coordinate = index_to_coordinate(random_index[ai, i].item(), self.occupancy_mapsize[1])
                if judge_obs(self.occupancy_map, temp_coordinate):
                    temp_coordinate = self.find_nearest_goal(temp_coordinate)
                robot_coordinate[i] = torch.Tensor(temp_coordinate)
            random_respawn_position[ai, :, 0:2] = parallel_coordinate_to_pos(robot_coordinate, self.occupancy_mapsize)
        random_respawn_position = random_respawn_position + self._env_pos[env_ids]
        for i in range(self._totnum_agents):
            self._agents[i].set_joint_positions(self.dof_pos[env_ids], indices=indices)
            self._agents[i].set_joint_velocities(self.dof_vel[env_ids], indices=indices)
            if self.env_name == 'poses0' or self.env_name == 'poses1':
                self._agents[i].set_world_poses(self.initial_pos_agents[i][env_ids].clone(), self.initial_rot_agents[i][env_ids].clone(), indices=indices)
            else:
                self._agents[i].set_world_poses(random_respawn_position[i, ...].clone(),
                                            self.initial_rot_agents[i, env_ids, :].clone(), indices=indices)
            if self.pathtest:
                self._agents[i].set_world_poses(self.initial_pos_agents[i][env_ids].clone(), self.initial_rot_agents[i][env_ids].clone(), indices=indices)
            self._agents[i].set_velocities(velocities_default, indices=indices)
            self._goal_agents[i].set_world_poses(self.initial_goal_agents[i][env_ids].clone(), indices=indices)
        for i in range(self.num_agents):
            self.pre_distance[i, env_ids] = self._distance(random_respawn_position[-1],
                                                           random_respawn_position[i])

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        random_respawn_position = 2 * torch.rand((self._totnum_agents, num_resets, 3), device=self._device) - 1
        random_respawn_position[..., 0] = random_respawn_position[..., 0] * (
                    self.arena_half_length - self.goal_pub_margin)
        random_respawn_position[..., 1] = random_respawn_position[..., 1] * (
                    self.arena_half_width - self.goal_pub_margin)
        random_respawn_position[..., 2] = 0.7

        for ai in range(self._totnum_agents):
            self.target_point_index[ai, env_ids] = parallel_pos_to_index(random_respawn_position[ai, :, 0:2],
                                                                         self.occupancy_mapsize)
            for i in env_ids:
                goal_coordinate = index_to_coordinate(self.target_point_index[ai, i].item(), self.occupancy_mapsize[1])
                if judge_obs(self.occupancy_map, goal_coordinate):
                    goal_coordinate = self.find_nearest_goal(goal_coordinate)
                    self.target_point_index[ai, i] = coordinate_to_index(goal_coordinate, self.occupancy_mapsize[1])

        if self.test:
            self.time_buf[env_ids] = time.time()
            self.trajectory_buf[env_ids] = [[] for i in range(self._totnum_agents)]
            self.rew_record[env_ids] = []

    def post_reset(self):
        self.dof_pos = self._agents[0].get_joint_positions()  # r+1
        self.dof_vel = self._agents[0].get_joint_velocities()  # r+2
        self.initial_goal_agents = torch.zeros((self._totnum_agents, self._num_envs, 3), dtype=torch.float32, device=self._device)
        self.initial_pos_agents = torch.zeros((self._totnum_agents, self._num_envs, 3), dtype=torch.float32, device=self._device)
        self.initial_rot_agents = torch.zeros((self._totnum_agents, self._num_envs, 4), dtype=torch.float32, device=self._device)

        for i in range(self._totnum_agents):
            self.initial_pos_agents[i], self.initial_rot_agents[i] = self._agents[i].get_world_poses()
            self.initial_goal_agents[i], _ = self._goal_agents[i].get_world_poses()

        self.robot_position = torch.zeros((self._totnum_agents, self._num_envs, 3), dtype=torch.float32,
                                          device=self._device)
        self.robot_position = self.initial_pos_agents.clone()
        self.pre_distance = torch.zeros((self.num_agents, self._num_envs), dtype=torch.float32, device=self._device)
        self.robot_rot = torch.zeros((self._totnum_agents, self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.robot_rot = self.initial_rot_agents.clone()

        self.robot_orientation = torch.zeros((self._totnum_agents, self._num_envs, 3), dtype=torch.float32, device=self._device)

        self.robot_velocities = torch.zeros((self._totnum_agents, self._num_envs, 6), dtype=torch.float32,
                                            device=self._device)

        self.obs_buf_op = torch.zeros((self._num_envs, self._num_observations), dtype=torch.float32,
                                      device=self._device)

        self.robot_roll = torch.zeros((self._totnum_agents, self._num_envs), dtype=torch.float32, device=self._device)
        self.robot_yaw = torch.zeros((self._totnum_agents, self._num_envs), dtype=torch.float32, device=self._device)

        # 相对位置
        self.robot_translation = torch.zeros((self._totnum_agents, self._num_envs, 2), dtype=torch.float32,
                                             device=self._device)

        # 目标点位置
        self.target_point = torch.ones((self._totnum_agents, self._num_envs, 3), dtype=torch.float32,
                                       device=self._device) * 0.06
        self.last_target_point = torch.ones((self._totnum_agents, self._num_envs, 3), dtype=torch.float32,
                                       device=self._device) * 0.06

        self.robot_vel = torch.zeros((self._totnum_agents, self._num_envs, 3), dtype=torch.float32, device=self._device)

        self.wheel_dof_vel = torch.zeros((self._totnum_agents, self._num_envs, 4), dtype=torch.float32,
                                         device=self._device)  # ?

        self.lidar_depth = torch.ones((self._num_agents, self._num_envs, 12), dtype=torch.float32,
                                      device=self._device) * self.lidar_point_scale

        self.robot_num_dof = self._agents[0].num_dof  # r+3
        self.target_point_index = torch.zeros((self._totnum_agents, self._num_envs), dtype=torch.int32,
                                              device=self._device, requires_grad=False)

        print("post reset done.")

    def _distance(self, pos1, pos2):  # pos: envs x dimension
        return torch.sqrt(torch.sum(torch.square(pos1 - pos2), axis=1))

    def _is_collision(self, pos1, pos2, dist_min):  # pos: envs x dimension
        dist = self._distance(pos1, pos2)
        vdist_min = dist_min * torch.ones((self._num_envs,), device=self._device, dtype=torch.float)
        return torch.gt(vdist_min, dist)

    def _compute_angle_coverage(self, center, agents):
        vecs = []
        for agent in agents:
            v = agent - center
            v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)
            vecs.append(v)

        total_angle = 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                dot = (vecs[i] * vecs[j]).sum(-1).clamp(-1.0, 1.0)
                angle = torch.acos(dot) * (180.0 / math.pi)
                total_angle += angle

        return total_angle  # 最大接近 360°


    def calculate_metrics(self) -> None:
        timeover = self.progress_buf >= self._max_episode_length
        rew = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float)
        step_penalty = True

        for i in range(3):
            rew += -torch.sum(1 / (self.lidar_depth[i, :, :] + 0.1)) / 12

        collision_flag = torch.ones((self._num_envs,), device=self._device, dtype=torch.bool)
        for agenti in range(0, self._num_agents):
            rew += 20 * self._is_collision(self.robot_position[self._num_agents], self.robot_position[agenti],
                                           self.robot_dist_min)
            collision_flag &= self._is_collision(self.robot_position[self._num_agents], self.robot_position[agenti],
                                                 self.robot_dist_min)
        rew += 100 * collision_flag

        if step_penalty:
            rew += (-1) * self.progress_buf
        rew += -50 * timeover

        self.rew_buf = rew

        if self.test:
            for i in range(self._num_envs):
                self.rew_record[i].append(self.rew_buf[i])

    # # 第二版的奖励函数
    # def calculate_metrics(self) -> None:
    #     rew = torch.zeros((self._num_envs,), device=self._device)

    #     # 1. 避障雷达惩罚
    #     for i in range(3):
    #         rew += -torch.sum(1 / (self.lidar_depth[i, :, :] + 0.1)) / 12

    #     # 2. 靠近敌车奖励
    #     for agenti in range(self._num_agents):
    #         distance = self._distance(self.robot_position[self._num_agents], self.robot_position[agenti])
    #         rew += 3.0 - distance
    #         # rew += -1.0 * (self._distance(self.last_target_point[agenti], self.target_point[agenti]) > 7.0).float()
    #         # rew += -1.0 * (self._distance(self.target_point[agenti], self.robot_position[self._num_agents]) > 5.0).float()
    #         # rew += 1 - 0.5 * self._distance(self.robot_position[self.num_agents], self.robot_position[agenti])
    #         # for agentj in range(agenti+1, self._num_agents):
    #         #     rew += -10 * self._is_collision(self.robot_position[agenti], self.robot_position[agentj],
    #         #                                self.robot_dist_min)

    #     # 3. 夹角奖励
    #     angle_sum = self._compute_angle_coverage(self.robot_position[self._num_agents], self.robot_position[:3])
    #     rew += 0.02 * angle_sum

    #     # 4. 成功围捕大奖励
    #     collision_flag = torch.ones((self._num_envs,), device=self._device, dtype=torch.bool)
    #     for agenti in range(self._num_agents):
    #         collision_flag &= self._is_collision(self.robot_position[self._num_agents], self.robot_position[agenti],
    #                                              self.robot_dist_min)
    #     rew += 100.0 * collision_flag.float()

    #     # 5. 步长惩罚
    #     rew += -0.5

    #     self.rew_buf = rew


    def is_done(self) -> None:
        is_caught = torch.ones((self._num_envs,), device=self._device, dtype=torch.bool)
        for agenti in range(0, self._num_agents):
            is_caught &= self._is_collision(self.robot_position[self._num_agents], self.robot_position[agenti],
                                            self.robot_dist_min)

        min_trans, _ = torch.min(torch.abs(self.robot_translation.permute(1, 0, 2).flatten(1)), dim=1)
        # 统一采用长边的边界
        outside = torch.where(min_trans > ((self.arena_half_length + self.outside_margin) / self.distance_scale), True,
                              False)

        min_roll, _ = torch.min(torch.abs(self.robot_roll.flatten(0, 1) - 3.14), dim=0)
        # 翻车时，roll大约在±pi, 正常时，roll大概在0或者2pi左右, pi±0.14
        overturn = torch.where(min_roll < 1.57, True, False)  # 1111ljx pi±(pi/2)

        resets = is_caught | outside | overturn

        resets = torch.where(self.progress_buf >= self._max_episode_length, torch.ones_like(self.reset_buf), resets)

        if self.test:
            time_now = time.time()
            for i in range(len(self.reset_buf)):
                if is_caught[i]:
                    self.game_record.append(
                        [True, self.progress_buf[i].item(), time_now - self.time_buf[i].item(), self.trajectory_buf[i], self.rew_record[i]])
                elif self.progress_buf[i] >= self._max_episode_length:
                    self.game_record.append([False, self.progress_buf[i].item(), time_now - self.time_buf[i].item(),
                                             self.trajectory_buf[i], self.rew_record[i]])
                # print(self.rew_record[i])

        self.reset_buf = resets
