from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.isaac.wheeled_robots")

from typing import Optional
import numpy as np

# from omni.isaac.core.robots.robot import Robot
from omni.isaac.wheeled_robots.robots import WheeledRobot

# from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdGeom, Sdf, Gf, UsdShade, UsdPhysics, PhysxSchema
from omni.isaac.core.utils.stage import get_current_stage

from omni.isaac.core.prims import XFormPrim

import os

import torch
from matplotlib.path import Path


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows - 1, :] += (hf[1:num_rows, :] - hf[:num_rows - 1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols - 1] += (hf[:, 1:num_cols] - hf[:, :num_cols - 1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols - 1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows - 1, :num_cols - 1] += (
                    hf[1:num_rows, 1:num_cols] - hf[:num_rows - 1, :num_cols - 1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (
                    hf[:num_rows - 1, :num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1:stop:2, 0] = ind0
        triangles[start + 1:stop:2, 1] = ind2
        triangles[start + 1:stop:2, 2] = ind3

    return vertices, triangles


class RoboGame(WheeledRobot):
    def __init__(
            self,
            prim_path: str,
            env_name='default',
            totnum_agents=4
            # name: Optional[str] = "simbot",
            # position: Optional[np.ndarray] = None,
            # orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._prim_path = prim_path

        dir = os.getcwd()

        self._usd_path = dir + "/assets/robot_mecanum.usd"
        # self._usd_path = dir + "/assets/robot_mecanum_changelidar.usd"
        # self._usd_path = dir + "/assets/robot_mecanum_nowheel.usd"

        self.device = torch.device('cuda:0')
        self._stage = get_current_stage()

        self.setup_arena()

        # self.name_list = ['red0', 'red1', 'blue0', 'blue1']

        self.robots = ['red0', 'red1', 'red2', 'blue0']

        if totnum_agents == 1:
            self.robots = ['red0']
            self.poses = {
                'red0': [[-7.0, -3.5, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
            }

        elif totnum_agents == 3:
            self.robots = ['red0', 'red1', 'blue0']
            self.poses = { # poses0
                'red0': [[-8.8, -0.3, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
                'red1': [[-8.8, -2.1, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
                'blue0': [[8.0, 1.5, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            }

            if env_name == 'poses0':
                self.poses = { # poses0
                    'red0': [[-8.8, -0.3, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
                    'red1': [[-8.8, -2.1, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
                    'blue0': [[8.0, 1.5, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                }
                print(f'{env_name} config was changed successfully!')
                
            elif env_name == 'poses1':
                self.poses = { # poses1
                    'red0': [[-1.4, 1.2, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
                    'red1': [[-1.4, -0.25, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
                    'blue0': [[0.3, 0.4, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                }
                print(f'{env_name} config was changed successfully!')

        else: # totnum_agents == 4
            self.poses = { # poses0
                'red0': [[-8.8, -0.3, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
                'red1': [[-8.8, -2.1, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
                'red2': [[-8.8, -4.0, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
                'blue0': [[8.0, 1.5, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            }

            if env_name == 'poses0':
                self.poses = { # poses0
                    'red0': [[-8.8, -0.3, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
                    'red1': [[-8.8, -2.1, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
                    'red2': [[-8.8, -4.0, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
                    'blue0': [[8.0, 1.5, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                }
                print(f'{env_name} config was changed successfully!')
            
            elif env_name == 'poses1':
                self.poses = { # poses1
                    'red0': [[-1.4, 1.2, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
                    'red1': [[-1.4, -0.25, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
                    'red2': [[-0.3, -1.6, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
                    'blue0': [[0.3, 0.4, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                }
                print(f'{env_name} config was changed successfully!')


        # self.poses = { # poses2
        #     'red0': [[-8.8, -0.3, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
        #     'red1': [[8.0, 1.5, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
        #     'red2': [[-8.8, -4.0, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
        #     'blue0': [[0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        # }
        
        # self.poses = { # poses3
        #     'red0': [[-6.0, -4.0, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
        #     'red1': [[-6.0, -2.0, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
        #     'red2': [[-6.0, 0.0, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
        #     'blue0': [[2.0, 0.0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        # }

        # self.poses = {
        #     'red0': [[-7.0, -3.5, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]],
        # }

        for robot in self.robots:
            # add_reference_to_stage(self._usd_path, f"{self._prim_path}/{robot}")

            super().__init__(
                prim_path=f"{self._prim_path}/{robot}",
                name=robot,
                wheel_dof_names=["wheel_joint_fr", "wheel_joint_fl", "wheel_joint_rr", "wheel_joint_rl"],
                create_robot=True,
                usd_path=self._usd_path,
                position=self.poses[robot][0],
                # translation=self.poses[robot][0],
                orientation=self.poses[robot][1],
            )

        if totnum_agents > 1:
            for blue in ['blue0']:
                ligth_path = f"{self._prim_path}/{blue}/light/visuals"
                material_path = f"{self._prim_path}/{blue}/Looks/material_blue"
                visual_material = UsdShade.Material(get_prim_at_path(material_path))
                binding_api = UsdShade.MaterialBindingAPI(get_prim_at_path(ligth_path))
                binding_api.Bind(visual_material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    def setup_arena(self):
        self._stage.GetRootLayer().subLayerPaths.append(os.getcwd() + "/assets/terrain/terrain.usd")