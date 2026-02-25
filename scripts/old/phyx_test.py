import torch
import time
from stable_baselines3 import PPO, SAC, TD3
from scipy.spatial.transform import Rotation as R

global x1,y1,z1,r1,p1,yaw1,lidar_depth1,x2,y2,z2,r2,p2,yaw2,lidar_depth2,x3,y3,z3,r3,p3,yaw3,lidar_depth3,x4,y4,z4,r4,p4,yaw4,t

x1 = y1 = z1 = r1 = p1 = yaw1 = lidar_depth1 = None
x2 = y2 = z2 = r2 = p2 = yaw2 = lidar_depth2 = None
x3 = y3 = z3 = r3 = p3 = yaw3 = lidar_depth3 = None
x4 = y4 = z4 = r4 = p4 = yaw4 = lidar_depth4 = None
t = None

# topic:订阅话题

# /A1_vrpn_client_node/A1/pose
# /A1_vrpn_client_node/A2/pose
# /A1_vrpn_client_node/A2/pose
# /A1_vrpn_client_node/B1/pose

# msg:消息类型

# geometry_msgs/PoseStamped

# 订阅解包——》坐标偏移——》四元数转欧拉——》强化学习输入

# rostopic

def quat_to_euler(quat):
    # 默认是 'xyz' 旋转顺序转换, list
    return R.from_quat(quat).as_euler('xyz', degrees=True)

def real2sim_transform(x_real, y_real, z_real):
    return [x_real - 3.424, y_real, z_real]

# 获取当前观测
def get_observation():
    ###x1_real,y1_real,z1_real  Q:4 -> ###x1_sim,y1_sim,z1_sim
    ###x1_sim,y1_sim,z1_sim

    x1_real = y1_real = z1_real = x2_real = y2_real = z2_real = x3_real = y3_real = z3_real = x4_real = y4_real = z4_real = 0
    quat1 = quat2 = quat3 = quat4 = [0, 0, 0]

    x1, y1, z1 = real2sim_transform(x1_real, y1_real, z1_real)
    x2, y2, z2 = real2sim_transform(x2_real, y2_real, z2_real)
    x3, y3, z3 = real2sim_transform(x3_real, y3_real, z3_real)
    x4, y4, z4 = real2sim_transform(x4_real, y4_real, z4_real)

    r1, p1, yaw1 = quat_to_euler(quat1)
    r2, p2, yaw2 = quat_to_euler(quat2)
    r3, p3, yaw3 = quat_to_euler(quat3)
    r4, p4, yaw4 = quat_to_euler(quat4)
    
    observation = [x1,y1,z1,r1,p1,yaw1,lidar_depth1,x2,y2,z2,r2,p2,yaw2,lidar_depth2,x3,y3,z3,r3,p3,yaw3,lidar_depth3,x4,y4,z4,r4,p4,yaw4,t]
    return observation

# 发送动作到实机执行
def send_action(action):
    pass

def main():
    model = TD3.load("checkpoints/TD3")
    max_steps = 1000

    # 主循环
    for step in range(max_steps):
        observation = get_observation()

        action, _ = model.predict(observation, deterministic=True)

        send_action(action)

        # 等待实机执行完成
        time.sleep(0.1)  


if __name__ == "__main__":
    main()
