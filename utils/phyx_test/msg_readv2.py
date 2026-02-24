import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import time
import tf
import torch
import time
from stable_baselines3 import PPO, SAC, TD3

# 定义全局变量
x1, y1, z1, r1, p1, yaw1 = None, None, None, None, None, None
x2, y2, z2, r2, p2, yaw2 = None, None, None, None, None, None
x3, y3, z3, r3, p3, yaw3 = None, None, None, None, None, None
x4, y4, z4, r4, p4, yaw4 = None, None, None, None, None, None
t = 0


def pose_callback_A1(msg):
    global x1, y1, z1, r1, p1, yaw1
    if msg is not None:
        # 从消息中提取四元数
        orientation_q = msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        # 使用 tf 库将四元数转换为欧拉角
        r1, p1, yaw1 = tf.transformations.euler_from_quaternion(orientation_list)
        # 更新全局变量
        x1, y1, z1 = msg.pose.position.x - 3.42, msg.pose.position.y, msg.pose.position.z

def pose_callback_A2(msg):
    global x2, y2, z2, r2, p2, yaw2
    if msg is not None:
        # 从消息中提取四元数
        orientation_q = msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        # 使用 tf 库将四元数转换为欧拉角
        r2, p2, yaw2 = tf.transformations.euler_from_quaternion(orientation_list)
        # 更新全局变量
        x2, y2, z2 = msg.pose.position.x - 3.42, msg.pose.position.y, msg.pose.position.z

def pose_callback_A3(msg):
    global x3, y3, z3, r3, p3, yaw3
    if msg is not None:
        # 从消息中提取四元数
        orientation_q = msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        # 使用 tf 库将四元数转换为欧拉角
        r3, p3, yaw3 = tf.transformations.euler_from_quaternion(orientation_list)
        # 更新全局变量
        x3, y3, z3 = msg.pose.position.x - 3.42, msg.pose.position.y, msg.pose.position.z

def pose_callback_B1(msg):
    global x4, y4, z4, r4, p4, yaw4
    if msg is not None:
        # 从消息中提取四元数
        orientation_q = msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        # 使用 tf 库将四元数转换为欧拉角
        r4, p4, yaw4 = tf.transformations.euler_from_quaternion(orientation_list)
        # 更新全局变量
        x4, y4, z4 = msg.pose.position.x - 3.42, msg.pose.position.y, msg.pose.position.z

def publish_agent_poses_init():
    # 初始化 ROS 节点
    rospy.init_node('a1_multi_agent_goal_publisher', anonymous=True)

    # 创建发布者
    pub0 = rospy.Publisher('A1/move_base_simple/goal', PoseStamped, queue_size=10)
    pub1 = rospy.Publisher('A2/move_base_simple/goal', PoseStamped, queue_size=10)
    pub2 = rospy.Publisher('A3/move_base_simple/goal', PoseStamped, queue_size=10)

    return pub0, pub1, pub2

def get_observation():
    global t
    t = t+1
    return [x1,y1,z1,r1,p1,yaw1,2,2,2,2,2,2,2,2,2,2,2,2,x2,y2,z2,r2,p2,yaw2,2,2,2,2,2,2,2,2,2,2,2,2,x3,y3,z3,r3,p3,yaw3,2,2,2,2,2,2,2,2,2,2,2,2,x4,y4,z4,r4,p4,yaw4,t]

def main():
    pub0, pub1, pub2 = publish_agent_poses_init()
    rospy.Subscriber('/A1_vrpn_client_node/A1/pose', PoseStamped, pose_callback_A1)
    rospy.Subscriber('/A1_vrpn_client_node/A2/pose', PoseStamped, pose_callback_A2)
    rospy.Subscriber('/A1_vrpn_client_node/A3/pose', PoseStamped, pose_callback_A3)
    rospy.Subscriber('/A1_vrpn_client_node/B1/pose', PoseStamped, pose_callback_B1)

    rate = rospy.Rate(10)  # 10Hz

    model = SAC.load("checkpoints/SAC")
    
    #max_steps = 1000

    time.sleep(2)
    
    while not rospy.is_shutdown():
        observation = get_observation()
        #print(type(observation))
        #print((observation))
        action, _ = model.predict(observation, deterministic=True)
        print(observation)
        print("#######")
        print("#######")
        print("#######")
        print(action)
        # 这里可以添加发布消息的逻辑
        rate.sleep()

if __name__ == '__main__':
    main()
