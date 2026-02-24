import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import time
import tf

# 定义全局变量
x1 = None
y1 = None
z1 = None
r1 = None
p1 = None
yaw1 = None
lidar_depth1 = None
x2 = None
y2 = None
z2 = None
r2 = None
p2 = None
yaw2 = None
lidar_depth2 = None
x3 = None
y3 = None
z3 = None
r3 = None
p3 = None
yaw3 = None
lidar_depth3 = None
x4 = None
y4 = None
z4 = None
r4 = None
p4 = None
yaw4 = None
t = None

def pose_callback_A1(msg):
    global x1,y1,z1,r1,p1,yaw1
    if msg is not None:
        # 从消息中提取四元数
        orientation_q = msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        # 使用 tf 库将四元数转换为欧拉角
        (r1, p1, yaw1) = tf.transformations.euler_from_quaternion(orientation_list)
        #print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
        x1 = msg.pose.position.x - 3.42
        y1 = msg.pose.position.y + 0.0
        z1 = msg.pose.position.z + 0.0
    
    
def pose_callback_A2(msg):
    global x2,y2,z2,r2,p2,yaw2
    if msg is not None:
        # 从消息中提取四元数
        orientation_q = msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        # 使用 tf 库将四元数转换为欧拉角
        (r2, p2, yaw2) = tf.transformations.euler_from_quaternion(orientation_list)
        #print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
        x2 = msg.pose.position.x - 3.42
        y2 = msg.pose.position.y + 0.0
        z2 = msg.pose.position.z + 0.0
    
def pose_callback_A3(msg):
    global x3,y3,z3,r3,p3,yaw3
    if msg is not None:
        # 从消息中提取四元数
        orientation_q = msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        # 使用 tf 库将四元数转换为欧拉角
        (r3, p3, yaw3) = tf.transformations.euler_from_quaternion(orientation_list)
        #print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
        x3 = msg.pose.position.x - 3.42
        y3 = msg.pose.position.y + 0.0
        z3 = msg.pose.position.z + 0.0

def pose_callback_B1(msg):
    global x4,y4,z4,r4,p4,yaw4
    if msg is not None:
        # 从消息中提取四元数
        orientation_q = msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        # 使用 tf 库将四元数转换为欧拉角
        (r4, p4, yaw4) = tf.transformations.euler_from_quaternion(orientation_list)
        #print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
        x4 = msg.pose.position.x - 3.42
        y4 = msg.pose.position.y + 0.0
        z4 = msg.pose.position.z + 0.0
    

def publish_agent_poses_init():
    # 初始化 ROS 节点
    rospy.init_node('a1_multi_agent_pose_publisher', anonymous=True)

    # 创建发布者
    pub0 = rospy.Publisher('A1/move_base_simple/goal', PoseStamped, queue_size=10)
    pub1 = rospy.Publisher('A2/move_base_simple/goal', PoseStamped, queue_size=10)
    pub2 = rospy.Publisher('A3/move_base_simple/goal', PoseStamped, queue_size=10)

    return pub0,pub1,pub2

def main():
    pub0,pub1,pub2=publish_agent_poses_init()
    rospy.Subscriber('/A1_vrpn_client_node/A1/pose', PoseStamped, pose_callback_A1)
    rospy.Subscriber('/A1_vrpn_client_node/A2/pose', PoseStamped, pose_callback_A2)
    rospy.Subscriber('/A1_vrpn_client_node/A3/pose', PoseStamped, pose_callback_A3)
    rospy.Subscriber('/A1_vrpn_client_node/B1/pose', PoseStamped, pose_callback_B1)
