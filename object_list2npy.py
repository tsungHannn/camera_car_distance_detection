import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import math

def pointcloud2_to_array(cloud_msg):
   # 定義 structured array 的資料型別
    dtype = [('id', '<f4'), ('time', '<f4'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('vx', '<f4'), ('vy', '<f4'), ('vz', '<f4')]

    # 建立空的 structured array
    pc_array = np.empty(0, dtype=dtype)

    for obj in cloud_msg.objectlist_objects:
        # id, time, x, y, z, vx, vy, vz
        ros_time = rospy.Time(cloud_msg.timestamp_seconds, cloud_msg.timestamp_nanoseconds)
        rospy.init_node("testTime", anonymous=True)
        # now = rospy.get_rostime()
        vx = obj.f_dynamics_absvel_x
        vy = obj.f_dynamics_absvel_y
        if math.sqrt(vx*vx + vy*vy) < 1:
            continue

        # 高度全部設1
        new_data = np.array([(obj.u_id, ros_time.to_sec(), obj.u_position_x, obj.u_position_y, 1.0, obj.f_dynamics_absvel_x, obj.f_dynamics_absvel_y, 0.0)], dtype=dtype)

        pc_array = np.append(pc_array, new_data)


    # 使用点云数据构建numpy数组
    # pc_array = np.frombuffer(cloud_msg.data, dtype=np.dtype(dtype))
    return pc_array

def save_pointcloud_to_npy(bag_file, topic_name, output_file):
    bag = rosbag.Bag(bag_file, 'r')
    radarFrame = []
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        pc_array = pointcloud2_to_array(msg)
        

        radarFrame.append(pc_array)
    
    np.save(output_file, radarFrame)
    bag.close()

if __name__ == '__main__':
    bag_file = '/HDD/ncsist/2024/data/1021/10.bag'  # 输入的ROS bag文件
    topic_name = '/radar/object_list'  # 点云数据的topic
    output_file = 'radarTrack.npy'  # 输出的.npy文件

    save_pointcloud_to_npy(bag_file, topic_name, output_file)
