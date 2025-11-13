import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge
import os

def save_compressed_images_to_avi(bag_file, topic_name, output_file):
    bag = rosbag.Bag(bag_file, 'r')
    bridge = CvBridge()
    
    # 获取视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = None
    frame_width = None
    frame_height = None
    
    timestamps = []
    images = []
    
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        # 将CompressedImage消息转换为OpenCV图像
        cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        images.append(cv_image)
        timestamps.append(t.to_sec())
        
        if video_writer is None:
            # 初始化视频写入器
            frame_height, frame_width = cv_image.shape[:2]
    
    bag.close()
    
    if len(timestamps) < 2:
        print("Not enough images to create a video.")
        return
    
    # 计算帧率（根据时间戳差异计算平均帧率）
    time_intervals = np.diff(timestamps)
    avg_time_interval = np.mean(time_intervals)
    frame_rate = 1.0 / avg_time_interval
    
    # 创建视频写入器
    video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, (frame_width, frame_height))
    
    for image in images:
        video_writer.write(image)
    
    if video_writer is not None:
        video_writer.release()

if __name__ == '__main__':
    bag_file = '/HDD/ncsist/2024/data/1021/10.bag'  # 输入的ROS bag文件
    topic_name = '/aravis_cam/image_color/compressed'  # 压缩图像的topic
    output_file = 'output.avi'  # 输出的AVI文件

    save_compressed_images_to_avi(bag_file, topic_name, output_file)
