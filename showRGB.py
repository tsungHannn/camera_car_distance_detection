import cv2
import numpy as np

def undistort_video(input_video, output_video, camera_matrix, dist_coeffs):
    # 打开输入视频
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"Cannot open video {input_video}")
        return
    
    # 获取视频的宽度、高度和帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 获取视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # 打开输出视频写入器
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (frame_width, frame_height))
    
    # 计算校正映射
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (frame_width, frame_height), 1, (frame_width, frame_height))
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (frame_width, frame_height), 5)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 应用校正
        undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        
        # 显示原始帧和校正后的帧
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Undistorted Frame', undistorted_frame)
        
        # 写入校正后的帧
        out.write(undistorted_frame)
        
        # 按下 'q' 键退出显示
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_video = 'data/output.avi'  # 输入的AVI视频文件
    output_video = 'undistorted_output.avi'  # 输出的校正后的AVI视频文件
    
    # 相机内参矩阵（示例）
    camera_matrix = np.array([[1.90753223e+03,0.00000000e+00,6.44360570e+02],
	[0.00000000e+00,1.98064319e+03,3.65685545e+02],
	[0.00000000e+00,0.00000000e+00,1.00000000e+00]])
    
    # 畸变系数（示例）
    dist_coeffs = np.array([0.0812306431723410,-0.100591860946000,0,0])
    
    undistort_video(input_video, output_video, camera_matrix, dist_coeffs)
