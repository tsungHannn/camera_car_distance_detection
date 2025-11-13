import cv2
import numpy as np
from tqdm import tqdm

# 1. 使用 Shi-Tomasi 角点检测算法检测视频中的特征点
def detect_feature_points(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    feature_points = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    if feature_points is not None:
        feature_points = np.float32(feature_points)  # 转换为float32
    return feature_points

# 计算图像对角线长度
def calculate_rho_max(image_shape):
    height, width = image_shape[:2]
    return int(np.sqrt(height**2 + width**2))

# 將霍夫變換的 rho, theta 轉換為直線的參數形式
def polar_to_cartesian(rho, theta):
    """
    將極座標形式的直線轉換為直角座標形式
    返回: (a, b, is_vertical) 或 None
    - 若非垂直線: y = ax + b, is_vertical = False
    - 若垂直線: x = c, 返回 (None, c, True)
    """
    # 處理垂直線的情況 (theta 接近 0 或 π)
    sin_theta = np.sin(theta)
    
    if abs(sin_theta) < 1e-5:  # 垂直線
        # 對於垂直線: rho = x * cos(theta)
        x = rho / np.cos(theta)
        return None, x, True
    
    # 一般情況: rho = x * cos(theta) + y * sin(theta)
    # 轉換為 y = ax + b
    a = -np.cos(theta) / sin_theta  # 斜率
    b = rho / sin_theta              # 截距
    return a, b, False

# 計算兩條直線的交點
def calculate_intersection(line1, line2):
    """
    計算兩條直線的交點
    line1, line2: (a, b, is_vertical) 元組
    """
    a1, b1, is_vertical1 = line1
    a2, b2, is_vertical2 = line2
    
    # 兩條都是垂直線，無交點
    if is_vertical1 and is_vertical2:
        return None
    
    # line1 是垂直線
    if is_vertical1:
        x = b1
        y = a2 * x + b2
        return int(x), int(y)
    
    # line2 是垂直線
    if is_vertical2:
        x = b2
        y = a1 * x + b1
        return int(x), int(y)
    
    # 兩條都不是垂直線
    if abs(a1 - a2) < 1e-5:  # 平行線，無交點
        return None
    
    # 交點的 x 坐標
    x = (b2 - b1) / (a1 - a2)
    # 交點的 y 坐標
    y = a1 * x + b1
    
    return int(x), int(y)


# 3. 修改 compute_vanishing_points 函數
def compute_vanishing_points(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    
    if lines is not None:
        vanishing_points = []
        
        print(f"偵測到 {len(lines)} 條直線，開始計算交點...")
        
        # 將霍夫空間中的直線轉換為笛卡爾坐標系中的直線，並尋找交點
        for i in tqdm(range(len(lines)), desc="計算交點進度"):
            for j in range(i + 1, len(lines)):
                rho1, theta1 = lines[i][0]
                rho2, theta2 = lines[j][0]
                
                # 將極坐標形式轉換為直角坐標形式
                line1 = polar_to_cartesian(rho1, theta1)
                line2 = polar_to_cartesian(rho2, theta2)
                
                # 計算交點
                intersection = calculate_intersection(line1, line2)
                
                if intersection:
                    x, y = intersection
                    # 可以選擇性地過濾掉太遠的消失點
                    if -5000 < x < 5000 and -5000 < y < 5000:
                        vanishing_points.append(intersection)
        
        print(f"計算完成，找到 {len(vanishing_points)} 個消失點")
        return vanishing_points
    return None


# 4. 主函数，只處理第一幀
def process_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if not ret:
        print("讀取視頻錯誤")
        cap.release()
        return
    
    print("成功讀取第一幀")
    
    # 檢測特征點
    feature_points = detect_feature_points(frame)
    if feature_points is not None:
        print(f"偵測到 {len(feature_points)} 個特征點")
        # 繪製特征點
        for point in feature_points:
            x, y = point.ravel()
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    # 計算並顯示消失點
    vanishing_points = compute_vanishing_points(frame)
    if vanishing_points:
        print(f"消失點座標: {vanishing_points}")
        for vp in vanishing_points:
            cv2.circle(frame, vp, 5, (0, 0, 255), -1)  # 在消失點處畫紅點
            # 標註消失點座標
            cv2.putText(frame, f"{vp}", (vp[0]+10, vp[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        print("未找到消失點")
    
    # 顯示結果
    cv2.imshow("First Frame - Vanishing Point Detection", frame)
    print("按任意鍵關閉視窗...")
    cv2.waitKey(0)
    
    # 儲存結果
    output_path = '/media/mvclab/SSD/workspace/ncsist/FusionCalib/data/first_frame_result.jpg'
    cv2.imwrite(output_path, frame)
    print(f"結果已儲存至: {output_path}")
    
    cap.release()
    cv2.destroyAllWindows()

# 调用函数
process_first_frame('/media/mvclab/SSD/workspace/ncsist/FusionCalib/data/output.avi')