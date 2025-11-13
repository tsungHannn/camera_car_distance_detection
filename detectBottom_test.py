import os
import sys
import random
import math
import numpy as np
import argparse,configparser
import cv2
import json
import time
from tqdm import tqdm
from ultralytics import YOLO

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush']

# class_names = {2:'car', 3:'motorcycle', 5:'bus', 7:'truck'}

def resize_frame_and_vp(frame, vp, scale_factor):
    """
    調整影像和消失點的解析度
    
    Args:
        frame: 原始影像
        vp: 消失點列表 [[x1,y1], [x2,y2], [x3,y3]]
        scale_factor: 縮放因子 (例如 0.5 代表縮小為原來的一半)
    
    Returns:
        resized_frame: 調整後的影像
        resized_vp: 調整後的消失點
    """
    # 調整影像大小
    height, width = frame.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # 調整消失點座標
    resized_vp = [[int(x * scale_factor), int(y * scale_factor)] for x, y in vp]
    
    return resized_frame, resized_vp

def scale_back_results(boxes, masks, scale_factor, original_shape):
    """
    將偵測結果從縮放後的解析度還原到原始解析度
    
    Args:
        boxes: 邊界框 [N, (y1, x1, y2, x2)]
        masks: 遮罩 [height, width, N]
        scale_factor: 縮放因子
        original_shape: 原始影像尺寸 (height, width)
    
    Returns:
        scaled_boxes: 還原後的邊界框
        scaled_masks: 還原後的遮罩
    """
    # 還原邊界框座標
    scaled_boxes = boxes / scale_factor
    scaled_boxes = scaled_boxes.astype(np.int32)
    
    # 還原遮罩大小
    if masks.shape[2] > 0:
        scaled_masks = np.zeros((original_shape[0], original_shape[1], masks.shape[2]), dtype=bool)
        for i in range(masks.shape[2]):
            scaled_masks[:, :, i] = cv2.resize(
                masks[:, :, i].astype(np.uint8), 
                (original_shape[1], original_shape[0]), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
    else:
        scaled_masks = masks
    
    return scaled_boxes, scaled_masks

def apply_mask(image, mask, color, alpha=0.5):
    """在影像上套用半透明遮罩"""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def draw_2d_detections(image, boxes, masks, class_ids, scores, threshold=0.7):
    """在影像上繪製 2D 偵測結果"""
    # 顏色列表 (BGR 格式)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    N = boxes.shape[0]
    if not N:
        return image
    
    # 創建一個副本來繪製
    result = image.copy()
    
    for i in range(N):
        # 只顯示信心分數大於閾值的偵測
        if scores[i] < threshold:
            continue
            
        class_id = class_ids[i]
        if class_id not in [3,4,6,8]:
            continue
        color = colors[i % len(colors)]
        
        # 繪製遮罩
        mask = masks[:, :, i]
        result = apply_mask(result, mask, color, alpha=0.3)
        
        # 繪製邊界框
        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # 繪製標籤
        label = f"{class_names[class_id]}: {scores[i]:.2f}"
        
        # 計算文字大小和背景
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # 繪製標籤背景
        cv2.rectangle(result, 
                     (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width, y1), 
                     color, -1)
        
        # 繪製文字
        cv2.putText(result, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result

def draw_2d_masks(image, masks):
    """在影像上繪製所有遮罩"""
    # 顏色列表 (BGR 格式)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    N = masks.shape[2]
    if not N:
        return image
    
    # 創建一個副本來繪製
    result = image.copy()
    
    for i in range(N):
        color = colors[i % len(colors)]
        mask = masks[:, :, i]
        result = apply_mask(result, mask, color, alpha=0.3)
    
    return result

def Cal3dBBox(boxes, masks, class_ids, scores, vp):
    """
    優化版本：使用邊界提取和單次遍歷替代排序
    計算 3D 邊界框的底部四個角點
    """
    N = boxes.shape[0]
    ret = []
    if not N:
        return ret
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    def lineIntersection(a, b, c, d):
        """計算兩條直線的交點"""
        a, b, c, d = np.array(a), np.array(b), np.array(c), np.array(d)
        denominator = np.cross(b-a, d-c)
        if abs(denominator) < 1e-6:
            return False
        x = a + (b-a) * (np.cross(c-a, d-c) / denominator)
        return x

    def find_extreme_angles(vectors, reverse_x=False):
        """
        找到角度最極端的兩個點（單次遍歷，不需排序）
        
        Args:
            vectors: 向量列表 [[dy, dx], ...]
            reverse_x: 是否反轉 x 軸方向
        
        Returns:
            min_point, max_point: 角度最小和最大的兩個點
        """
        if len(vectors) == 0:
            return np.array([0, 0]), np.array([0, 0])
        
        min_point = np.array(vectors[0])
        max_point = np.array(vectors[0])
        
        if reverse_x:
            min_angle = math.atan2(min_point[1], -min_point[0])
            max_angle = min_angle
            
            for vec in vectors[1:]:
                angle = math.atan2(vec[1], -vec[0])
                if angle < min_angle:
                    min_angle = angle
                    min_point = np.array(vec)
                if angle > max_angle:
                    max_angle = angle
                    max_point = np.array(vec)
        else:
            min_angle = math.atan2(min_point[1], min_point[0])
            max_angle = min_angle
            
            for vec in vectors[1:]:
                angle = math.atan2(vec[1], vec[0])
                if angle < min_angle:
                    min_angle = angle
                    min_point = np.array(vec)
                if angle > max_angle:
                    max_angle = angle
                    max_point = np.array(vec)
        
        return min_point, max_point

    for i in range(N):
        class_id = class_ids[i]
        if class_id not in [3, 4, 6, 8]:
            continue
        
        if not np.any(boxes[i]):
            continue
        
        now = dict()
        now['box'] = boxes[i]
        now['class_id'] = class_id
        now['class_name'] = class_names[class_id]
        now['score'] = scores[i]
        y1, x1, y2, x2 = boxes[i]
        
        # 優化 1: 只提取邊界像素
        mask_region = masks[y1:y2, x1:x2, i]
        kernel = np.ones((3, 3), np.uint8)
        mask_eroded = cv2.erode(mask_region.astype(np.uint8), kernel, iterations=1)
        boundary = mask_region.astype(np.uint8) - mask_eroded
        boundary_ys, boundary_xs = np.where(boundary)
        
        # 如果邊界點太少，跳過
        if len(boundary_xs) < 4:
            continue
        
        # 將邊界點轉換為絕對座標
        boundary_points = [[x1 + x, y1 + y] for x, y in zip(boundary_xs, boundary_ys)]
        
        # 如果邊界點太多，可以下採樣以進一步加速
        if len(boundary_points) > 1000:
            step = len(boundary_points) // 500
            boundary_points = boundary_points[::step]
        
        # 對每個消失點計算向量
        maskvec = [[[y-v[1], x-v[0]] for x, y in boundary_points] for v in vp]
        
        # 優化 2: 使用單次遍歷找極值點（不需排序）
        vp = np.array(vp)
        edg = []
        
        # 處理前兩個消失點
        for j in range(2):
            min_pt, max_pt = find_extreme_angles(maskvec[j], reverse_x=False)
            
            # 根據角度絕對值決定順序
            angle_min = abs(math.atan2(min_pt[1], min_pt[0]))
            angle_max = abs(math.atan2(max_pt[1], max_pt[0]))
            
            if angle_min < angle_max:
                edg.append([min_pt[::-1], max_pt[::-1]])
            else:
                edg.append([max_pt[::-1], min_pt[::-1]])
        
        # 處理第三個消失點（X軸反向）
        min_pt, max_pt = find_extreme_angles(maskvec[2], reverse_x=True)
        
        angle_min = abs(math.atan2(min_pt[1], -min_pt[0]))
        angle_max = abs(math.atan2(max_pt[1], -max_pt[0]))
        
        if angle_min < angle_max:
            edg.append([min_pt[::-1], max_pt[::-1]])
        else:
            edg.append([max_pt[::-1], min_pt[::-1]])

        # 根據不同情況計算 3D 底部的四個交點
        if edg[0][0][0] * edg[0][-1][0] < 0:
            cross1 = lineIntersection(vp[1], vp[1]+edg[1][0], vp[2], vp[2]+edg[2][0])
            cross2 = lineIntersection(vp[1], vp[1]+edg[1][0], vp[2], vp[2]+edg[2][1])
            if cross1[0] > cross2[0]:
                cross1, cross2 = cross2, cross1
            cross5 = lineIntersection(vp[0], vp[0]+edg[0][0], vp[1], vp[1]+edg[1][1])
            cross6 = lineIntersection(vp[0], vp[0]+edg[0][1], vp[1], vp[1]+edg[1][1])
            if cross5[0] > cross6[0]:
                cross5, cross6 = cross6, cross5
            cross3 = lineIntersection(vp[0], cross1, vp[2], cross5)
            cross4 = lineIntersection(vp[0], cross2, vp[2], cross6)
        elif edg[1][0][0] * edg[1][-1][0] < 0:
            cross1 = lineIntersection(vp[0], vp[0]+edg[0][0], vp[2], vp[2]+edg[2][0])
            cross2 = lineIntersection(vp[0], vp[0]+edg[0][0], vp[2], vp[2]+edg[2][1])
            if cross1[0] > cross2[0]:
                cross1, cross2 = cross2, cross1
            cross5 = lineIntersection(vp[1], vp[1]+edg[1][0], vp[0], vp[0]+edg[0][1])
            cross6 = lineIntersection(vp[1], vp[1]+edg[1][1], vp[0], vp[0]+edg[0][1])
            if cross5[0] > cross6[0]:
                cross5, cross6 = cross6, cross5
            cross3 = lineIntersection(vp[1], cross1, vp[2], cross5)
            cross4 = lineIntersection(vp[1], cross2, vp[2], cross6)
        else:
            cross1 = lineIntersection(vp[0], vp[0]+edg[0][0], vp[1], vp[1]+edg[1][0])
            tmp1 = lineIntersection(vp[0], vp[0]+edg[0][0], vp[2], vp[2]+edg[2][0])
            tmp2 = lineIntersection(vp[1], vp[1]+edg[1][0], vp[2], vp[2]+edg[2][0])
            cross2 = tmp1 if tmp1[1] < tmp2[1] else tmp2
            tmp1 = lineIntersection(vp[0], vp[0]+edg[0][0], vp[2], vp[2]+edg[2][1])
            tmp2 = lineIntersection(vp[1], vp[1]+edg[1][0], vp[2], vp[2]+edg[2][1])
            cross3 = tmp1 if tmp1[1] < tmp2[1] else tmp2

            if type(lineIntersection(vp[0], cross1, vp[0], cross2)) == bool:
                cross4 = lineIntersection(vp[0], cross3, vp[1], cross2)
            else:
                cross4 = lineIntersection(vp[0], cross2, vp[1], cross3)

        assert type(cross1) == np.ndarray and type(cross2) == np.ndarray and type(cross3) == np.ndarray and type(cross4) == np.ndarray
        now['bottom'] = np.array([cross1, cross2, cross3, cross4]).reshape([-1, 2])
        ret.append(now)

    return ret


def yolo_to_maskrcnn_format(yolo_results, img_shape):
    """
    將 YOLO 的輸出格式轉換為 Mask R-CNN 格式
    
    Args:
        yolo_results: YOLO 模型的偵測結果
        img_shape: 影像尺寸 (height, width, channels)
    
    Returns:
        dict: 包含 'rois', 'masks', 'class_ids', 'scores' 的字典
    """
    if len(yolo_results) == 0 or yolo_results[0].boxes is None:
        return {
            'rois': np.array([]),
            'masks': np.zeros((img_shape[0], img_shape[1], 0), dtype=bool),
            'class_ids': np.array([]),
            'scores': np.array([])
        }
    
    result = yolo_results[0]
    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    scores = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)

    
    
    # 轉換 boxes 格式從 [x1, y1, x2, y2] 到 [y1, x1, y2, x2]
    rois = np.zeros_like(boxes)
    rois[:, 0] = boxes[:, 1]  # y1
    rois[:, 1] = boxes[:, 0]  # x1
    rois[:, 2] = boxes[:, 3]  # y2
    rois[:, 3] = boxes[:, 2]  # x2
    rois = rois.astype(np.int32)
    
    # 處理遮罩
    if hasattr(result, 'masks') and result.masks is not None:
        masks_data = result.masks.data.cpu().numpy()  # [N, H, W]
        N = masks_data.shape[0]
        H, W = img_shape[:2]
        
        # 將遮罩轉換為 [H, W, N] 格式
        masks = np.zeros((H, W, N), dtype=bool)
        for i in range(N):
            # YOLO 的遮罩可能需要調整大小
            mask = masks_data[i]
            if mask.shape != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            masks[:, :, i] = mask > 0.5
    else:
        # 如果沒有遮罩，創建基於邊界框的簡單遮罩
        N = len(boxes)
        masks = np.zeros((img_shape[0], img_shape[1], N), dtype=bool)
        for i in range(N):
            y1, x1, y2, x2 = rois[i]
            masks[y1:y2, x1:x2, i] = True
    
    return {
        'rois': rois,
        'masks': masks,
        'class_ids': class_ids,
        'scores': scores
    }



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', type=str, help='Folder which include output.avi and config',
                        default='tunnel_data_south16_test')
    parser.add_argument('--yolo_model',help='YOLO model path.',
                        default='yolo11m-seg.pt')
    parser.add_argument('--rcnn_model',help='Mask R-CNN model path.',
                        default='mask_rcnn_coco.h5')
    parser.add_argument("--save-vid",help='Draw the detected bottoms on the video and save it to another video file', action='store_true',
                        default=True)
    parser.add_argument("--save-2d-vid",help='Draw 2D detection results on the video and save it', action='store_true',
                        default=True)
    parser.add_argument("--scale",help='Scale factor for input resolution (e.g., 0.5 for half size, 1.0 for original)', 
                       type=float, default=1.0)
    args = parser.parse_args()

    ####### parameters ##########
    DATA_PATH=args.dataPath
    COCO_MODEL_PATH=args.rcnn_model
    YOLO_MODEL_PATH=args.yolo_model
    STORE_VID_FLAG=args.save_vid
    SAVE_2D_VID_FLAG=args.save_2d_vid
    SCALE_FACTOR=args.scale


    print(f"使用縮放因子: {SCALE_FACTOR}")
    if SCALE_FACTOR != 1.0:
        print(f"注意: 影像將被縮放至原始大小的 {SCALE_FACTOR*100:.1f}%")

    ########## Load video and config ###########
    if not os.path.exists('{}/output.mp4'.format(DATA_PATH)):
        print('{}/output.mp4 not exist!'.format(DATA_PATH))
        exit(0)
    if not os.path.exists('{}/config'.format(DATA_PATH)):
        print('{}/config not exist!'.format(DATA_PATH))
        exit(0)

    cap=cv2.VideoCapture('{}/output.mp4'.format(DATA_PATH))
    if not cap.isOpened():
        print('Open {}/output.mp4 failed!'.format(DATA_PATH))
        exit(0)

    config=configparser.ConfigParser()
    config.read('{}/config'.format(DATA_PATH))

    # Load vanishing points (原始解析度)
    vp_original=[json.loads(config.get('vps', 'vp{}'.format(i))) for i in range(1,4)]
    print(f"原始消失點: {vp_original}")

    # Create model object in inference mode.
    from mrcnn.config import Config
    class InferenceConfig(Config):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        NAME = 'coco'
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 80

    configCOCO = InferenceConfig()
    configCOCO.display()

    # 原本的 Mask R-CNN 實作
    # Create model object in inference mode.
    from mrcnn import utils
    import mrcnn.model as modellib
    model = modellib.MaskRCNN(mode="inference", model_dir='logs', config=configCOCO)

    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)


    # # YOLO
    # model = YOLO(YOLO_MODEL_PATH)
    # print("模型載入完成")

    # Solve and save
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"原始影片解析度: {width}x{height}")
    if SCALE_FACTOR != 1.0:
        print(f"處理解析度: {int(width*SCALE_FACTOR)}x{int(height*SCALE_FACTOR)}")

    out_3d = None
    out_2d = None
    idx = 0
    rets = []
    
    # 計時統計
    total_cal3d_time = 0
    frame_count = 0
    tqdm_bar = tqdm(total=total_frames, desc='Processing frames')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 儲存原始影像 (BGR)
        frame_original = frame.copy()
        original_shape = frame.shape[:2]  # (height, width)
        
        # 調整影像和消失點解析度
        if SCALE_FACTOR != 1.0:
            frame_resized, vp_resized = resize_frame_and_vp(frame, vp_original, SCALE_FACTOR)
        else:
            frame_resized = frame
            vp_resized = vp_original
        
        time1 = time.time()

        # 轉換為 RGB 進行偵測
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = model.detect([frame_rgb], verbose=0) # coco detection
        r = results[0]

        # yolo_results = model(frame_resized, verbose=False, classes=[2,3,5,7])  # 只偵測 car, motorcycle, bus, truck
        # r = yolo_to_maskrcnn_format(yolo_results, frame_resized.shape)
        

        time2 = time.time()

        # 計算 3D 底部 (使用縮放後的解析度和消失點來加速運算)
        start_time = time.time()
        ret_3d = Cal3dBBox(r['rois'], r['masks'], r['class_ids'], r['scores'], vp_resized)
        cal3d_time = time.time() - start_time
        total_cal3d_time += cal3d_time
        frame_count += 1
        
        time3 = time.time()

        # 將 3D 底部座標還原到原始解析度
        ret_3d_scaled = []
        for item in ret_3d:
            item_scaled = item.copy()
            if SCALE_FACTOR != 1.0:
                # 還原 box 座標
                item_scaled['box'] = (item['box'] / SCALE_FACTOR).astype(np.int32)
                # 還原 bottom 座標
                item_scaled['bottom'] = item['bottom'] / SCALE_FACTOR
            ret_3d_scaled.append(item_scaled)
        
        rets.append(ret_3d_scaled)

        time4 = time.time()
        
        # 如果有縮放,將偵測結果還原到原始解析度用於繪圖
        if SCALE_FACTOR != 1.0:
            boxes_scaled, masks_scaled = scale_back_results(
                r['rois'], r['masks'], SCALE_FACTOR, original_shape
            )
        else:
            boxes_scaled = r['rois']
            masks_scaled = r['masks']

        # 繪製 2D 偵測結果 (使用原始解析度)
        if SAVE_2D_VID_FLAG:
            # frame_2d = draw_2d_detections(frame_original.copy(), boxes_scaled, masks_scaled, 
            #                              r['class_ids'], r['scores'])
            frame_2d = draw_2d_masks(frame_original.copy(), masks_scaled)
            if not out_2d:
                out_2d = cv2.VideoWriter(os.path.join(DATA_PATH, 'output-2d-detection.avi'), 
                                        fourcc, fps, (width, height))
            out_2d.write(frame_2d)

        # 繪製 3D 底部 (使用原始解析度)
        if STORE_VID_FLAG:
            frame_3d = frame_original.copy()
            for item in ret_3d_scaled:
                if np.any(item['bottom'] < 0):
                    continue
                for i in range(4):
                    for j in range(4):
                        if i == j:
                            continue
                        cv2.line(frame_3d, tuple(item['bottom'][i].astype(int)), 
                               tuple(item['bottom'][j].astype(int)), (255, 255, 255), 2)
            
            if not out_3d:
                out_3d = cv2.VideoWriter(os.path.join(DATA_PATH, 'output-bottom.avi'), 
                                        fourcc, fps, (width, height))
            out_3d.write(frame_3d)
        
        time5 = time.time()
        # print('time1 (detection): {:.4f}s, time2 (3D bottom): {:.4f}s, time3 (scaling & drawing): {:.4f}s, time4: {:.4f}s'.format(
        #     time2 - time1, time3 - time2, time4 - time3, time5 - time4))
        # print(f'frame {idx} finished. Cal3dBBox time: {cal3d_time:.4f}s')
        idx += 1
        tqdm_bar.update(1)

    
    # 輸出效能統計
    if frame_count > 0:
        avg_cal3d_time = total_cal3d_time / frame_count
        print(f'\n效能統計:')
        print(f'處理幀數: {frame_count}')
        print(f'Cal3dBBox 平均時間: {avg_cal3d_time:.4f}s/frame')
        print(f'Cal3dBBox 總時間: {total_cal3d_time:.2f}s')
    
    print('savePath=', DATA_PATH)
    
    # 儲存 3D 底部結果
    rets = np.array(rets, dtype=object)
    np.save(os.path.join(DATA_PATH, 'output-bottom.npy'), rets)
    
    # 釋放影片寫入器
    if out_3d:
        out_3d.release()
        print('3D bottom video saved to: {}'.format(os.path.join(DATA_PATH, 'output-bottom.avi')))
    
    if out_2d:
        out_2d.release()
        print('2D detection video saved to: {}'.format(os.path.join(DATA_PATH, 'output-2d-detection.avi')))
    
    cap.release()