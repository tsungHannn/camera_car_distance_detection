# fileName: single_image_detect.py
# 功能: 處理單張圖片進行物件分割、計算車輛底盤、測量前後車距離
import os
import sys
import math
import numpy as np
import argparse
import configparser
import cv2
import json
from ultralytics import YOLO

# === 匯入測距模組 ===
try:
    from Homography import DistanceManager
except ImportError:
    print("錯誤：找不到 Homography.py，請確保該檔案在同一目錄下。")
    sys.exit(1)

class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

COLOR_PALETTE = [
    (0, 255, 255), (0, 165, 255), (255, 0, 255), (50, 205, 50),
    (238, 130, 238), (127, 255, 0), (255, 140, 0), (0, 191, 255),
    (255, 20, 147), (138, 43, 226), (0, 250, 154), (255, 215, 0)
]

def apply_mask(image, mask, color, alpha=0.5):
    """應用遮罩到圖片上"""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def draw_2d_masks(image, masks):
    """繪製 2D 遮罩"""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    N = masks.shape[2]
    if not N: 
        return image
    result = image.copy()
    for i in range(N):
        color = colors[i % len(colors)]
        mask = masks[:, :, i]
        result = apply_mask(result, mask, color, alpha=0.3)
    return result

def Cal3dBBox(boxes, masks, class_ids, scores, vp):
    """計算 3D 邊界框"""
    N = boxes.shape[0]
    ret = []
    if not N: 
        return ret
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    def lineIntersection(a, b, c, d):
        a, b, c, d = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64), np.array(c, dtype=np.float64), np.array(d, dtype=np.float64)
        
        # 計算方向向量
        ba = b - a  # (b_x - a_x, b_y - a_y)
        dc = d - c  # (d_x - c_x, d_y - c_y)
        ca = c - a  # (c_x - a_x, c_y - a_y)
        
        # 2D 叉積: (u_x, u_y) × (v_x, v_y) = u_x * v_y - u_y * v_x
        denominator = ba[0] * dc[1] - ba[1] * dc[0]
        
        if abs(denominator) < 1e-6: 
            return False
        
        # 計算交點參數
        cross_ca_dc = ca[0] * dc[1] - ca[1] * dc[0]
        t = cross_ca_dc / denominator
        
        x = a + t * ba
        return x
    
    def is_valid_bottom_points(points, debug_info=""):
        """檢查底盤四個點是否有效 (無重複點、坐標有限)"""
        if len(points) != 4:
            print(f"  [跳過] {debug_info}: 點數不為4 ({len(points)} 個點)")
            return False
        # 檢查所有點坐標是否有限
        for idx, pt in enumerate(points):
            if not np.all(np.isfinite(pt)):
                print(f"  [跳過] {debug_info}: 點{idx}含有非有限值 {pt}")
                return False
        # 檢查是否有重複的點 (允許極小的誤差)
        for i in range(4):
            for j in range(i + 1, 4):
                dist = np.linalg.norm(points[i] - points[j])
                if dist < 1.0:  # 如果兩點距離小於1像素，視為重複
                    print(f"  [跳過] {debug_info}: 點{i}與點{j}距離過近 ({dist:.2f}px)")
                    print(f"         點坐標: [{points[0]}, {points[1]}, {points[2]}, {points[3]}]")
                    return False
        return True

    def find_extreme_angles(vectors, reverse_x=False):
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
                    min_angle, min_point = angle, np.array(vec)
                if angle > max_angle: 
                    max_angle, max_point = angle, np.array(vec)
        else:
            min_angle = math.atan2(min_point[1], min_point[0])
            max_angle = min_angle
            for vec in vectors[1:]:
                angle = math.atan2(vec[1], vec[0])
                if angle < min_angle: 
                    min_angle, min_point = angle, np.array(vec)
                if angle > max_angle: 
                    max_angle, max_point = angle, np.array(vec)
        return min_point, max_point

    for i in range(N):
        class_id = class_ids[i]
        if not np.any(boxes[i]): 
            continue
        now = dict()
        now['box'] = boxes[i]
        now['class_id'] = class_id
        now['class_name'] = class_names[class_id]
        now['score'] = scores[i]
        y1, x1, y2, x2 = boxes[i]
        mask_region = masks[y1:y2, x1:x2, i]
        kernel = np.ones((3, 3), np.uint8)
        mask_eroded = cv2.erode(mask_region.astype(np.uint8), kernel, iterations=1)
        boundary_ys, boundary_xs = np.where(mask_region.astype(np.uint8) - mask_eroded)
        if len(boundary_xs) < 4: 
            continue
        boundary_points = [[x1 + x, y1 + y] for x, y in zip(boundary_xs, boundary_ys)]
        if len(boundary_points) > 1000:
            step = len(boundary_points) // 500
            boundary_points = boundary_points[::step]
        maskvec = [[[y - v[1], x - v[0]] for x, y in boundary_points] for v in vp]
        vp_arr = np.array(vp)
        edg = []
        for j in range(2):
            min_pt, max_pt = find_extreme_angles(maskvec[j], reverse_x=False)
            angle_min = abs(math.atan2(min_pt[1], min_pt[0]))
            angle_max = abs(math.atan2(max_pt[1], max_pt[0]))
            if angle_min < angle_max: 
                edg.append([min_pt[::-1], max_pt[::-1]])
            else: 
                edg.append([max_pt[::-1], min_pt[::-1]])
        min_pt, max_pt = find_extreme_angles(maskvec[2], reverse_x=True)
        angle_min = abs(math.atan2(min_pt[1], -min_pt[0]))
        angle_max = abs(math.atan2(max_pt[1], -max_pt[0]))
        if angle_min < angle_max: 
            edg.append([min_pt[::-1], max_pt[::-1]])
        else: 
            edg.append([max_pt[::-1], min_pt[::-1]])

        if edg[0][0][0] * edg[0][-1][0] < 0:
            c1 = lineIntersection(vp_arr[1], vp_arr[1] + edg[1][0], vp_arr[2], vp_arr[2] + edg[2][0])
            c2 = lineIntersection(vp_arr[1], vp_arr[1] + edg[1][0], vp_arr[2], vp_arr[2] + edg[2][1])
            if c1 is False or c2 is False:
                continue
            if c1[0] > c2[0]: 
                c1, c2 = c2, c1
            c5 = lineIntersection(vp_arr[0], vp_arr[0] + edg[0][0], vp_arr[1], vp_arr[1] + edg[1][1])
            c6 = lineIntersection(vp_arr[0], vp_arr[0] + edg[0][1], vp_arr[1], vp_arr[1] + edg[1][1])
            if c5 is False or c6 is False:
                continue
            if c5[0] > c6[0]: 
                c5, c6 = c6, c5
            c3, c4 = lineIntersection(vp_arr[0], c1, vp_arr[2], c5), lineIntersection(vp_arr[0], c2, vp_arr[2], c6)
            if c3 is False or c4 is False:
                continue
            _branch = "Branch1 (edg[0] crosses x-axis)"
        elif edg[1][0][0] * edg[1][-1][0] < 0:
            c1 = lineIntersection(vp_arr[0], vp_arr[0] + edg[0][0], vp_arr[2], vp_arr[2] + edg[2][0])
            c2 = lineIntersection(vp_arr[0], vp_arr[0] + edg[0][0], vp_arr[2], vp_arr[2] + edg[2][1])
            if c1 is False or c2 is False:
                continue
            if c1[0] > c2[0]: 
                c1, c2 = c2, c1
            c5 = lineIntersection(vp_arr[1], vp_arr[1] + edg[1][0], vp_arr[0], vp_arr[0] + edg[0][1])
            c6 = lineIntersection(vp_arr[1], vp_arr[1] + edg[1][1], vp_arr[0], vp_arr[0] + edg[0][1])
            if c5 is False or c6 is False:
                continue
            if c5[0] > c6[0]: 
                c5, c6 = c6, c5
            c3, c4 = lineIntersection(vp_arr[1], c1, vp_arr[2], c5), lineIntersection(vp_arr[1], c2, vp_arr[2], c6)
            if c3 is False or c4 is False:
                continue
            _branch = "Branch2 (edg[1] crosses x-axis)"
        else:
            c1 = lineIntersection(vp_arr[0], vp_arr[0] + edg[0][0], vp_arr[1], vp_arr[1] + edg[1][0])
            if c1 is False:
                continue
            t1, t2 = lineIntersection(vp_arr[0], vp_arr[0] + edg[0][0], vp_arr[2], vp_arr[2] + edg[2][0]), lineIntersection(vp_arr[1], vp_arr[1] + edg[1][0], vp_arr[2], vp_arr[2] + edg[2][0])
            if t1 is False or t2 is False:
                continue
            c2 = t1 if t1[1] < t2[1] else t2
            t1, t2 = lineIntersection(vp_arr[0], vp_arr[0] + edg[0][0], vp_arr[2], vp_arr[2] + edg[2][1]), lineIntersection(vp_arr[1], vp_arr[1] + edg[1][0], vp_arr[2], vp_arr[2] + edg[2][1])
            if t1 is False or t2 is False:
                continue
            c3 = t1 if t1[1] < t2[1] else t2
            # c4 = lineIntersection(vp_arr[0], c2, vp_arr[1], c3) if isinstance(lineIntersection(vp_arr[0], c1, vp_arr[0], c2), bool) else lineIntersection(vp_arr[0], c3, vp_arr[1], c2)
            # if isinstance(lineIntersection(vp_arr[0], c1, vp_arr[0], c2), bool):
            #     print("HERE")
            # else:
            #     print("vp_arr[0]:", vp_arr[0], "c1:", c1, "c2:", c2)
            #     print(lineIntersection(vp_arr[0], c1, vp_arr[0], c2))
            # input("Press Enter to continue...")
            # # 修正: c4 应该是从 vp[0] 经过 c2，与从 vp[1] 经过 c3 的线的交点
            # c4 = lineIntersection(vp_arr[0], c2, vp_arr[1], c3)


            if c4 is False:
                continue
            _branch = "Branch3 (neither crosses x-axis)"
        
        # 將四個點轉換為 numpy 陣列並驗證
        bottom_pts = np.array([c1, c2, c3, c4]).reshape(-1, 2)
        
        # 驗證底盤點是否有效
        debug_label = f"{now['class_name']}(obj{i}) - {_branch}"
        if not is_valid_bottom_points(bottom_pts, debug_label):
            print(f"  詳細: c1={c1}, c2={c2}, c3={c3}, c4={c4}")
            continue
        
        now['bottom'] = bottom_pts
        ret.append(now)
    return ret

def yolo_to_maskrcnn_format(yolo_results, img_shape):
    """轉換 YOLO 結果為 MaskRCNN 格式"""
    if len(yolo_results) == 0 or yolo_results[0].boxes is None:
        return {'rois': np.array([]), 'masks': np.zeros((img_shape[0], img_shape[1], 0), dtype=bool), 'class_ids': np.array([]), 'scores': np.array([])}
    result = yolo_results[0]
    boxes, scores, class_ids = result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)
    rois = np.zeros_like(boxes)
    rois[:, 0], rois[:, 1], rois[:, 2], rois[:, 3] = boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]
    rois = rois.astype(np.int32)
    H, W = img_shape[:2]
    if hasattr(result, 'masks') and result.masks is not None:
        masks_data = result.masks.data.cpu().numpy()
        N = masks_data.shape[0]
        masks = np.zeros((H, W, N), dtype=bool)
        for i in range(N):
            mask = masks_data[i]
            if mask.shape != (H, W): 
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            masks[:, :, i] = mask > 0.5
    else:
        N = len(boxes)
        masks = np.zeros((img_shape[0], img_shape[1], N), dtype=bool)
        for i in range(N):
            y1, x1, y2, x2 = rois[i]
            masks[y1:y2, x1:x2, i] = True
    return {'rois': rois, 'masks': masks, 'class_ids': class_ids, 'scores': scores}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='單張圖片物件分割與距離測量')
    parser.add_argument('--image', type=str, default="data/test.png", help='輸入圖片路徑')
    parser.add_argument('--dataPath', type=str, default='data', help='配置資料目錄')
    parser.add_argument('--yolo_model', default='yolo11m-seg.pt', help='YOLO 模型路徑')
    parser.add_argument('--output', type=str, default="data/detected_test.png", help='輸出圖片路徑 (預設: ./images_detected_[輸入名稱])')
    parser.add_argument('--ref_dist', type=str, default='referenceDist.txt', help='參考距離檔案')
    args = parser.parse_args()

    # 檢查輸入圖片
    if not os.path.exists(args.image):
        print(f"錯誤：圖片檔案不存在: {args.image}")
        sys.exit(1)

    # 設定輸入輸出路徑
    DATA_PATH = args.dataPath
    ref_path = os.path.join(DATA_PATH, 'referenceDist.txt')
    lane_path = os.path.join(DATA_PATH, 'lanes8.txt')

    if not os.path.exists(ref_path) and os.path.exists(args.ref_dist):
        ref_path = args.ref_dist

    # 讀取圖片
    print(f"讀取圖片: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"錯誤：無法讀取圖片: {args.image}")
        sys.exit(1)

    image_original = image.copy()
    height, width = image.shape[:2]

    # 初始化距離管理器
    dist_manager = None
    if os.path.exists(ref_path):
        dist_manager = DistanceManager(ref_path, lane_path if os.path.exists(lane_path) else None)
        print(f"距離校正檔已載入: {ref_path}")
    else:
        print(f"警告: {ref_path} 檔案未找到，將跳過距離測量")

    # 讀取視點配置
    config_path = os.path.join(DATA_PATH, 'config')
    if not os.path.exists(config_path):
        print(f"警告：{config_path} 不存在，使用預設視點")
        vp_original = [[640, 360], [320, 360], [960, 360]]  # 預設視點
    else:
        config = configparser.ConfigParser()
        config.read(config_path)
        vp_original = [json.loads(config.get('vps', f'vp{i}')) for i in range(1, 4)]

    # 載入 YOLO 模型
    print("載入 YOLO 模型...")
    model = YOLO(args.yolo_model)
    print("模型載入完成")

    # === 物件偵測與分割 ===
    print("執行物件偵測...")
    yolo_results = model(image, verbose=False, classes=[2, 3, 5, 7])
    r = yolo_to_maskrcnn_format(yolo_results, image.shape)
    print(f"偵測到 {len(r['class_ids'])} 個物件")

    # === 計算 3D 邊界框 ===
    print("計算 3D 邊界框...")
    ret_3d = Cal3dBBox(r['rois'], r['masks'], r['class_ids'], r['scores'], vp_original)
    print(f"計算了 {len(ret_3d)} 個車輛底盤")
    
    # 繪製ret_3d底盤點的debug圖
    # print(ret_3d)
    debug_image = image.copy()
    for item in ret_3d:
        if np.any(item['bottom'] < 0): 
            continue
        pts = item['bottom']
        
        # 檢查底盤點是否有重複
        unique_pts = []
        for pt in pts:
            is_duplicate = False
            for upt in unique_pts:
                if np.linalg.norm(pt - upt) < 1.0:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_pts.append(pt)
        
        if len(unique_pts) < 4:
            print(f"警告: 物件 {item['class_name']} 有重複的底盤點! 原點數: 4, 獨特點數: {len(unique_pts)}")
            print(f"  底盤點: {pts}")
        
        for k in range(4):
            for l in range(k + 1, 4): 
                cv2.line(debug_image, tuple(pts[k].astype(int)), tuple(pts[l].astype(int)), (255, 255, 255), 2)
                # 寫出每個點的座標
                cv2.putText(debug_image, f"{pts[k][0]:.1f},{pts[k][1]:.1f}", tuple(pts[k].astype(int) + np.array([5, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    debug_output_path = os.path.splitext(args.output)[0] + '_debug.png'
    cv2.imwrite(debug_output_path, debug_image)

    # === 距離測量與分組 ===
    drawing_data = []
    lane_groups = {}
    lane_debug_info = []

    if dist_manager and dist_manager.H is not None:
        # (A) 分組階段
        for obj_idx, item in enumerate(ret_3d):
            if np.any(item['bottom'] < 0): 
                continue

            corners = item['bottom']
            center_y = np.mean(corners[:, 1])
            center_x = np.mean(corners[:, 0])
            item['center_y'] = center_y

            # 取得主要車道
            primary_lane = dist_manager.get_primary_lane(corners)

            # 記錄除錯資訊
            lane_debug_info.append({
                'obj_idx': obj_idx,
                'center': (center_x, center_y),
                'lane': primary_lane,
                'class_name': item.get('class_name', 'unknown')
            })

            if primary_lane:
                if primary_lane not in lane_groups: 
                    lane_groups[primary_lane] = []
                lane_groups[primary_lane].append(item)
            elif not dist_manager.lanes:
                # 沒讀到 lanes8.txt 的兼容模式
                if 'default' not in lane_groups: 
                    lane_groups['default'] = []
                lane_groups['default'].append(item)

        # (B) 計算距離
        for lane_name, objects in lane_groups.items():
            if len(objects) < 2: 
                continue

            # 依 Center Y 從小(遠)到大(近)排序
            objects.sort(key=lambda x: x['center_y'])

            for i in range(len(objects) - 1):
                lead = objects[i]       # 前車
                follow = objects[i + 1]  # 後車

                is_valid_pair = True
                if dist_manager.lanes and lane_name != 'default':
                    lb = np.array(lead['bottom']).reshape(-1, 2)
                    fb = np.array(follow['bottom']).reshape(-1, 2)

                    # 取 y 值最大的兩個點視為車尾，最小兩個點視為車頭
                    lead_idxs_sorted = np.argsort(lb[:, 1])
                    follow_idxs_sorted = np.argsort(fb[:, 1])
                    lead_rear_pts = lb[lead_idxs_sorted[-2:]]
                    follow_front_pts = fb[follow_idxs_sorted[:2]]

                    # 檢查安全性
                    lead_check = any(dist_manager.is_pt_in_lane(tuple(p), lane_name) for p in lead_rear_pts)
                    follow_check = any(dist_manager.is_pt_in_lane(tuple(p), lane_name) for p in follow_front_pts)
                    is_valid_pair = lead_check and follow_check

                if is_valid_pair:
                    p1_px, p1_world = dist_manager.get_edge_point(lead['bottom'], 'closest')
                    p2_px, p2_world = dist_manager.get_edge_point(follow['bottom'], 'furthest')
                    dist_m = np.linalg.norm(p1_world - p2_world)

                    color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
                    drawing_data.append({
                        'p1': (int(p1_px[0]), int(p1_px[1])),
                        'p2': (int(p2_px[0]), int(p2_px[1])),
                        'color': color,
                        'dist': dist_m,
                        'lead_class': lead.get('class_name', 'unknown'),
                        'follow_class': follow.get('class_name', 'unknown')
                    })

    # === 繪製結果圖片 ===
    print("繪製結果...")
    result_image = draw_2d_masks(image_original.copy(), r['masks'])

    # 繪製車道範圍 (參考用)
    if dist_manager and dist_manager.lanes:
        for lname, poly in dist_manager.lanes.items():
            cv2.polylines(result_image, [poly], isClosed=True, color=(0, 255, 255), thickness=1)
            cv2.putText(result_image, lname, tuple(poly[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 繪製車輛底面框
    for item in ret_3d:
        if np.any(item['bottom'] < 0): 
            continue
        pts = item['bottom']
        for k in range(4):
            for l in range(k + 1, 4): 
                cv2.line(result_image, tuple(pts[k].astype(int)), tuple(pts[l].astype(int)), (255, 255, 255), 2)

    # 繪製距離測量結果
    for data in drawing_data:
        p1, p2, color, dist_m = data['p1'], data['p2'], data['color'], data['dist']
        lead_class, follow_class = data['lead_class'], data['follow_class']
        
        cv2.line(result_image, p1, p2, color, 2)
        cv2.circle(result_image, p1, 4, color, -1)
        cv2.circle(result_image, p2, 4, color, -1)
        
        mid_pt = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        label = f"{dist_m:.2f}m ({lead_class}->{follow_class})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result_image, (mid_pt[0] - 2, mid_pt[1] - th - 2), (mid_pt[0] + tw + 2, mid_pt[1] + 2), (0, 0, 0), -1)
        cv2.putText(result_image, label, mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 在左上角顯示物件資訊
    if lane_debug_info:
        y_offset = 20
        for debug in lane_debug_info:
            lane_label = debug['lane'] if debug['lane'] else 'None'
            confidence = "N/A"
            # 尋找對應的分數
            for item in ret_3d:
                if (item.get('center_y', -1) == debug['center'][1] and 
                    item.get('class_name') == debug['class_name']):
                    confidence = f"{item.get('score', 0):.2f}"
                    break
            
            text = f"obj{debug['obj_idx']}: {debug['class_name']} ({confidence}) -> {lane_label}"
            color_map = {'lane0': (0, 255, 0), 'lane1': (0, 0, 255), 'None': (128, 128, 128)}
            color = color_map.get(lane_label, (255, 255, 255))
            cv2.putText(result_image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 18

    # === 輸出結果 ===
    if args.output is None:
        image_name = os.path.splitext(os.path.basename(args.image))[0]
        args.output = f'images_detected_{image_name}.png'

    cv2.imwrite(args.output, result_image)
    print(f"✓ 結果已保存: {args.output}")

    # 打印統計資訊
    print(f"\n=== 統計資訊 ===")
    print(f"偵測到 {len(r['class_ids'])} 個物件")
    print(f"計算了 {len(ret_3d)} 個車輛底盤")
    print(f"測量了 {len(drawing_data)} 對車輛距離")
    if drawing_data:
        distances = [d['dist'] for d in drawing_data]
        print(f"距離範圍: {min(distances):.2f}m - {max(distances):.2f}m (平均: {np.mean(distances):.2f}m)")
