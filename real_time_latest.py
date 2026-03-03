# fileName: real_time_one.py
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

# === 匯入測距模組 ===
try:
    from Homography import DistanceManager
except ImportError:
    print("錯誤：找不到 Homography.py，請確保該檔案在同一目錄下。")
    sys.exit(1)

class_names = {2:'car', 3:'motorcycle', 5:'bus', 7:'truck'}

COLOR_PALETTE = [
    (0, 255, 255), (0, 165, 255), (255, 0, 255), (50, 205, 50),
    (238, 130, 238), (127, 255, 0), (255, 140, 0), (0, 191, 255),
    (255, 20, 147), (138, 43, 226), (0, 250, 154), (255, 215, 0)
]

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def draw_2d_masks(image, masks):
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    N = masks.shape[2]
    if not N: return image
    result = image.copy()
    for i in range(N):
        color = colors[i % len(colors)]
        mask = masks[:, :, i]
        result = apply_mask(result, mask, color, alpha=0.3)
    return result

def Cal3dBBox(boxes, masks, class_ids, scores, vp):
    N = boxes.shape[0]
    ret = []
    if not N: return ret
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    def lineIntersection(a, b, c, d):
        a, b, c, d = np.array(a), np.array(b), np.array(c), np.array(d)
        denominator = np.cross(b-a, d-c)
        if abs(denominator) < 1e-6: return False
        x = a + (b-a) * (np.cross(c-a, d-c) / denominator)
        return x

    def find_extreme_angles(vectors, reverse_x=False):
        if len(vectors) == 0: return np.array([0, 0]), np.array([0, 0])
        min_point = np.array(vectors[0])
        max_point = np.array(vectors[0])
        if reverse_x:
            min_angle = math.atan2(min_point[1], -min_point[0])
            max_angle = min_angle
            for vec in vectors[1:]:
                angle = math.atan2(vec[1], -vec[0])
                if angle < min_angle: min_angle, min_point = angle, np.array(vec)
                if angle > max_angle: max_angle, max_point = angle, np.array(vec)
        else:
            min_angle = math.atan2(min_point[1], min_point[0])
            max_angle = min_angle
            for vec in vectors[1:]:
                angle = math.atan2(vec[1], vec[0])
                if angle < min_angle: min_angle, min_point = angle, np.array(vec)
                if angle > max_angle: max_angle, max_point = angle, np.array(vec)
        return min_point, max_point

    for i in range(N):
        class_id = class_ids[i]
        if not np.any(boxes[i]): continue
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
        if len(boundary_xs) < 4: continue
        boundary_points = [[x1 + x, y1 + y] for x, y in zip(boundary_xs, boundary_ys)]
        if len(boundary_points) > 1000:
            step = len(boundary_points) // 500
            boundary_points = boundary_points[::step]
        maskvec = [[[y-v[1], x-v[0]] for x, y in boundary_points] for v in vp]
        vp_arr = np.array(vp)
        edg = []
        for j in range(2):
            min_pt, max_pt = find_extreme_angles(maskvec[j], reverse_x=False)
            angle_min = abs(math.atan2(min_pt[1], min_pt[0]))
            angle_max = abs(math.atan2(max_pt[1], max_pt[0]))
            if angle_min < angle_max: edg.append([min_pt[::-1], max_pt[::-1]])
            else: edg.append([max_pt[::-1], min_pt[::-1]])
        min_pt, max_pt = find_extreme_angles(maskvec[2], reverse_x=True)
        angle_min = abs(math.atan2(min_pt[1], -min_pt[0]))
        angle_max = abs(math.atan2(max_pt[1], -max_pt[0]))
        if angle_min < angle_max: edg.append([min_pt[::-1], max_pt[::-1]])
        else: edg.append([max_pt[::-1], min_pt[::-1]])

        if edg[0][0][0] * edg[0][-1][0] < 0:
            c1 = lineIntersection(vp_arr[1], vp_arr[1]+edg[1][0], vp_arr[2], vp_arr[2]+edg[2][0])
            c2 = lineIntersection(vp_arr[1], vp_arr[1]+edg[1][0], vp_arr[2], vp_arr[2]+edg[2][1])
            if c1[0] > c2[0]: c1, c2 = c2, c1
            c5 = lineIntersection(vp_arr[0], vp_arr[0]+edg[0][0], vp_arr[1], vp_arr[1]+edg[1][1])
            c6 = lineIntersection(vp_arr[0], vp_arr[0]+edg[0][1], vp_arr[1], vp_arr[1]+edg[1][1])
            if c5[0] > c6[0]: c5, c6 = c6, c5
            c3, c4 = lineIntersection(vp_arr[0], c1, vp_arr[2], c5), lineIntersection(vp_arr[0], c2, vp_arr[2], c6)
        elif edg[1][0][0] * edg[1][-1][0] < 0:
            c1 = lineIntersection(vp_arr[0], vp_arr[0]+edg[0][0], vp_arr[2], vp_arr[2]+edg[2][0])
            c2 = lineIntersection(vp_arr[0], vp_arr[0]+edg[0][0], vp_arr[2], vp_arr[2]+edg[2][1])
            if c1[0] > c2[0]: c1, c2 = c2, c1
            c5 = lineIntersection(vp_arr[1], vp_arr[1]+edg[1][0], vp_arr[0], vp_arr[0]+edg[0][1])
            c6 = lineIntersection(vp_arr[1], vp_arr[1]+edg[1][1], vp_arr[0], vp_arr[0]+edg[0][1])
            if c5[0] > c6[0]: c5, c6 = c6, c5
            c3, c4 = lineIntersection(vp_arr[1], c1, vp_arr[2], c5), lineIntersection(vp_arr[1], c2, vp_arr[2], c6)
        else:
            c1 = lineIntersection(vp_arr[0], vp_arr[0]+edg[0][0], vp_arr[1], vp_arr[1]+edg[1][0])
            t1, t2 = lineIntersection(vp_arr[0], vp_arr[0]+edg[0][0], vp_arr[2], vp_arr[2]+edg[2][0]), lineIntersection(vp_arr[1], vp_arr[1]+edg[1][0], vp_arr[2], vp_arr[2]+edg[2][0])
            c2 = t1 if t1[1] < t2[1] else t2
            t1, t2 = lineIntersection(vp_arr[0], vp_arr[0]+edg[0][0], vp_arr[2], vp_arr[2]+edg[2][1]), lineIntersection(vp_arr[1], vp_arr[1]+edg[1][0], vp_arr[2], vp_arr[2]+edg[2][1])
            c3 = t1 if t1[1] < t2[1] else t2
            # c4 = lineIntersection(vp_arr[0], c2, vp_arr[1], c3) if isinstance(lineIntersection(vp_arr[0], c1, vp_arr[0], c2), bool) else lineIntersection(vp_arr[0], c3, vp_arr[1], c2)
            # 修正: c4 應該是從 vp[0]、c2 還有 vp[1]、c3 這兩條線的交點
            c4 = lineIntersection(vp_arr[0], c2, vp_arr[1], c3)
        now['bottom'] = np.array([c1, c2, c3, c4]).reshape([-1, 2])
        ret.append(now)
    return ret

def yolo_to_maskrcnn_format(yolo_results, img_shape):
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
            if mask.shape != (H, W): mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            masks[:, :, i] = mask > 0.5
    else:
        N = len(boxes)
        masks = np.zeros((img_shape[0], img_shape[1], N), dtype=bool)
        for i in range(N):
            y1, x1, y2, x2 = rois[i]
            masks[y1:y2, x1:x2, i] = True
    return {'rois': rois, 'masks': masks, 'class_ids': class_ids, 'scores': scores}

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', type=str, default='data')
    parser.add_argument('--yolo_model', default='yolo11m-seg.pt') 
    parser.add_argument("--save-vid", action='store_true', default=True)
    parser.add_argument("--save-2d-vid", action='store_true', default=True)
    parser.add_argument("--ref_dist", type=str, default='referenceDist.txt')
    args = parser.parse_args()

    DATA_PATH = args.dataPath
    STORE_VID_FLAG = args.save_vid
    SAVE_2D_VID_FLAG = args.save_2d_vid

    dist_manager = None
    ref_path = os.path.join(DATA_PATH, 'referenceDist.txt')
    lane_path = os.path.join(DATA_PATH, 'lanes8.txt')

    if not os.path.exists(ref_path) and os.path.exists(args.ref_dist): ref_path = args.ref_dist
    
    if os.path.exists(ref_path): 
        dist_manager = DistanceManager(ref_path, lane_path if os.path.exists(lane_path) else None)
    else: 
        print(f"Warning: {ref_path} not found.")

    # --- 修改點 1: 設定輸入影片名稱為 output.mp4 ---
    input_video_name = 'output.mp4'
    if not os.path.exists(f'{DATA_PATH}/{input_video_name}'): 
        exit(f'{DATA_PATH}/{input_video_name} not exist!')
    if not os.path.exists(f'{DATA_PATH}/config'): 
        exit(f'{DATA_PATH}/config not exist!')

    cap = cv2.VideoCapture(f'{DATA_PATH}/{input_video_name}')
    config = configparser.ConfigParser()
    config.read(f'{DATA_PATH}/config')
    vp_original = [json.loads(config.get('vps', f'vp{i}')) for i in range(1,4)]
    model = YOLO(args.yolo_model)
    print("模型載入完成")

    width, height = int(cap.get(3)), int(cap.get(4))
    fps, total_frames = cap.get(5), int(cap.get(7))
    
    # --- 修改點 2: 設定四碼編碼為 mp4v (更適合 MP4) ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out_3d, out_2d = None, None
    rets = []
    
    total_det_time = total_3d_time = total_homo_time = total_draw_save_time = 0
    frame_count = 0
    tqdm_bar = tqdm(total=total_frames, desc='Processing frames')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_original = frame.copy()
        
        # --- Time 1: Detection ---
        time1 = time.time() 
        yolo_results = model(frame, verbose=False, classes=[2,3,5,7])
        r = yolo_to_maskrcnn_format(yolo_results, frame.shape)
        
        # --- Time 2: 3D Calc ---
        time2 = time.time() 
        ret_3d = Cal3dBBox(r['rois'], r['masks'], r['class_ids'], r['scores'], vp_original)
        
        # --- Time 3: Scaling & Homography (LANE BASED) ---
        time3 = time.time() 
        
        ret_3d_scaled = []
        for item in ret_3d: ret_3d_scaled.append(item.copy())
        rets.append(ret_3d_scaled)

        # 2. Sequential Homography (分車道 - Center Point Logic)
        drawing_data = [] 
        lane_groups = {} 
        lane_debug_info = []  # 記錄每個物體的車道判定結果

        if dist_manager and dist_manager.H is not None:
            # (A) 分組階段：嚴格判定
            for obj_idx, item in enumerate(ret_3d_scaled):
                if np.any(item['bottom'] < 0): continue
                
                corners = item['bottom']
                center_y = np.mean(corners[:, 1])
                center_x = np.mean(corners[:, 0])
                item['center_y'] = center_y 
                
                # 只有中心點在車道內，才算該車道
                primary_lane = dist_manager.get_primary_lane(corners)
                
                # 記錄除錯資訊
                lane_debug_info.append({
                    'obj_idx': obj_idx,
                    'center': (center_x, center_y),
                    'lane': primary_lane,
                    'class_name': item.get('class_name', 'unknown')
                })
                
                if primary_lane:
                    if primary_lane not in lane_groups: lane_groups[primary_lane] = []
                    lane_groups[primary_lane].append(item)
                elif not dist_manager.lanes: 
                    # 沒讀到 lanes8.txt 的兼容模式
                    if 'default' not in lane_groups: lane_groups['default'] = []
                    lane_groups['default'].append(item)

            # (B) 計算階段
            for lane_name, objects in lane_groups.items():
                if len(objects) < 2: continue

                # 依 Center Y 從小(遠)到大(近)排序
                objects.sort(key=lambda x: x['center_y'])
                
                for i in range(len(objects) - 1):
                    lead = objects[i]       # 前車
                    follow = objects[i+1]   # 後車
                    
                    is_valid_pair = True
                    if dist_manager.lanes and lane_name != 'default':
                        # 動態判定哪兩點為車尾（y 值較大）與車頭（y 值較小），
                        # 因為底面四點的排序並不保證固定方向。
                        lb = np.array(lead['bottom']).reshape(-1, 2)
                        fb = np.array(follow['bottom']).reshape(-1, 2)

                        # 取 y 值最大的兩個點視為車尾，最小兩個點視為車頭
                        lead_idxs_sorted = np.argsort(lb[:, 1])
                        follow_idxs_sorted = np.argsort(fb[:, 1])
                        lead_rear_pts = lb[lead_idxs_sorted[-2:]]
                        follow_front_pts = fb[follow_idxs_sorted[:2]]

                        # 檢查: 前車車尾 OR 後車車頭 是否有部分在車道內 (安全檢查)
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
                            'dist': dist_m
                        })

        # --- Time 4: Draw & Save ---
        time4 = time.time() 
        if SAVE_2D_VID_FLAG:
            frame_2d = draw_2d_masks(frame_original.copy(), r['masks'])
            
            # 繪製車道範圍 (參考用)
            if dist_manager and dist_manager.lanes:
                for lname, poly in dist_manager.lanes.items():
                    cv2.polylines(frame_2d, [poly], isClosed=True, color=(0, 255, 255), thickness=1)
                    cv2.putText(frame_2d, lname, tuple(poly[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 繪製車輛底面框
            for item in ret_3d_scaled:
                if np.any(item['bottom'] < 0): continue
                pts = item['bottom']
                for k in range(4):
                    for l in range(k+1, 4): cv2.line(frame_2d, tuple(pts[k].astype(int)), tuple(pts[l].astype(int)), (255, 255, 255), 2)
            
            # 繪製距離測量結果
            for data in drawing_data:
                p1, p2, color, dist_m = data['p1'], data['p2'], data['color'], data['dist']
                cv2.line(frame_2d, p1, p2, color, 2)
                cv2.circle(frame_2d, p1, 4, color, -1)
                cv2.circle(frame_2d, p2, 4, color, -1)
                mid_pt = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
                label = f"{dist_m:.2f}m"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame_2d, (mid_pt[0]-2, mid_pt[1]-th-2), (mid_pt[0]+tw+2, mid_pt[1]+2), (0,0,0), -1)
                cv2.putText(frame_2d, label, mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if not out_2d: out_2d = cv2.VideoWriter(f'{DATA_PATH}/realtime2d.avi', fourcc, fps, (width, height))
            out_2d.write(frame_2d)

        if STORE_VID_FLAG:
            frame_3d = frame_original.copy()
            
            # 在畫面左上角顯示車道判定摘要
            if lane_debug_info:
                print(f'Frame {frame_count}: lane assignment summary')
                for debug in lane_debug_info:
                    lane_label = debug['lane'] if debug['lane'] else 'None'
                    print(f"  obj {debug['obj_idx']} ({debug['class_name']}): center=({debug['center'][0]:.0f}, {debug['center'][1]:.0f}) -> {lane_label}")
                
                y_offset = 20
                for debug in lane_debug_info:
                    lane_label = debug['lane'] if debug['lane'] else 'None'
                    text = f"obj{debug['obj_idx']}: {lane_label}"
                    color_map = {'lane0': (0, 255, 0), 'lane1': (0, 0, 255), 'None': (128, 128, 128)}
                    color = color_map.get(lane_label, (255, 255, 255))
                    cv2.putText(frame_3d, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += 18
            
            # 畫出車道範圍 (除錯用)
            if dist_manager and dist_manager.lanes:
                for lname, poly in dist_manager.lanes.items():
                    cv2.polylines(frame_3d, [poly], isClosed=True, color=(0, 255, 255), thickness=1)
                    cv2.putText(frame_3d, lname, tuple(poly[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            for item in ret_3d_scaled:
                if np.any(item['bottom'] < 0): continue
                pts = item['bottom']
                for k in range(4):
                    for l in range(k+1, 4): cv2.line(frame_3d, tuple(pts[k].astype(int)), tuple(pts[l].astype(int)), (255, 255, 255), 2)
            
            for data in drawing_data:
                p1, p2, color, dist_m = data['p1'], data['p2'], data['color'], data['dist']
                cv2.line(frame_3d, p1, p2, color, 2)
                cv2.circle(frame_3d, p1, 4, color, -1)
                cv2.circle(frame_3d, p2, 4, color, -1)
                mid_pt = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
                label = f"{dist_m:.2f}m"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame_3d, (mid_pt[0]-2, mid_pt[1]-th-2), (mid_pt[0]+tw+2, mid_pt[1]+2), (0,0,0), -1)
                cv2.putText(frame_3d, label, mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            # --- 修改點 3: 輸出檔名改為 measure.mp4 ---
            if not out_3d: out_3d = cv2.VideoWriter(f'{DATA_PATH}/measure.mp4', fourcc, fps, (width, height))
            out_3d.write(frame_3d)
        
        time5 = time.time()
        t_det, t_3d, t_homo, t_draw = time2-time1, time3-time2, time4-time3, time5-time4
        total_det_time += t_det; total_3d_time += t_3d; total_homo_time += t_homo; total_draw_save_time += t_draw
        frame_count += 1
        print('time1: {:.4f}s, time2: {:.4f}s, time3: {:.4f}s, time4: {:.4f}s, fps: {:.2f}'.format(t_det, t_3d, t_homo, t_draw, 1.0 / (time5 - time1)))
        tqdm_bar.update(1)

    # print('savePath=', DATA_PATH)
    # np.save(f'{DATA_PATH}/realtimebottom.npy', np.array(rets, dtype=object))
    if out_3d: out_3d.release()
    if out_2d: out_2d.release()
    cap.release()