# fileName: measure_v1.py
import numpy as np
import cv2

class DistanceManager:
    def __init__(self, ref_path, lane_path=None):
        self.H = None
        self.width_m = 0
        self.height_m = 0
        self.lanes = {} 

        # 1. 讀取透視變換校正檔
        try:
            ref_pts_raw, self.height_m, self.width_m = self.load_reference(ref_path)
            ref_pts = self.canonicalize_quad(ref_pts_raw)
            self.H = self.compute_homography(ref_pts, self.width_m, self.height_m)
            print(f"[DistanceManager] 成功讀取校正檔: {ref_path}")
        except Exception as e:
            print(f"[DistanceManager] 讀取校正檔失敗: {e}")
            self.H = None
        
        # 2. 讀取車道檔
        if lane_path:
            self.load_lanes(lane_path)

    def load_reference(self, ref_path):
        pts = []
        height = width = None
        with open(ref_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                if line.startswith("p"):
                    _, xs, ys = line.split()
                    pts.append((float(xs), float(ys)))
                elif line.startswith("height_m:"):
                    height = float(line.split(":")[1])
                elif line.startswith("width_m:"):
                    width = float(line.split(":")[1])
        if len(pts) != 4 or height is None or width is None:
            raise ValueError("referenceDist.txt 格式錯誤")
        return np.array(pts, dtype=float), height, width
    
    def load_lanes(self, lane_path):
        """讀取 lanes.txt，支援每條車道任意數量的點"""
        try:
            with open(lane_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    
                    parts = line.split()
                    if len(parts) < 5: continue  # 至少要有名稱 + 2個點(4個數字)
                    
                    lane_name = parts[0]
                    try:
                        coords = list(map(int, parts[1:]))
                        
                        if len(coords) % 2 != 0:
                            print(f"[WARN] 車道 {lane_name} 座標數量為奇數，略過最後一個數字")
                            coords = coords[:-1]
                        
                        n_pts = len(coords) // 2
                        if n_pts < 2:
                            print(f"[WARN] 車道 {lane_name} 點數不足 2 個，略過")
                            continue

                        pts = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
                        self.lanes[lane_name] = pts
                        print(f"[DistanceManager] 車道 {lane_name}：{n_pts} 個點")
                    except ValueError as ve:
                        print(f"[WARN] 車道 {lane_name} 解析失敗：{ve}")
                        continue

            print(f"[DistanceManager] 成功讀取車道檔: {lane_path}, 共 {len(self.lanes)} 個車道")
        except Exception as e:
            print(f"[DistanceManager] 讀取車道檔失敗: {e}")

    def is_pt_in_lane(self, pt, lane_name):
        if lane_name not in self.lanes: return False
        return cv2.pointPolygonTest(self.lanes[lane_name], (float(pt[0]), float(pt[1])), False) >= 0

    def get_primary_lane(self, pts):
        """
        [修正重點] 嚴格判定：只用「中心點」來決定這台車屬於哪個車道。
        這能避免一台車同時出現在兩個車道的清單中。
        """
        if not self.lanes:
            return None

        # 將輸入點整理為 Nx2 陣列，再計算中心點（兼容多種形狀）
        arr = np.asarray(pts).reshape(-1, 2)
        cx, cy = np.mean(arr, axis=0)

        for name, poly in self.lanes.items():
            # 檢查中心點是否在該車道內
            if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                return name
        return None

    def canonicalize_quad(self, pts):
        idx_y = np.argsort(pts[:, 1])
        top = pts[idx_y[:2]]
        bottom = pts[idx_y[2:]]
        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]
        # 回傳順序：top-left, top-right, bottom-left, bottom-right
        return np.array([top[0], top[1], bottom[0], bottom[1]], dtype=float)

    def compute_homography(self, pts_img, width_m, height_m):
        W, H_real = width_m, height_m
        pts_dst = np.array([[0, 0], [W, 0], [0, H_real], [W, H_real]], dtype=float)
        A = []
        b = []
        for (x, y), (X, Y) in zip(pts_img, pts_dst):
            A.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
            b.append(X)
            A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
            b.append(Y)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        try:
            h = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.eye(3)
        return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])

    def img_to_world(self, pt):
        if self.H is None: return np.array([0.0, 0.0])
        v = np.array([pt[0], pt[1], 1.0])
        w = self.H @ v
        w /= w[2]
        return w[:2]

    def get_edge_point(self, corners, mode='center'):
        idx_y = np.argsort(corners[:, 1])
        furthest_pts = corners[idx_y[:2]] 
        closest_pts = corners[idx_y[2:]]  

        if mode == 'closest':
            px = np.mean(closest_pts, axis=0)
        elif mode == 'furthest':
            px = np.mean(furthest_pts, axis=0)
        else:
            px = np.mean(corners, axis=0)
        world = self.img_to_world(px)
        return px, world