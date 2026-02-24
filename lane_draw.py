import cv2
import argparse
import os

# 全域變數
lanes = {}            # {lane_name: [points]}，例如 {"lane1": [(x,y), ...]}
current_lane_name = None
img = None
vis = None


def redraw():
    global vis, img, lanes, current_lane_name
    vis = img.copy()

    colors = {
        "lane1": (0, 255, 0),    # 綠
        "lane2": (0, 128, 255),  # 橙
        "lane3": (255, 0, 255),  # 紫
        "lane4": (255, 255, 0),  # 青
    }
    default_color = (200, 200, 200)

    for lane_name, pts in lanes.items():
        # 正在編輯的車道用黃色，其他用各自顏色
        if lane_name == current_lane_name:
            color = (0, 255, 255)
        else:
            color = colors.get(lane_name, default_color)

        for i, (x, y) in enumerate(pts):
            cv2.circle(vis, (int(x), int(y)), 5, color, -1)
            if i > 0:
                x0, y0 = pts[i - 1]
                cv2.line(vis, (int(x0), int(y0)), (int(x), int(y)), color, 2)
        if pts:
            # 在第一個點旁標示車道名
            cv2.putText(vis, lane_name, (int(pts[0][0]) + 6, int(pts[0][1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # 提示文字
    hint1 = "Left-click: add point | 'z': undo | 'q'/ESC: save & quit"
    cv2.putText(vis, hint1, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    hint2 = "Press '1','2','3'... to select lane | Current: {}  Points: {}".format(
        current_lane_name if current_lane_name else "None",
        len(lanes.get(current_lane_name, [])) if current_lane_name else 0
    )
    cv2.putText(vis, hint2, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


def mouse_callback(event, x, y, flags, param):
    global lanes, current_lane_name, vis

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_lane_name is None:
            print("請先按數字鍵 '1'、'2'... 選擇車道，再用滑鼠點。")
            return

        lanes[current_lane_name].append((x, y))
        n = len(lanes[current_lane_name])
        print(f"[{current_lane_name}] 加入第 {n} 個點：({x}, {y})")
        redraw()


def save_lanes(output_path):
    active_lanes = {k: v for k, v in lanes.items() if v}
    if not active_lanes:
        print("[WARN] 沒有任何 lane 被標記，不寫入檔案。")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# lane_name x0 y0 x1 y1 ...\n")
        for lane_name, pts in active_lanes.items():
            coord_str = " ".join(f"{int(x)} {int(y)}" for (x, y) in pts)
            f.write(f"{lane_name} {coord_str}\n")

    print(f"[INFO] 已儲存 {len(active_lanes)} 條車道到 {output_path}")
    for lane_name, pts in active_lanes.items():
        print(f"  {lane_name}: {len(pts)} 個點")


def switch_lane(new_lane_name):
    global current_lane_name, lanes
    if new_lane_name not in lanes:
        lanes[new_lane_name] = []
    current_lane_name = new_lane_name
    print(f"切換到 {current_lane_name}，目前已有 {len(lanes[current_lane_name])} 個點。")
    redraw()


def main():
    global img, vis, current_lane_name

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="lane.png")
    parser.add_argument("--output", type=str, default="data/lanes.txt")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] 找不到影像：{args.image}")
        return

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] 無法讀取影像：{args.image}")
        return

    redraw()
    cv2.namedWindow("lane editor", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("lane editor", mouse_callback)

    print("操作說明：")
    print("  '1'～'9'：選擇要標記的車道（自動建立）")
    print("  左鍵：在當前車道新增一個點（可無限新增）")
    print("  'z'：刪除當前車道的最後一個點")
    print("  'q' 或 ESC：儲存並結束")
    print()

    while True:
        cv2.imshow("lane editor", vis)
        key = cv2.waitKey(30) & 0xFF

        # 數字鍵 1~9 切換車道
        if ord('1') <= key <= ord('9'):
            switch_lane(f"lane{chr(key)}")

        elif key == ord('z'):
            if current_lane_name and lanes.get(current_lane_name):
                removed = lanes[current_lane_name].pop()
                print(f"[{current_lane_name}] 移除點：{removed}，剩 {len(lanes[current_lane_name])} 個")
                redraw()
            else:
                print("目前車道沒有點可以刪除。")

        elif key == ord('q') or key == 27:
            print("結束標記，準備儲存。")
            break

    cv2.destroyAllWindows()
    save_lanes(args.output)


if __name__ == "__main__":
    main()