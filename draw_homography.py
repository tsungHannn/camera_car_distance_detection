import cv2
import sys
import argparse

points = []
img_display = None

def mouse_callback(event, x, y, flags, param):
    global points, img_display
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(img_display, (x, y), 6, (0, 255, 0), -1)
        idx = len(points) - 1
        cv2.putText(img_display, f'p{idx}', (x+8, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(points) > 1:
            cv2.line(img_display, points[-2], points[-1], (0, 200, 255), 2)
        if len(points) == 4:
            cv2.line(img_display, points[-1], points[0], (0, 200, 255), 2)
        cv2.imshow("Select 4 Points", img_display)

def main():
    global img_display

    parser = argparse.ArgumentParser(description="在圖片上點選四邊形並輸出 quad points txt")
    parser.add_argument("--image_path", default='lane.png', type=str, help="輸入圖片路徑")
    parser.add_argument("-o", "--output", type=str, default="referenceDist.txt", help="輸出檔案名稱（預設 referenceDist.txt）")
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"無法讀取圖片：{args.image_path}，請確認路徑是否正確。")
        sys.exit(1)

    img_display = img.copy()
    # 建立可調整大小的視窗
    cv2.namedWindow("Select 4 Points", cv2.WINDOW_NORMAL)

    # 預設視窗大小為 1280x720，可自由拖曳調整
    cv2.resizeWindow("Select 4 Points", 1280, 720)
    cv2.imshow("Select 4 Points", img_display)
    cv2.setMouseCallback("Select 4 Points", mouse_callback)

    print("請在圖片上依序點選 4 個點（p0 → p1 → p2 → p3）")
    print("點完後按 Enter 確認，按 ESC 重新來過。")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(points) == 4:  # Enter
            break
        elif key == 27:  # ESC 重置
            points.clear()
            img_display = img.copy()
            cv2.imshow("Select 4 Points", img_display)
            print("已重置，請重新點選。")

    cv2.destroyAllWindows()

    print(f"\n已選取的點：")
    for i, p in enumerate(points):
        print(f"  p{i}: {p}")

    try:
        height_m = float(input("\n請輸入四邊形的高度 height_m（公尺）: "))
        width_m  = float(input("請輸入四邊形的寬度 width_m（公尺）: "))
    except ValueError:
        print("輸入格式錯誤，請輸入數字。")
        sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# quad points: x y\n")
        for i, p in enumerate(points):
            f.write(f"p{i} {p[0]} {p[1]}\n")
        f.write(f"height_m: {height_m}\n")
        f.write(f"width_m: {width_m}\n")

    print(f"\n✅ 已儲存至 {args.output}")

if __name__ == "__main__":
    main()