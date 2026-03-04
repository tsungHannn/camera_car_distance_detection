import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 在導入 pyplot 之前設定後端
import matplotlib.pyplot as plt
from lu_vp_detect import VPDetection

# Set up argument parser + options
parser = argparse.ArgumentParser(
    description="Main script for Lu's Vanishing Point Algorithm")
parser.add_argument('-i',
                    '--image-path',
                    help='Path to the input image',
                    default='data/test.png')
parser.add_argument('-lt',
                    '--length-thresh',
                    default=30,
                    type=float,
                    help='Minimum line length (in pixels) for detecting lines')
parser.add_argument(
    '-pp',
    '--principal-point',
    default=None,
    nargs=2,
    type=float,
    help='Principal point of the camera (default is image centre)')
parser.add_argument('-f',
                    '--focal-length',
                    default=1500,
                    type=float,
                    help='Focal length of the camera (in pixels)')
parser.add_argument('-d',
                    '--debug',
                    action='store_true',
                    help='Turn on debug image mode')
parser.add_argument('-ds',
                    '--debug-show',
                    action='store_true',
                    help='Show the debug image in an OpenCV window')
parser.add_argument('-dp',
                    '--debug-path',
                    default=None,
                    help='Path for writing the debug image')
parser.add_argument('-s',
                    '--seed',
                    default=None,
                    type=int,
                    help='Specify random seed for reproducible results')
parser.add_argument('-roi',
                    '--use-roi',
                    default=False,
                    action='store_true',
                    help='Enable ROI selection to mask detection area')
parser.add_argument('-v',
                    '--visualize',
                    default=True,
                    action='store_true',
                    help='Visualize the vanishing points on the image')
args = parser.parse_args()


def select_roi(image):
    """
    讓使用者框選要偵測的範圍 (ROI)
    """
    print("\n請框選要偵測的範圍 (按空白鍵或Enter確認，按C取消)")
    print("提示: 視窗可以用滑鼠拖曳邊框來調整大小")
    
    # 創建可調整大小的視窗
    window_name = "選擇ROI區域"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 選擇 ROI
    roi = cv2.selectROI(window_name, image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    return roi


def apply_roi_mask(image, roi):
    """
    根據ROI創建mask，將ROI外的區域遮罩
    """
    x, y, w, h = roi
    
    # 創建黑色mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 將ROI區域設為白色
    mask[y:y+h, x:x+w] = 255
    
    # 應用mask到圖像
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    print(f"ROI區域: x={x}, y={y}, width={w}, height={h}")
    return masked_image, mask


def visualize_vps(image_path, vps_2D, output_path, title="Vanishing Points Detection"):
    """
    在圖像上標註消失點並保存為圖片
    """
    # 讀取圖像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 顯示圖像
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    
    # 定義顏色和標籤
    colors = ['red', 'blue', 'green']
    labels = ['VP1', 'VP2', 'VP3']
    
    # 標註每個消失點
    for i, (vp, color, label) in enumerate(zip(vps_2D, colors, labels)):
        if vp is not None and len(vp) >= 2:
            plt.scatter(vp[0], vp[1], color=color, s=100, label=label, marker='x', linewidths=3)
            # 添加文字標註
            plt.annotate(f'{label}\n({vp[0]:.1f}, {vp[1]:.1f})', 
                        xy=(vp[0], vp[1]), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        fontsize=10,
                        color=color,
                        weight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    plt.legend(fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"消失點可視化圖片已保存至: {output_path}")


args = parser.parse_args()


def main():
    # Extract command line arguments
    input_path = args.image_path
    length_thresh = args.length_thresh
    principal_point = args.principal_point
    focal_length = args.focal_length
    debug_mode = args.debug
    debug_show = args.debug_show
    debug_path = args.debug_path
    seed = args.seed
    use_roi = args.use_roi
    visualize = args.visualize

    print('Input path: {}'.format(input_path))
    print('Seed: {}'.format(seed))
    print('Line length threshold: {}'.format(length_thresh))
    print('Focal length: {}'.format(focal_length))
    
    # 處理 ROI 選擇和 mask
    processed_image_path = input_path
    if use_roi:
        # 讀取圖像
        image = cv2.imread(input_path)
        if image is None:
            print(f"錯誤: 無法讀取圖像 {input_path}")
            return
        
        # 選擇 ROI
        roi = select_roi(image)
        
        if roi[2] > 0 and roi[3] > 0:  # 確認有選擇有效的ROI
            # 應用 mask
            masked_image, mask = apply_roi_mask(image, roi)
            
            # 保存 masked 圖像到臨時文件
            processed_image_path = input_path.rsplit('.', 1)[0] + '_masked.' + input_path.rsplit('.', 1)[1]
            cv2.imwrite(processed_image_path, masked_image)
            print(f"已保存 masked 圖像到: {processed_image_path}")
        else:
            print("未選擇有效的ROI，使用原始圖像")

    # Create object
    vpd = VPDetection(length_thresh, principal_point, focal_length, seed)

    # Run VP detection algorithm
    vps = vpd.find_vps(processed_image_path)
    print('Principal point: {}'.format(vpd.principal_point))

    # Show VP information
    print("The vanishing points in 3D space are: ")
    for i, vp in enumerate(vps):
        print("Vanishing Point {:d}: {}".format(i + 1, vp))

    vp2D = vpd.vps_2D
    print("\nThe vanishing points in image coordinates are: ")
    for i, vp in enumerate(vp2D):
        print("Vanishing Point {:d}: {}".format(i + 1, vp))

    # Extra stuff
    if debug_mode or debug_show:
        st = "Creating debug image"
        if debug_show:
            st += " and showing to the screen"
        if debug_path is not None:
            st += "\nAlso writing debug image to: {}".format(debug_path)

        if debug_show or debug_path is not None:
            print(st)
            vpd.create_debug_VP_image(debug_show, debug_path)
    
    # 可視化消失點
    if visualize and vp2D is not None and len(vp2D) > 0:
        print("\n正在生成消失點可視化圖片...")
        # 生成輸出文件名
        output_path = processed_image_path.rsplit('.', 1)[0] + '_vp_result.png'
        visualize_vps(processed_image_path, vp2D, output_path, "Vanishing Points Detection Result")


if __name__ == "__main__":
    main()
