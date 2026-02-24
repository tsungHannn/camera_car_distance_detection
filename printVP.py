import matplotlib
matplotlib.use('TkAgg')  # 在導入 pyplot 之前設定後端
import matplotlib.pyplot as plt
import cv2

# 定义消失点
# vp1 = [1.063949044585987e+03,-23.184713375796150]
# vp2 = [-1.904197530864198e+03,1.387654320987654e+02]
# vp3 = [7.906068019518707e+02,3.120251076916543e+03]

# 天橋上
# vp1 = [1708.5226, 503.95215]
# vp2 = [-2426.383, 344.34625]
# vp3 = [725.00476, 8514.0625]

# 雪隧 south16
# vp1 =  [7018.3667, 639.27576]
# vp2 = [299.83838, -24.310333]
# vp3 = [167.88293, 5259.986]

# 雪隧 south26
vp1 = [35935.2,     2948.0808]
vp2 = [ 616.9469,  -101.96484]
vp3 = [ 304.972,  4359.4844]


# 读取图像
image_path = 'lane.png'  # 替换为你图像的路径
image = cv2.imread(image_path)

# 将图像转换为 RGB（OpenCV 读取的图像是 BGR）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 显示图像
plt.imshow(image_rgb)
plt.scatter(vp1[0], vp1[1], color='red', s=100, label='VP1')  # 标注VP1
plt.scatter(vp2[0], vp2[1], color='blue', s=100, label='VP2')  # 标注VP2
plt.scatter(vp3[0], vp3[1], color='green', s=100, label='VP3')  # 标注VP3

# 添加图例
plt.legend()

# 显示结果
plt.show()
