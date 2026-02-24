# 中科院 - 偵測隧道中前後汽車距離
## 車道分割
為了不影響左右車道前後車輛距離判斷，人工標註兩條車道範圍
- 執行 lane_draw.py，會讀一張圖片，輸出一個 txt檔。
    `python lane_draw.py --image 00428.jpg --output lanes8.txt`

## 消失點偵測
使用消失點、物件分割來計算車輛底盤位置
- `pip install lu-vp-detect`
- `python vp_detect.py --image-path lane.png`：會輸出三個消失點，把消失點貼到 `data/config` 裡面(用image coordinates)
    
## 單應性矩陣
透過人工標註，計算單應性矩陣(Homography)。
- `python .\draw_homography.py --image_path lane.png` (終端會顯示如何操作)
- 標註四個點後，輸入長、寬，會輸出 `data/referenceDist.txt`

