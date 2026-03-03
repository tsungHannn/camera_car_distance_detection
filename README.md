# 中科院 - 偵測隧道中前後汽車距離
- 執行前，須完成車道分割(人工標註)、單應性矩陣(人工標註)、消失點偵測。
- 主程式: `python real_time_latest.py`
## 車道分割
為了不影響左右車道前後車輛距離判斷，人工標註兩條車道範圍
- 執行 lane_draw.py，會讀一張圖片，輸出一個 txt檔。
- `python lane_draw.py --image lane.jpg --output lanes8.txt`
    <details>
    <summary> 執行教學 </summary>
        
    <img width="2879" height="1703" alt="image" src="https://github.com/user-attachments/assets/4a579678-0516-455a-bb70-c293c70c46c8" />

    - 終端會顯示當前狀態
    - 按下數字鍵: 1 → 開始標第一條車道；2 → 標第二條車道
    - 左鍵: 新增點
    - z: 刪除最後一個點
    - q: 儲存並退出
    </details>
  
  

## 消失點偵測
使用消失點、物件分割來計算車輛底盤位置
- `pip install lu-vp-detect`
- `python vp_detect.py --image-path lane.jpg`：會輸出三個消失點，把消失點貼到 `data/config` 裡面(用image coordinates)
    
## 單應性矩陣
透過人工標註，計算單應性矩陣(Homography)。
- `python .\draw_homography.py --image_path lane.png` (終端會顯示如何操作)
- 標註四個點後，輸入長、寬，會輸出 `data/referenceDist.txt`
  <details>
      <summary> 執行教學 </summary>

  <img width="2879" height="1700" alt="image" src="https://github.com/user-attachments/assets/440f0959-f4dc-4874-99d6-3c573c08a9cd" />
  
  - 終端會顯示當前狀態
  - 直接用滑鼠點四個點
  - 按 Enter 確認，按 ESC 再標一次
  - 按下 Enter 後在終端輸入長、寬

  </details>

