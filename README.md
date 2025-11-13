# 中科院 - 偵測隧道中前後汽車距離
- 目前是檢測車輛底部，判斷汽車位置，並使用定位點(貓眼石、車道線等)作為輔助。
- [其他檔案](https://mailntustedutw.sharepoint.com/:f:/s/MVCLab/EgpZfDd8fylJoMabVcW112kB2iCsODBwt-N2q0OWfJofeA?e=fSD9xp)

## Car bottom detection
- 用 FusionCalib 的第一步進一步改良: 原先的演算法計算量大，沒辦法達到 FPS 10。
- 作法: 先做 Object segmentation，之後用消失點來計算車輛的底部，可參考論文。
- 論文、影片等其他檔案放在雲端上。


# FusionCalib

If you use the code, please cite our paper:

```text
@article{deng2023fusioncalib,
  title={Fusioncalib: Automatic extrinsic parameters calibration based on road plane reconstruction for roadside integrated radar camera fusion sensors},
  author={Deng, Jiayin and Hu, Zhiqun and Lu, Zhaoming and Wen, Xiangming},
  journal={Pattern Recognition Letters},
  year={2023},
  publisher={Elsevier}
}
```

## Pipeline

| Step | Description                   | Input              | Output                | Method              |
| ---- | ---- | ---- | ---- | ---- |
| 1    | Vehicle Bottom Detection | Video **(output.avi)** | Pixel Coordinates of Vehicles’ Bottom Rear **(output-bottom.npy)** | MaskRCNN and vanishing points |
| 2 | ByteTrack Vehicle Bottom Tracking | output-bottom.npy | Pixel Trajectories of Vehicles’ Bottom Rear **(output-ByteTrack.npy)** | ByteTrack |
| 3 | Plane Points Calculation | Radar Trajectories **(radarTrack.npy)** and output-ByteTrack.npy | All 3D points of the road plane **(PlanePoint.npy)** | Bidirectional Selection Association and Extended Kalman Filter |
| 4 | Extrinsic Parameters Calculation | PlanePoint.npy | Results of Extrinsic Parameters **(res)** | RANSAC algorithm |


## Installation

* Clone the repository and cd to it.

    ```bash
    git clone https://github.com/RadarCameraFusionTeam-BUPT/FusionCalib.git
    cd FusionCalib
    ```

* Create and activate virtual environment using anaconda. (tested using python 3.8)

  **Note**: Change the `name` and `prefix` field in the `environment.yaml` file.

    ```bash
    conda env create -f environment.yaml
    conda activate FusionCalib
    ```

    or install using pip:

    ```bash
    pip3 install -r requirements.txt
    ```

* Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).

## Demo Data

* One of the real-world experimental data can be found in:
https://data.mendeley.com/datasets/xr38s95sm9/

* Download it and unzip all the files into a folder named `<data folder>`.

## Usage

**1. Vehicle Bottom Detection**

* You can view help of the vehicle bottom detection code using:

    ```bash
    python detectBottom.py --help
    ```

* Detect pixel coordinates of vehicles’ bottom Rear

    ```bash
    python detectBottom.py <data folder> <path to mask_rcnn_coco.h5>
    ```

**2. Vehicle Bottom Tracking**

* You can view help of the vehicle bottom tracking code using:

    ```bash
    python tracknpy.py --help
    ```

* Tracking pixel coordinates of vehicles’ bottom Rear

    ```bash
    python tracknpy.py <data folder>
    ```

**3. Plane Ploints Calculation**

* You can view help of the plane points calculation code using:

    ```bash
    python CalPlanePoint.py --help
    ```

* Plane Points Calculation

    ```bash
    python CalPlanePoint.py <data folder>
    ```

**4. Extrinsic Parameters Calculation**

* You can view help of the extrinsic parameters calculation code using:

    ```bash
    python PlaneEstAndCalExtrinsic.py --help
    ```

* Extrinsic Parameters Calculation

    ```bash
    python PlaneEstAndCalExtrinsic.py <data folder>
    ```
