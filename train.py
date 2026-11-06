# train.py
import os

import torch
from ultralytics import YOLO


def main():
    # 檢查是否有可用 GPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"使用的裝置: {'GPU (CUDA)' if device == 0 else 'CPU'}")

    # 載入預訓練模型 (可選 yolov8n, yolov8s, yolo11s 等)
    model = YOLO("yolov8s.pt")

    # 開始訓練
    model.train(
        data="wheat.yaml",
        epochs=30,  # 先設 30 回合驗證流程；正式比賽通常需 50~100 回合
        imgsz=1024,  # 原圖是 1024x1024。若顯存不足 (OOM)，請改為 640 或 800
        batch=4,  # 顯卡顯存小於 8G 建議設 4；若顯存足夠可設 8 或 16
        workers=4,
        device=device,
        # 使用 abspath 轉為絕對路徑，YOLO 就不會在前面亂加 runs/detect/
        project=os.path.abspath("runs/wheat"),
        name="yolov8s_run1",
        exist_ok=True,  # 若資料夾已存在則直接覆蓋，不會產生 -2, -3
        save=True,
    )


if __name__ == "__main__":
    main()
