# inference.py (含存圖功能版)
import glob
import os

import pandas as pd
from ultralytics import YOLO


def main():
    model_weight = "runs/wheat/yolov8s_run1/weights/best.pt"
    if not os.path.exists(model_weight):
        raise FileNotFoundError(f"找不到權重檔案 {model_weight}，請先確認訓練已完成。")

    # 建立存放預測結果圖片的資料夾
    output_img_dir = "runs/wheat/test_predictions"
    os.makedirs(output_img_dir, exist_ok=True)

    model = YOLO(model_weight)
    test_images = glob.glob("data/raw/test/*.jpg")

    records = []
    print(f"正在對 {len(test_images)} 張測試圖片進行預測並繪製邊界框...")

    for img_path in test_images:
        img_id = os.path.basename(img_path).replace(".jpg", "")

        # 進行推論
        preds = model(img_path, conf=0.35, imgsz=1024, verbose=False)[0]

        # ★ 將畫好預測框與信心度的圖片存下來 ★
        save_path = os.path.join(output_img_dir, f"{img_id}_pred.jpg")
        preds.save(filename=save_path)

        pred_strs = []
        for box in preds.boxes:
            conf = float(box.conf[0])
            xc, yc, w, h = box.xywh[0].tolist()

            xmin = round(xc - w / 2.0)
            ymin = round(yc - h / 2.0)
            w = round(w)
            h = round(h)

            pred_strs.append(f"{conf:.4f} {xmin} {ymin} {w} {h}")

        records.append({"image_id": img_id, "PredictionString": " ".join(pred_strs)})

    sub_df = pd.DataFrame(records)
    sub_df.to_csv("submission.csv", index=False)
    print(f"\n[完成] 預測圖片已全部存放在: {output_img_dir}/")
    print("[完成] submission.csv 已產出！")


if __name__ == "__main__":
    main()
