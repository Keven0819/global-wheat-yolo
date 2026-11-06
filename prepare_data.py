# prepare_data.py
import ast
import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

RAW_DIR = "data/raw"
OUT_DIR = "data/yolo_dataset"


def main():
    csv_path = os.path.join(RAW_DIR, "train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 {csv_path}，請確認資料已解壓縮。")

    df = pd.read_csv(csv_path)
    # 將 "[x, y, w, h]" 字串解析為數值陣列
    df["bbox"] = df["bbox"].apply(ast.literal_eval)

    # 建立 YOLO 資料夾架構
    for split in ["train", "val"]:
        os.makedirs(f"{OUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT_DIR}/labels/{split}", exist_ok=True)

    # 依照 image_id 切分 80% 訓練集、20% 驗證集
    unique_images = df["image_id"].unique()
    train_ids, val_ids = train_test_split(unique_images, test_size=0.2, random_state=42)

    def process_split(ids, split):
        for img_id in tqdm(ids, desc=f"處理 {split} 集"):
            src_img = os.path.join(RAW_DIR, "train", f"{img_id}.jpg")
            dst_img = f"{OUT_DIR}/images/{split}/{img_id}.jpg"
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)

            # 抓取該圖片所有的 bounding boxes
            records = df[df["image_id"] == img_id]
            label_file = f"{OUT_DIR}/labels/{split}/{img_id}.txt"

            lines = []
            for _, row in records.iterrows():
                x, y, w, h = row["bbox"]
                img_w, img_h = row["width"], row["height"]

                # 排除異常標註 (有些寬高小於等於 0)
                if w <= 0 or h <= 0:
                    continue

                # 轉換為 YOLO 要求的中心點與相對比例
                x_center = (x + w / 2.0) / img_w
                y_center = (y + h / 2.0) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                # 限制在 [0, 1] 範圍內避免越界報錯
                x_center = min(max(x_center, 0.0), 1.0)
                y_center = min(max(y_center, 0.0), 1.0)
                w_norm = min(max(w_norm, 0.0), 1.0)
                h_norm = min(max(h_norm, 0.0), 1.0)

                # 小麥類別 ID 為 0
                lines.append(
                    f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                )

            with open(label_file, "w") as f:
                f.write("\n".join(lines))

    process_split(train_ids, "train")
    process_split(val_ids, "val")
    print(f"\n[成功] 資料集已整理完成，存放在: {OUT_DIR}")


if __name__ == "__main__":
    main()
