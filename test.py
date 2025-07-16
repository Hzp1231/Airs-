import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion
import pandas as pd

# ---------- 基础工具函数 ----------
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def list_all_files(startpath):
    return [f for f in os.listdir(startpath) if f.lower().endswith('.jpg') or f.lower().endswith('.png')]

# ---------- 加载模型并生成预测 ----------
def load_model(weights_path, device):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}!")
    model = YOLO(weights_path).to(device)
    model.fuse()
    return model

def run_yolo_predictions(image_dir_rgb, image_dir_tir, output_dirs, model_paths, device):
    all_files = list_all_files(image_dir_rgb)
    for model_path, output_dir in zip(model_paths, output_dirs):
        model = load_model(model_path, device)
        ensure_dir(output_dir)

        for filename in tqdm(all_files, desc=f"Inferencing with {os.path.basename(model_path)}"):
            rgb_path = os.path.join(image_dir_rgb, filename)
            tir_path = os.path.join(image_dir_tir, filename)

            img_rgb = cv2.imread(rgb_path)
            img_ir = cv2.imread(tir_path)
            if img_rgb is None or img_ir is None:
                continue
            input_img = np.concatenate((img_rgb, img_ir), axis=2)

            results = model.predict(input_img, imgsz=640, iou=0.5, conf=0.0001, save=False)[0]

            boxes = results.boxes
            if boxes is None or boxes.shape[0] == 0:
                open(os.path.join(output_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt')), 'w').close()
                continue

            with open(os.path.join(output_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt')), 'w') as f:
                for box in boxes:
                    cls = int(box.cls)
                    cx, cy, w, h = box.xywh.cpu().numpy()[0]
                    conf = float(box.conf.cpu().numpy()[0])
                    f.write(f"{cls} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f} {conf:.6f}\n")

# ---------- 加载预测框用于 WBF ----------
def load_predictions(pred_path, img_wh):
    boxes, scores, labels = [], [], []
    if not os.path.exists(pred_path):
        return boxes, scores, labels

    with open(pred_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: continue
            cls = int(parts[0])
            cx, cy, w, h, conf = map(float, parts[1:])
            x1 = (cx - w / 2) / img_wh[0]
            y1 = (cy - h / 2) / img_wh[1]
            x2 = (cx + w / 2) / img_wh[0]
            y2 = (cy + h / 2) / img_wh[1]
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            labels.append(cls)
    return boxes, scores, labels

def run_wbf(image_dir, pred_dirs, output_dir, weights):
    ensure_dir(output_dir)
    img_list = list_all_files(image_dir)

    for img_name in tqdm(img_list, desc="Running WBF"):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        H, W = img.shape[:2]

        boxes_list, scores_list, labels_list = [], [], []
        for pred_dir in pred_dirs:
            pred_path = os.path.join(pred_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            boxes, scores, labels = load_predictions(pred_path, (W, H))
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        if not any(scores_list):
            continue

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=0.55, skip_box_thr=0.00001
        )

        out_path = os.path.join(output_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(out_path, 'w') as f:
            for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
                abs_x1, abs_y1, abs_x2, abs_y2 = x1 * W, y1 * H, x2 * W, y2 * H
                cx = (abs_x1 + abs_x2) / 2
                cy = (abs_y1 + abs_y2) / 2
                w = abs_x2 - abs_x1
                h = abs_y2 - abs_y1
                f.write(f"{int(label)} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f} {score:.6f}\n")

# ---------- 保存为 CSV ----------
def convert_txts_to_csv(label_folder, output_csv):
    data_rows = []
    id_counter = 0

    for filename in sorted(os.listdir(label_folder)):
        if not filename.endswith('.txt'):
            continue

        image_id = int(os.path.splitext(filename)[0])  # 假设图像名是纯数字（如100.txt）
        txt_path = os.path.join(label_folder, filename)

        # 默认值
        category_ids = ["0"]
        bboxes = ["[1.0,1.0,1.0,1.0]"]
        scores = ["0.01"]

        # 检查文件是否为空
        if os.path.getsize(txt_path) > 0:
            category_ids = []
            bboxes = []
            scores = []

            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    class_id = parts[0]
                    xmin = float(parts[1])
                    ymin = float(parts[2])
                    xmax = float(parts[3])
                    ymax = float(parts[4])
                    score = float(parts[5])

                    category_ids.append(str(class_id))
                    bboxes.append(f"[{xmin:.4f},{ymin:.4f},{xmax:.4f},{ymax:.4f}]")
                    scores.append(f"{score:.6f}")

            # 如果文件有内容但格式不正确，仍使用默认值
            if not category_ids:
                category_ids = ["0"]
                bboxes = ["[1.0,1.0,1.0,1.0]"]
                scores = ["0.01"]

        row = {
            "id": id_counter,
            "image_id": image_id,
            "category_id": ",".join(category_ids),
            "bbox": ",".join(bboxes),
            "score": ",".join(scores)
        }
        data_rows.append(row)
        id_counter += 1

    # 转为DataFrame并写出
    df = pd.DataFrame(data_rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

# ---------- 主流程 ----------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 相对路径设置
    image_dir_rgb = os.path.join(BASE_DIR, "datasets", "RGB")
    image_dir_tir = os.path.join(BASE_DIR, "datasets", "TIR")
    model_paths = [
        os.path.join(BASE_DIR, "weights", "ADDL_conf0.0001.pt"),
        os.path.join(BASE_DIR, "weights", "ADDL_P4_conf0.0001.pt"),
        os.path.join(BASE_DIR, "weights", "ConcatL_conf0.0001.pt"),
        os.path.join(BASE_DIR, "weights", "Concat2L_conf0.0001.pt")
    ]

    # 统一输出目录
    output_base = os.path.join(BASE_DIR, "output")
    temp_output_dirs = [os.path.join(output_base, f"model{i+1}") for i in range(len(model_paths))]
    wbf_output_dir = os.path.join(output_base, "wbf_result")
    final_csv_path = os.path.join(output_base, "submission.csv")

    # 步骤1：多个模型推理
    run_yolo_predictions(image_dir_rgb, image_dir_tir, temp_output_dirs, model_paths, device)

    # 步骤2：WBF融合多个模型预测
    run_wbf(image_dir_rgb, temp_output_dirs, wbf_output_dir, weights=[1.0] * len(model_paths))

    # 步骤3：生成CSV
    convert_txts_to_csv(wbf_output_dir, final_csv_path)


if __name__ == "__main__":
    main()

