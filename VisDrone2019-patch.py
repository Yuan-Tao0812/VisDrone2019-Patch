import os
import cv2
import shutil

# 配置路径
image_dir = 'D:/resource/dissertation/dataset-VisDrone2019-DET/VisDrone2019-DET-train/images'
anno_dir = 'D:/resource/dissertation/dataset-VisDrone2019-DET/VisDrone2019-DET-train/annotations'
patch_save_dir = 'D:/resource/dissertation/dataset-VisDrone2019-DET/VisDrone2019-Patch'

# 创建保存路径
os.makedirs(patch_save_dir, exist_ok=True)

# VisDrone 标注格式：
# 每行: x, y, w, h, score, class_id, truncation, occlusion

# 类别映射（可按需修改/简化）
label_map = {
    '0': 'pedestrian',
    '1': 'people',
    '2': 'bicycle',
    '3': 'car',
    '4': 'van',
    '5': 'truck',
    '6': 'tricycle',
    '7': 'awning-tricycle',
    '8': 'bus',
    '9': 'motor'
}

# 控制 patch 尺寸最小值
MIN_PATCH_SIZE = 20

for anno_file in os.listdir(anno_dir):
    image_id = os.path.splitext(anno_file)[0]
    img_path = os.path.join(image_dir, f"{image_id}.jpg")
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    h_img, w_img = img.shape[:2]

    with open(os.path.join(anno_dir, anno_file), 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split(',')
        if len(parts) < 8:
            continue

        x, y, w, h = map(int, parts[0:4])
        class_id = parts[5]

        # 跳过不感兴趣类别
        if class_id not in label_map:
            continue

        class_name = label_map[class_id]

        # 跳过太小的目标
        if w < MIN_PATCH_SIZE or h < MIN_PATCH_SIZE:
            continue

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        patch = img[y1:y2, x1:x2]

        save_path = os.path.join(patch_save_dir, class_name)
        os.makedirs(save_path, exist_ok=True)

        filename = f"{image_id}_{idx}.jpg"
        cv2.imwrite(os.path.join(save_path, filename), patch)

print("✅ Patch 裁剪完成！每类图像已分别保存。")
