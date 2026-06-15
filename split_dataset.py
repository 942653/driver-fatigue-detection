# split_dataset.py
import os
import random
import shutil
from sklearn.model_selection import train_test_split

# ========== 配置区 ==========
DATASET_ROOT = r'C:\Users\34051\Desktop\Fatigue_Detection_Project\dataset'  # 原始数据集根目录
IMAGES_DIR = os.path.join(DATASET_ROOT, 'images')
LABELS_DIR = os.path.join(DATASET_ROOT, 'labels')
VAL_RATIO = 0.2               # 验证集比例（20%）
RANDOM_SEED = 42              # 固定随机种子，保证可复现
# ============================

# 创建输出目录结构
train_img = os.path.join(DATASET_ROOT, 'train', 'images')
train_lbl = os.path.join(DATASET_ROOT, 'train', 'labels')
val_img = os.path.join(DATASET_ROOT, 'valid', 'images')
val_lbl = os.path.join(DATASET_ROOT, 'valid', 'labels')
os.makedirs(train_img, exist_ok=True)
os.makedirs(train_lbl, exist_ok=True)
os.makedirs(val_img, exist_ok=True)
os.makedirs(val_lbl, exist_ok=True)

# 获取所有图片文件名（不含扩展名）
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
base_names = [os.path.splitext(f)[0] for f in image_files]

# 随机划分
train_names, val_names = train_test_split(base_names, test_size=VAL_RATIO, random_state=RANDOM_SEED)

# 移动文件（用复制更安全，这里直接用复制，避免原始文件丢失）
for name in train_names:
    img_ext = None
    for f in image_files:
        if f.startswith(name):
            img_ext = os.path.splitext(f)[1]
            break
    if img_ext is None:
        continue
    src_img = os.path.join(IMAGES_DIR, name + img_ext)
    src_lbl = os.path.join(LABELS_DIR, name + '.txt')
    if os.path.exists(src_img):
        shutil.copy(src_img, os.path.join(train_img, name + img_ext))
    if os.path.exists(src_lbl):
        shutil.copy(src_lbl, os.path.join(train_lbl, name + '.txt'))

for name in val_names:
    img_ext = None
    for f in image_files:
        if f.startswith(name):
            img_ext = os.path.splitext(f)[1]
            break
    if img_ext is None:
        continue
    src_img = os.path.join(IMAGES_DIR, name + img_ext)
    src_lbl = os.path.join(LABELS_DIR, name + '.txt')
    if os.path.exists(src_img):
        shutil.copy(src_img, os.path.join(val_img, name + img_ext))
    if os.path.exists(src_lbl):
        shutil.copy(src_lbl, os.path.join(val_lbl, name + '.txt'))

print(f"划分完成！训练集：{len(train_names)} 张，验证集：{len(val_names)} 张")