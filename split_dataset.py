import os
import shutil
import random

# === CONFIGURATION ===
source_images = '/scratch/svaidy33/CMB_detection/YOLO_data/vimages'   # e.g. /scratch/svaidy33/cmbproj/source_volumes
source_labels = '/scratch/svaidy33/CMB_detection/YOLO_data/vlabels'   # e.g. /scratch/svaidy33/cmbproj/source_labels

target_base = '/scratch/svaidy33/CMB_detection/YOLO_data'
splits = {
    "train": 0.6,
    "val": 0.0,
    "test": 0.4
}

# === CREATE TARGET FOLDERS ===
for split in splits.keys():
    os.makedirs(os.path.join(target_base, 'valdo/images', split), exist_ok=True)
    os.makedirs(os.path.join(target_base, 'valdo/labels', split), exist_ok=True)

# === GET LIST OF .nii.gz FILES ===
image_files = [f for f in os.listdir(source_images) if f.endswith('.nii.gz')]
random.shuffle(image_files)

# === SPLIT DATA ===
n_total = len(image_files)
print("total files : " , n_total)
n_train = int(splits["train"] * n_total)
n_val = int(splits["val"] * n_total)
n_test = n_total - n_train - n_val

split_data = {
    "train": image_files[:n_train],
    "val": image_files[n_train:n_train + n_val],
    "test": image_files[n_train + n_val:]
}

# === COPY FILES ===
for split, files in split_data.items():
    for img_file in files:
        base_name = img_file.replace(".nii.gz","")  # removes .nii.gz
        label_file = base_name + '.txt'

        # Copy image (.nii.gz)
        src_img_path = os.path.join(source_images, img_file)
        dst_img_path = os.path.join(target_base, 'valdo/images', split, img_file)
        shutil.copyfile(src_img_path, dst_img_path)

        # Copy label
        src_label_path = os.path.join(source_labels, label_file)
        dst_label_path = os.path.join(target_base, 'valdo/labels', split, label_file)
        if os.path.exists(src_label_path):
            shutil.copyfile(src_label_path, dst_label_path)
        else:
            print(f"[WARNING] Label file not found for {img_file}: {label_file}")


