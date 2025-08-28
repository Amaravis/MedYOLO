import os
import shutil

labels_dir = '/scratch/svaidy33/CMB_detection/YOLO_data/labels/val/'
images_dir = '/scratch/svaidy33/CMB_detection/YOLO_data/images/val/'
backup_dir_labels = '/scratch/svaidy33/CMB_detection/YOLO_data/empty_labels'
backup_dir_images = '/scratch/svaidy33/CMB_detection/YOLO_data/empty_images'

# Create backup dirs
os.makedirs(backup_dir_labels, exist_ok=True)
os.makedirs(backup_dir_images, exist_ok=True)

def is_label_file_empty(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        # Remove blank lines and comments or whitespace-only lines
        lines = [line.strip() for line in lines if line.strip()]
        return len(lines) == 0
    except Exception as e:
        print(f"⚠️ Error reading {filepath}: {e}")
        return True

for fname in os.listdir(labels_dir):
    if not fname.endswith('.txt'):
        continue
    fpath = os.path.join(labels_dir, fname)
    if is_label_file_empty(fpath):
        # Move label
        shutil.move(fpath, os.path.join(backup_dir_labels, fname))

        # Move corresponding .nii.gz
        base = os.path.splitext(fname)[0]
        imgname = base + '.nii.gz'
        imgpath = os.path.join(images_dir, imgname)
        if os.path.exists(imgpath):
            shutil.move(imgpath, os.path.join(backup_dir_images, imgname))
        else:
            print(f"⚠️ Image file not found for empty label: {imgname}")

print("✅ Empty label/image pairs moved based on content.")
