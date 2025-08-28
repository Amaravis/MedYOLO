import os
import shutil

# Paths
src_dir = r"/scratch/svaidy33/CMB_detection/YOLO_data/images/test"
dst_dir = r"/scratch/svaidy33/CMB_detection/YOLO_data/images/swi_test"

# Make sure destination exists
os.makedirs(dst_dir, exist_ok=True)

# Loop over all files in the source directory
for filename in os.listdir(src_dir):
    file_path = os.path.join(src_dir, filename)

    # Skip directories, process only files
    if os.path.isfile(file_path):
        
        # Example: move if filename contains 'report'
        if "cmb" not in filename.lower():  
            shutil.move(file_path, os.path.join(dst_dir, filename))
            print(f"Moved: {filename}")