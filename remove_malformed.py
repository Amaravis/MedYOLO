import os

label_dir = '/scratch/svaidy33/CMB_detection/YOLO_data/labels/val'
for fname in os.listdir(label_dir):
    if not fname.endswith('.txt'):
        continue
    with open(os.path.join(label_dir, fname)) as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 7:
                print(f"❌ {fname} Line {i+1}: Invalid format — {line.strip()}")
