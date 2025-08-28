import os
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np


# ---------------------- Utils ----------------------
def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)  # H x W x D
    data = np.transpose(data, (2, 0, 1))       # → D x H x W
    return data


def extract_patch(volume, center, patch_size=(6, 14, 14)):
    z, y, x = center
    dz, dy, dx = patch_size[0]//2, patch_size[1]//2, patch_size[2]//2
    z1, z2 = max(0, z-dz), min(volume.shape[0], z+dz)
    y1, y2 = max(0, y-dy), min(volume.shape[1], y+dy)
    x1, x2 = max(0, x-dx), min(volume.shape[2], x+dx)
    patch = np.zeros(patch_size, dtype=np.float32)
    patch[:z2-z1, :y2-y1, :x2-x1] = volume[z1:z2, y1:y2, x1:x2]
    return patch


# ---------------------- Dataset ----------------------
class PatchDataset(Dataset):
    def __init__(self, image_dir, label_dir, patch_size=(6, 14, 14), num_negatives=1):
        self.samples = []
        self.patch_size = patch_size

        for fname in os.listdir(image_dir):
            if not fname.endswith(".nii") and not fname.endswith(".nii.gz"):
                continue
            img_path = os.path.join(image_dir, fname)
            vol = load_nifti(img_path)
            basename = fname.replace('.nii.gz', '')
            label_path = os.path.join(label_dir, basename + '.txt')
            pos_centers = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = parts = line.strip().split()
                    if len(parts) >= 3:
                        z = int(float(parts[1])*vol.shape[0])
                        x = int(float(parts[2])*vol.shape[2])
                        y = int(float(parts[3])*vol.shape[1])
                        pos_centers.append((z,y,x))

            #pos_centers = labels.get(fname, [])

            # positives
            for c in pos_centers:
                self.samples.append((img_path, c, 1))

            # negatives
            for _ in range(num_negatives * len(pos_centers)):
                c = (
                    random.randint(patch_size[0]//2, vol.shape[0]-patch_size[0]//2-1),
                    random.randint(patch_size[1]//2, vol.shape[1]-patch_size[1]//2-1),
                    random.randint(patch_size[2]//2, vol.shape[2]-patch_size[2]//2-1)
                )
                self.samples.append((img_path, c, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, center, label = self.samples[idx]
        vol = load_nifti(img_path)
        patch = extract_patch(vol, center, self.patch_size)
        patch = torch.from_numpy(patch).unsqueeze(0)  # [1, D,H,W]
        return patch, torch.tensor(label, dtype=torch.float32)


# ---------------------- Model ----------------------
class BinaryClassifier3D(nn.Module):
    def __init__(self):
        super().__init__()
        patch_size = (6, 14, 14)
        self.conv1 = nn.Conv3d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *patch_size)  # (batch, channel, D, H, W)
            out = F.relu(self.conv1(dummy))
            out = F.relu(self.conv2(out))
            flat_size = out.view(1, -1).size(1)

        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

def load_model(checkpoint_path, device):
    model = BinaryClassifier3D().to(device)  # same architecture
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ---------------------- Training ----------------------
def train(args):
    # load labels (assuming JSON: {"file.nii.gz": [[z,y,x], ...]})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PatchDataset(args.images, args.labels, patch_size=(6, 14, 14))
    test_dataset  = PatchDataset(args.test_images, args.test_labels, patch_size=(6, 14, 14))
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = BinaryClassifier3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    for epoch in range(args.epochs):
        total_loss = 0.0
        test_loss = 0.0
        model.train()
        for patches, targets in train_loader:
            patches, targets = patches.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(patches)
            
            loss = criterion(logits, targets)
            preds = torch.sigmoid(logits)
            #print(preds)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        avg_train_loss = total_loss / len(train_loader)
        model.eval()
        test_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for patches, labels in test_loader:
                patches, labels = patches.to(device), labels.to(device).float()
                logits = model(patches).squeeze()
                
                loss = criterion(logits, labels)
                preds = torch.sigmoid(logits)

                test_loss += loss.item()

                # compute accuracy
                predicted = (preds > 0.5).long()
                correct += (predicted == labels.long()).sum().item()
                total += labels.size(0)

        avg_test_loss = test_loss / len(test_loader)
        accuracy = 100.0 * correct / total
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss={avg_train_loss:.4f} | Test Loss={avg_test_loss:.4f} | Acc={accuracy:.2f}%")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_epoch = epoch + 1

            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, args.checkpoint)


# ---------------------- Main ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True,
                        help="Path to directory with NIfTI images")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to txt file with labels {filename: [[z,y,x], ...]}")
    parser.add_argument("--test_images", type=str, required=True,
                        help="Path to directory with NIfTI images")
    parser.add_argument("--test_labels", type=str, required=True,
                        help="Path to txt file with labels {filename: [[z,y,x], ...]}")
    parser.add_argument("--checkpoint", type=str, default="class_runs/binary_classifier3d_best.pth",
                        help="Path to save checkpoint")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    args = parser.parse_args()

    train(args)
