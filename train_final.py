"""
FINAL - Offroad Segmentation Training
Clean, minimal, optimized for Colab
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter

# ============================================================================
# CONFIG
# ============================================================================

# UPDATE THESE PATHS
DATA_DIR = '/content/drive/MyDrive/Offroad_Segmentation_Training_Dataset/train'
VAL_DIR = '/content/drive/MyDrive/Offroad_Segmentation_Training_Dataset/val'
OUTPUT_DIR = '/content/drive/MyDrive/output'

# Training settings
EPOCHS = 12
BATCH_SIZE = 8
LR = 1e-3
IMG_H, IMG_W = 518, 952  # High res for Colab

# Classes
VALUE_MAP = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 600:6, 700:7, 800:8, 7100:9, 10000:10}
CLASS_NAMES = ['BG','Trees','LushBush','DryGrass','DryBush','Clutter','Flowers','Logs','Rocks','Land','Sky']
N_CLASSES = 11

# ============================================================================
# DATASET
# ============================================================================

def get_transforms(h, w, train=True):
    if train:
        return A.Compose([
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(10,30), p=0.2),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])

class SegDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.img_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.ids = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png','.jpg','.jpeg'))])
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        name = self.ids[idx]
        img = np.array(Image.open(os.path.join(self.img_dir, name)).convert('RGB'))
        mask = np.array(Image.open(os.path.join(self.mask_dir, name)))
        
        # Convert mask
        new_mask = np.zeros_like(mask, dtype=np.int64)
        for raw, cls in VALUE_MAP.items():
            new_mask[mask == raw] = cls
        
        aug = self.transform(image=img, mask=new_mask)
        return aug['image'], aug['mask'].long()

# ============================================================================
# MODEL
# ============================================================================

class SegHead(nn.Module):
    def __init__(self, in_ch, n_cls, th, tw):
        super().__init__()
        self.th, self.tw = th, tw
        h = 512 if in_ch >= 768 else 256
        
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, h, 1), nn.BatchNorm2d(h), nn.GELU(),
            nn.Conv2d(h, h, 3, padding=1, groups=h), nn.Conv2d(h, h, 1), nn.BatchNorm2d(h), nn.GELU(),
            nn.Conv2d(h, h, 3, padding=1, groups=h), nn.Conv2d(h, h, 1), nn.BatchNorm2d(h), nn.GELU(),
            nn.Conv2d(h, h//2, 3, padding=1), nn.BatchNorm2d(h//2), nn.GELU(), nn.Dropout2d(0.1),
            nn.Conv2d(h//2, n_cls, 1),
        )
        
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.th, self.tw, C).permute(0,3,1,2)
        return self.net(x)

# ============================================================================
# LOSS
# ============================================================================

class Loss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.w = weights
        
    def forward(self, p, t):
        ce = self.ce(p, t)
        # Dice
        probs = F.softmax(p, dim=1)
        t_oh = F.one_hot(t, p.shape[1]).permute(0,3,1,2).float()
        inter = (probs * t_oh).sum(dim=(0,2,3))
        union = (probs + t_oh).sum(dim=(0,2,3))
        dice = (2*inter + 1e-6) / (union + 1e-6)
        if self.w is not None:
            dice_loss = 1 - (dice * self.w.to(dice.device)).sum() / self.w.sum()
        else:
            dice_loss = 1 - dice.mean()
        return ce + 0.5 * dice_loss

# ============================================================================
# UTILS
# ============================================================================

def get_class_weights(data_dir):
    print("Computing class weights...")
    mask_dir = os.path.join(data_dir, 'Segmentation')
    files = [f for f in os.listdir(mask_dir) if f.endswith('.png')][:50]
    counts = Counter()
    for f in files:
        m = np.array(Image.open(os.path.join(mask_dir, f)))
        for raw, cls in VALUE_MAP.items():
            counts[cls] += (m == raw).sum()
    total = sum(counts.values())
    w = []
    for i in range(N_CLASSES):
        freq = counts.get(i, 1) / total
        w.append((1/(freq+1e-6))**0.5)
    w = np.array(w)
    w = w / w.sum() * N_CLASSES
    return torch.tensor(w, dtype=torch.float32)

def compute_iou(pred, target):
    pred = pred.argmax(dim=1).view(-1)
    target = target.view(-1)
    ious = []
    for c in range(N_CLASSES):
        p, t = pred==c, target==c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        if union > 0:
            ious.append((inter/union).item())
    return np.mean(ious) if ious else 0

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*50)
    print("FINAL TRAINING")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Auto-adjust for GPU memory
    if device.type == 'cuda':
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f}GB")
        global BATCH_SIZE, IMG_H, IMG_W
        if vram < 10:
            BATCH_SIZE, IMG_H, IMG_W = 4, 266, 476
        elif vram < 16:
            BATCH_SIZE, IMG_H, IMG_W = 6, 518, 952
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Data
    train_ds = SegDataset(DATA_DIR, get_transforms(IMG_H, IMG_W, train=True))
    val_ds = SegDataset(VAL_DIR, get_transforms(IMG_H, IMG_W, train=False))
    train_ld = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Data: {len(train_ds)} train, {len(val_ds)} val")
    
    # Backbone
    print("Loading DINOv2-base...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    
    # Get dims
    with torch.no_grad():
        dummy = torch.randn(1,3,IMG_H,IMG_W).to(device)
        feat = backbone.forward_features(dummy)["x_norm_patchtokens"]
        n_emb, th, tw = feat.shape[2], IMG_H//14, IMG_W//14
    print(f"Tokens: {th}x{tw}, Emb: {n_emb}")
    
    # Model
    model = SegHead(n_emb, N_CLASSES, th, tw).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    weights = get_class_weights(DATA_DIR).to(device)
    criterion = Loss(weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler()
    
    best_iou = 0
    history = {'loss':[], 'iou':[]}
    
    print(f"\nTraining {EPOCHS} epochs...")
    print("="*50)
    
    for ep in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_ld, desc=f"Ep{ep+1} Train", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with autocast():
                with torch.no_grad():
                    feat = backbone.forward_features(imgs)["x_norm_patchtokens"]
                out = model(feat)
                out = F.interpolate(out, size=imgs.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(out, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        train_loss /= len(train_ld)
        
        # Val with TTA
        model.eval()
        val_iou = 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_ld, desc=f"Ep{ep+1} Val", leave=False):
                imgs, masks = imgs.to(device), masks.to(device)
                
                with autocast():
                    # Original
                    f1 = backbone.forward_features(imgs)["x_norm_patchtokens"]
                    o1 = F.interpolate(model(f1), size=imgs.shape[2:], mode='bilinear', align_corners=False)
                    
                    # Flip
                    f2 = backbone.forward_features(torch.flip(imgs, [3]))["x_norm_patchtokens"]
                    o2 = torch.flip(F.interpolate(model(f2), size=imgs.shape[2:], mode='bilinear', align_corners=False), [3])
                    
                    out = (o1 + o2) / 2
                
                val_iou += compute_iou(out, masks)
        
        val_iou /= len(val_ld)
        scheduler.step()
        
        history['loss'].append(train_loss)
        history['iou'].append(val_iou)
        
        print(f"Epoch {ep+1}/{EPOCHS} | Loss: {train_loss:.4f} | mIoU: {val_iou:.4f}", end="")
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best.pth'))
            print(" *BEST*")
        else:
            print()
    
    # Save final
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final.pth'))
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].plot(history['loss'], 'b-')
    ax[0].set_title('Loss')
    ax[0].grid(True)
    ax[1].plot(history['iou'], 'g-')
    ax[1].axhline(best_iou, color='r', linestyle='--')
    ax[1].set_title(f'mIoU (Best: {best_iou:.4f})')
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'curves.png'))
    plt.close()
    
    print("\n" + "="*50)
    print(f"DONE! Best mIoU: {best_iou:.4f}")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
