"""
Resume Training - SAME Architecture as Original
- Loads your existing .pth checkpoint
- Keeps original SegmentationHeadConvNeXt architecture
- Adds class weighting + better loss
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
import warnings
warnings.filterwarnings('ignore')

plt.switch_backend('Agg')

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

CONFIG = {
    # Paths
    'data_dir': r'C:\Users\ASUS\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train',
    'val_dir': r'C:\Users\ASUS\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val',
    'output_dir': r'Z:\train_output_resume',
    
    # ========== CHECKPOINT TO RESUME FROM ==========
    'resume_checkpoint': r'Z:\train_output\best_model.pth',  # YOUR EXISTING .pth FILE
    # ================================================
    
    # Model (must match your checkpoint)
    'backbone_size': 'small',
    
    # Training  
    'batch_size': 2,
    'accumulation_steps': 4,
    'n_epochs': 100,  # Total epochs (will continue from checkpoint)
    'lr': 5e-4,  # Lower LR for fine-tuning
    'weight_decay': 1e-4,
    
    # Image size (must match your checkpoint)
    'img_height': 266,
    'img_width': 476,
    
    # Loss - simplified
    'use_class_weights': True,
    
    # Early stopping
    'patience': 20,
    
    # Mixed precision
    'use_amp': True,
}

# Class mapping
VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10,
}

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

N_CLASSES = len(VALUE_MAP)

# ============================================================================
# Class Weights
# ============================================================================

def compute_class_weights(data_dir, num_samples=50):
    print("Computing class weights...")
    mask_dir = os.path.join(data_dir, 'Segmentation')
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))][:num_samples]
    
    pixel_counts = Counter()
    for mf in tqdm(mask_files, desc="Scanning"):
        mask = np.array(Image.open(os.path.join(mask_dir, mf)))
        for raw_val, idx in VALUE_MAP.items():
            pixel_counts[idx] += np.sum(mask == raw_val)
    
    total = sum(pixel_counts.values())
    weights = []
    for i in range(N_CLASSES):
        freq = pixel_counts.get(i, 1) / total
        w = (1.0 / (freq + 1e-6)) ** 0.5  # sqrt inverse frequency
        weights.append(w)
        print(f"  {CLASS_NAMES[i]:15s}: {freq*100:5.2f}% -> w={w:.2f}")
    
    weights = np.array(weights)
    weights = weights / weights.sum() * N_CLASSES
    return torch.tensor(weights, dtype=torch.float32)

# ============================================================================
# Dataset
# ============================================================================

def get_train_transforms(h, w):
    return A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.1, rotate_limit=8, p=0.4),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 30), p=1),
            A.GaussianBlur(blur_limit=3, p=1),
        ], p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(h, w):
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def convert_mask(mask):
    out = np.zeros_like(mask, dtype=np.int64)
    for raw, idx in VALUE_MAP.items():
        out[mask == raw] = idx
    return out

class OffroadDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.ids = [f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        name = self.ids[idx]
        img = np.array(Image.open(os.path.join(self.img_dir, name)).convert('RGB'))
        mask = np.array(Image.open(os.path.join(self.mask_dir, name)))
        mask = convert_mask(mask)
        
        if self.transform:
            t = self.transform(image=img, mask=mask)
            img, mask = t['image'], t['mask']
        return img, mask.long()

# ============================================================================
# Model - ORIGINAL ARCHITECTURE (matches your checkpoint)
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    """EXACT same architecture as your original checkpoint."""
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )
        
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )
        
        self.classifier = nn.Conv2d(128, out_channels, 1)
        
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)

# ============================================================================
# Loss - Weighted CE + Dice
# ============================================================================

class WeightedCEDiceLoss(nn.Module):
    def __init__(self, class_weights=None, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_weight = dice_weight
        self.class_weights = class_weights
        
    def dice_loss(self, pred, target):
        num_classes = pred.shape[1]
        probs = F.softmax(pred, dim=1)
        target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        inter = (probs * target_oh).sum(dim=(0, 2, 3))
        union = (probs + target_oh).sum(dim=(0, 2, 3))
        dice = (2 * inter + 1e-6) / (union + 1e-6)
        
        if self.class_weights is not None:
            dice = dice * self.class_weights.to(dice.device)
            return 1 - dice.sum() / self.class_weights.sum()
        return 1 - dice.mean()
    
    def forward(self, pred, target):
        ce = self.ce(pred, target)
        dice = self.dice_loss(pred, target)
        return ce + self.dice_weight * dice

# ============================================================================
# Metrics
# ============================================================================

def compute_iou_per_class(pred, target, n_classes):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    
    ious = {}
    for c in range(n_classes):
        p = pred == c
        t = target == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        ious[c] = (inter / union).item() if union > 0 else float('nan')
    return ious

def mean_iou(iou_dict):
    valid = [v for v in iou_dict.values() if not np.isnan(v)]
    return np.mean(valid) if valid else 0.0

# ============================================================================
# Training
# ============================================================================

def train_epoch(model, backbone, loader, criterion, optimizer, scaler, device, accum_steps, use_amp):
    model.train()
    total_loss = 0
    n = 0
    
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Train", leave=False)
    
    for i, (imgs, masks) in enumerate(pbar):
        imgs, masks = imgs.to(device), masks.to(device)
        
        with autocast(enabled=use_amp):
            with torch.no_grad():
                feat = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(feat)
            out = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(out, masks) / accum_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps
        n += 1
        pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")
    
    return total_loss / n

@torch.no_grad()
def validate(model, backbone, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0
    all_iou = {i: [] for i in range(N_CLASSES)}
    n = 0
    
    for imgs, masks in tqdm(loader, desc="Val", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        
        with autocast(enabled=use_amp):
            feat = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(feat)
            out = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(out, masks)
        
        total_loss += loss.item()
        
        iou = compute_iou_per_class(out, masks, N_CLASSES)
        for c, v in iou.items():
            if not np.isnan(v):
                all_iou[c].append(v)
        n += 1
    
    avg_iou = {c: np.mean(v) if v else 0 for c, v in all_iou.items()}
    return total_loss / n, mean_iou(avg_iou), avg_iou

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("RESUME TRAINING (Original Architecture)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Class weights
    class_weights = None
    if CONFIG['use_class_weights']:
        class_weights = compute_class_weights(CONFIG['data_dir']).to(device)
    
    # Data
    train_ds = OffroadDataset(CONFIG['data_dir'], get_train_transforms(CONFIG['img_height'], CONFIG['img_width']))
    val_ds = OffroadDataset(CONFIG['val_dir'], get_val_transforms(CONFIG['img_height'], CONFIG['img_width']))
    
    train_loader = DataLoader(train_ds, CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Data: {len(train_ds)} train, {len(val_ds)} val")
    
    # Backbone
    print("Loading DINOv2...")
    backbone_map = {"small": "vits14", "base": "vitb14", "large": "vitl14"}
    backbone = torch.hub.load("facebookresearch/dinov2", f"dinov2_{backbone_map[CONFIG['backbone_size']]}")
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    
    # Get dims
    with torch.no_grad():
        dummy = torch.randn(1, 3, CONFIG['img_height'], CONFIG['img_width']).to(device)
        feat = backbone.forward_features(dummy)["x_norm_patchtokens"]
        n_emb = feat.shape[2]
        token_h = CONFIG['img_height'] // 14
        token_w = CONFIG['img_width'] // 14
    
    # Model - ORIGINAL ARCHITECTURE
    model = SegmentationHeadConvNeXt(
        in_channels=n_emb,
        out_channels=N_CLASSES,
        tokenW=token_w,
        tokenH=token_h,
    ).to(device)
    
    # Load checkpoint
    start_epoch = 0
    best_iou = 0.0
    
    if CONFIG['resume_checkpoint'] and os.path.exists(CONFIG['resume_checkpoint']):
        print(f"\nLoading checkpoint: {CONFIG['resume_checkpoint']}")
        ckpt = torch.load(CONFIG['resume_checkpoint'], map_location=device, weights_only=False)
        
        # Get state dict
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
        
        model.load_state_dict(state)
        print("  Weights loaded!")
        
        start_epoch = ckpt.get('epoch', 0)
        best_iou = ckpt.get('best_iou', 0.0)
        print(f"  Resuming from epoch {start_epoch}, best mIoU: {best_iou:.4f}")
    else:
        print("No checkpoint found, training from scratch")
    
    # Loss & Optimizer
    criterion = WeightedCEDiceLoss(class_weights=class_weights, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    # Scheduler - reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    scaler = GradScaler(enabled=CONFIG['use_amp'])
    
    # History
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'iou_per_class': []}
    patience_counter = 0
    
    print(f"\nTraining epochs {start_epoch + 1} to {CONFIG['n_epochs']}")
    print("=" * 60)
    
    for epoch in range(start_epoch, CONFIG['n_epochs']):
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch + 1}/{CONFIG['n_epochs']} (LR: {lr:.2e})")
        
        train_loss = train_epoch(
            model, backbone, train_loader, criterion, optimizer,
            scaler, device, CONFIG['accumulation_steps'], CONFIG['use_amp']
        )
        
        val_loss, val_iou, iou_per_class = validate(
            model, backbone, val_loader, criterion, device, CONFIG['use_amp']
        )
        
        scheduler.step(val_iou)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['iou_per_class'].append(iou_per_class)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | mIoU: {val_iou:.4f}")
        
        # Print per-class IoU for weak classes
        weak = [(CLASS_NAMES[c], v) for c, v in iou_per_class.items() if v < 0.4]
        if weak:
            print(f"  Weak classes: {', '.join([f'{n}:{v:.2f}' for n,v in weak])}")
        
        # Save best
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_iou': best_iou,
            }, os.path.join(CONFIG['output_dir'], 'best_model.pth'))
            print(f"  *** NEW BEST: {best_iou:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print("\nEarly stopping!")
                break
    
    # Save final
    torch.save({'model_state_dict': model.state_dict()}, 
               os.path.join(CONFIG['output_dir'], 'final_model.pth'))
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], 'g-')
    plt.axhline(best_iou, color='r', linestyle='--')
    plt.title(f'Val mIoU (Best: {best_iou:.4f})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'training_curves.png'))
    plt.close()
    
    print("\n" + "=" * 60)
    print(f"DONE! Best mIoU: {best_iou:.4f}")
    print(f"Output: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()
