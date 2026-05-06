import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

try:
    from model import StarKAN
except ImportError:
    print("Error: model.py not found. Please check the file location.")
    exit()

from dataset import CrackDataset  # replace with your dataset class

CONFIG = {
    'project_name': 'StarKAN_DeepCrack',
    'root_dir': './DeepCrack',
    'img_size': 320,
    'batch_size': 8,
    'epochs': 120,
    'lr': 1e-3,
    'weight_decay': 2e-2,
    'device': 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
}


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-5
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_score = (2 * intersection + smooth) / (union + smooth)
        return 0.5 * bce + 0.5 * (1 - dice_score.mean())


class SoftCLDiceLoss(nn.Module):
    """
    Soft-CLDice loss for topology-preserving segmentation.
    Reference: Shit et al., CVPR 2021.
    """
    def __init__(self, iter_=3, smooth=1.):
        super().__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        def soft_skel(img, iter_):
            img1 = F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
            skel = F.relu(img - img1)
            return skel

        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)


def validate(model, loader, criterion):
    """Evaluate model using global pixel-level metrics."""
    model.eval()

    total_inter = 0
    total_union = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(CONFIG['device'])
            masks = batch['mask'].to(CONFIG['device'])

            out = model(imgs)
            loss = criterion(out, masks)
            total_loss += loss.item()
            num_batches += 1

            pred = (torch.sigmoid(out) > 0.5).float()

            inter = (pred * masks).sum().item()
            union = pred.sum().item() + masks.sum().item() - inter
            tp = inter
            fp = pred.sum().item() - tp
            fn = masks.sum().item() - tp

            total_inter += inter
            total_union += union
            total_tp += tp
            total_fp += fp
            total_fn += fn

    global_iou = (total_inter + 1e-6) / (total_union + 1e-6)
    precision = (total_tp + 1e-6) / (total_tp + total_fp + 1e-6)
    recall = (total_tp + 1e-6) / (total_tp + total_fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    return global_iou


def main():
    set_seed()
    save_dir = f"./checkpoints/{CONFIG['project_name']}"
    os.makedirs(save_dir, exist_ok=True)

    train_ds = CrackDataset(CONFIG['root_dir'], split='train', img_size=CONFIG['img_size'])
    val_ds = CrackDataset(CONFIG['root_dir'], split='test', img_size=CONFIG['img_size'], is_training=False)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    model = StarKAN(n_classes=1).to(CONFIG['device'])
    print(f"Device: {CONFIG['device']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    criterion_base = BCEDiceLoss().to(CONFIG['device'])
    criterion_topo = SoftCLDiceLoss().to(CONFIG['device'])

    best_iou = 0.0
    print(f"Starting training for {CONFIG['epochs']} epochs...")

    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            imgs = batch['image'].to(CONFIG['device'])
            masks = batch['mask'].to(CONFIG['device'])

            optimizer.zero_grad()
            final, ds1, ds2 = model(imgs)

            loss_pixel = (criterion_base(final, masks) +
                          0.4 * criterion_base(ds1, masks) +
                          0.3 * criterion_base(ds2, masks))
            loss_topo = criterion_topo(torch.sigmoid(final), masks)
            loss = loss_pixel + 0.3 * loss_topo

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        val_iou = validate(model, val_loader, criterion_base)
        scheduler.step()

        print(f"Epoch {epoch} | Val IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            print(f"Best IoU updated: {best_iou:.4f}")

    print(f"Training complete. Best IoU: {best_iou:.4f}")


if __name__ == '__main__':
    main()
