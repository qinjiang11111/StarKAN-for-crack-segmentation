import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob


class CrackDataset(Dataset):
    """
    Dataset loader for crack segmentation.
    Supports DeepCrack dataset structure (train_img/train_lab, test_img/test_lab).
    Training: random crop + geometric and color augmentations.
    Testing: center crop + normalization only.
    """
    def __init__(self, root_dir, split='train', img_size=320, is_training=True):
        self.split = split
        self.img_size = img_size
        self.is_training = is_training

        if split == 'train':
            self.img_paths = sorted(glob(os.path.join(root_dir, 'train_img', '*.jpg')) +
                                    glob(os.path.join(root_dir, 'train_img', '*.png')))
            self.mask_paths = sorted(glob(os.path.join(root_dir, 'train_lab', '*.png')) +
                                     glob(os.path.join(root_dir, 'train_lab', '*.jpg')))
        else:
            self.img_paths = sorted(glob(os.path.join(root_dir, 'test_img', '*.jpg')) +
                                    glob(os.path.join(root_dir, 'test_img', '*.png')))
            self.mask_paths = sorted(glob(os.path.join(root_dir, 'test_lab', '*.png')) +
                                     glob(os.path.join(root_dir, 'test_lab', '*.jpg')))

        if len(self.img_paths) != len(self.mask_paths):
            min_len = min(len(self.img_paths), len(self.mask_paths))
            self.img_paths = self.img_paths[:min_len]
            self.mask_paths = self.mask_paths[:min_len]

        print(f"[{split.upper()}] Loaded {len(self.img_paths)} image-mask pairs.")

        if is_training:
            self.transform = A.Compose([
                A.RandomCrop(height=img_size, width=img_size, p=1.0),
                A.HorizontalFlip(p=0.7),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.7),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.2),
                    contrast_limit=(-0.2, 0.2),
                    p=0.5
                ),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.3),
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0),
                    A.ChannelShuffle(p=1.0),
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=1.0),
                ], p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.CenterCrop(height=img_size, width=img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        img_name = os.path.basename(img_path)

        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: failed to load image {img_path}, using blank.")
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask / 255.0

        augmented = self.transform(image=image, mask=mask)
        img_tensor = augmented['image']
        mask_tensor = augmented['mask'].unsqueeze(0).float()  # [1, H, W]

        return {'image': img_tensor, 'mask': mask_tensor, 'img_name': img_name}
