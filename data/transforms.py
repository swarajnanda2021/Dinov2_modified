"""
Data augmentation transforms for DINOv2 training on pathology images.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image


class TMEDinoTransforms(object):
    """
    DINO-style augmentation transforms for pathology images.
    
    Args:
        local_size: Size of local crops
        global_size: Size of global crops
        local_crop_scale: Scale range for local crops
        global_crop_scale: Scale range for global crops
        n_local_crops: Number of local crops
        mean: Normalization mean
        std: Normalization std
    """
    def __init__(
        self,
        local_size=96,
        global_size=224,
        local_crop_scale=(0.05, 0.4),
        global_crop_scale=(0.4, 1.0),
        n_local_crops=2,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.n_local_crops = n_local_crops
        self.global_size = global_size
        self.local_size = local_size
        self.local_crop_scale = local_crop_scale
        self.global_crop_scale = global_crop_scale
        self.mean = mean
        self.std = std
        
        # Basic transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Compose([
            self.to_tensor,
            transforms.Normalize(mean=mean, std=std),
        ])
        
        # DINO color augmentation
        self.flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.01),
        ])

        # Global view 1
        self.global_1 = transforms.Compose([
            transforms.Resize((global_size, global_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(size=global_size, scale=global_crop_scale, interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            transforms.GaussianBlur(3, (0.1, 0.15)),
            self.to_tensor,
            transforms.Normalize(mean=mean, std=std),
        ])

        # Global view 2
        self.global_2 = transforms.Compose([
            transforms.Resize((global_size, global_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(size=global_size, scale=global_crop_scale, interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.15)),
            transforms.RandomSolarize(threshold=64, p=0.5),
            self.to_tensor,
            transforms.Normalize(mean=mean, std=std),
        ])

        # Local crops
        self.local = transforms.Compose([
            transforms.Resize((global_size, global_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(size=local_size, scale=local_crop_scale, interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            transforms.GaussianBlur(3, (0.1, 0.15)),
            self.to_tensor,
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, x):
        """
        Apply all augmentation transforms.
        
        Returns:
            List of augmented crops: [global1, global2, local1, ..., localN]
        """
        crops = []
        
        # DINO global views
        crops.append(self.global_1(x))
        crops.append(self.global_2(x))
        
        # Local crops
        for _ in range(self.n_local_crops):
            crops.append(self.local(x))
        
        return crops