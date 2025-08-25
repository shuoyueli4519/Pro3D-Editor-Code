from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from typing import Literal, Tuple, Optional, Any
import cv2
from transformers import AutoModelForImageSegmentation
import random

import json
import os, sys
import math

import PIL.Image
import pdb

from mvadapter.utils import (
    get_orthogonal_camera,
    get_plucker_embeds_from_cameras_ortho,
    make_image_grid,
)

class MVAdapterDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 prompt_path: str,
                 trainids_path: str):
        super().__init__()
        self.base_path = Path(root_dir)
        self.trainids_path = trainids_path
        with open(os.path.join(root_dir, self.trainids_path)) as f:
            metadata = json.load(f)
        self.object_dirs = [os.path.join(self.base_path, d) for d in metadata]
        print(self.object_dirs)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.prompt_path = prompt_path
        with open(os.path.join(root_dir, self.prompt_path), 'r', encoding='utf-8') as file:
            self.data = json.load(file)

        # 预加载所有图像数据
        self.images_data = []
        for object_dir in self.object_dirs:
            images = [f for f in os.listdir(object_dir) if f.endswith(".png")]
            images = sorted(images)
            reference_img_path = os.path.join(object_dir, images[0])
            reference_img = self._load_image(reference_img_path)
            reference_img = torch.from_numpy(reference_img).permute(2, 0, 1).float()
            imgs_in = [self.transform(self._load_image(os.path.join(object_dir, f)).astype(np.float32) / 255.0) for f in images]
            self.images_data.append({
                "reference_img": reference_img,
                "imgs_in": torch.stack(imgs_in, dim=0).float()
            })
            
        cameras = get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 0, 0],
        distance=[1.8] * 6,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 45, 90, 180, 270, 315]],
        )

        plucker_embeds = get_plucker_embeds_from_cameras_ortho(
        cameras.c2w, [1.1] * 6, 768
        )
        self.control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1)
    
    def __len__(self):
        return len(self.object_dirs)
    
    def _load_image(self, path):
        image = PIL.Image.open(path)
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = (image * 255).clip(0, 255).astype(np.uint8)
        return image
    
    def __getitem__(self, index):
        prompt = self.data[0]  # 假设所有的样本使用相同的提示
        reference_img = self.images_data[index]["reference_img"]
        imgs_in = self.images_data[index]["imgs_in"]
        imgs_out = imgs_in.clone()  # 假设输出图像与输入图像相同
        control_images = self.control_images
        
        return {
            "imgs_in": imgs_in,
            "imgs_out": imgs_out,
            "prompts": prompt,
            "reference_imgs": reference_img,
            "control_images": control_images,
            "index": index
        }