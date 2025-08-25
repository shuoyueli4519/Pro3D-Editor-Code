import argparse

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, UNet2DConditionModel
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation
import os

def remove_bg(image, net, transform, device):
    image_size = image.size
    image = image.convert("RGB")
    input_images = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = net(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image

birefnet = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
birefnet.to("cuda")
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
remove_bg_fn = lambda x: remove_bg(x, birefnet, transform_image, "cuda")

images_file = [f for f in os.listdir("../dataset/rccar_fixed") if f.endswith(".png")]
for image_name in images_file:
    print(image_name)
    image_path = os.path.join("../dataset/rccar_fixed", image_name)
    image = Image.open(image_path)
    image = remove_bg_fn(image)
    image.save("../dataset/rccar_fixed/" + str(image_name.replace(".png", ".png")))