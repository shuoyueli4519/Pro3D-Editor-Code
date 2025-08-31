import argparse

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, UNet2DConditionModel
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation

from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils import (
    get_orthogonal_camera,
    get_plucker_embeds_from_cameras_ortho,
    make_image_grid,
)


def prepare_pipeline(
    base_model,
    vae_model,
    unet_model,
    lora_model,
    adapter_path,
    scheduler,
    num_views,
    device,
    dtype,
):
    # Load vae and unet if provided
    pipe_kwargs = {}
    if vae_model is not None:
        pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model)
    if unet_model is not None:
        pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(unet_model)

    # Prepare pipeline
    pipe: MVAdapterI2MVSDXLPipeline
    pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(base_model, **pipe_kwargs)

    # Load scheduler if provided
    scheduler_class = None
    if scheduler == "ddpm":
        scheduler_class = DDPMScheduler
    elif scheduler == "lcm":
        scheduler_class = LCMScheduler

    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=scheduler_class,
    )
    pipe.init_custom_adapter(num_views=num_views)
    pipe.load_custom_adapter(
        adapter_path, weight_name="mvadapter_i2mv_sdxl.safetensors"
    )

    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)
    pipe.copy_mv_linear_adapter()
    
    def check_copies_success(model, target_keys=None, num_copies=6):
        if target_keys is None:
            target_keys = ['to_q_mv', 'to_k_mv', 'to_v_mv']  # 默认目标层
        
        # 遍历模型中所有模块
        for name, module in model.named_modules():
            print(f"now : {name}")
    
    # check_copies_success(pipe.unet)

    # load lora if provided
    if lora_model is not None:
        model_, name_ = lora_model.rsplit("/", 1)
        pipe.load_lora_weights(model_, weight_name=name_)

    # vae slicing for lower memory usage
    pipe.enable_vae_slicing()

    return pipe


def remove_bg(image, net, transform, device):
    image_size = image.size
    input_images = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = net(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image


def preprocess_image(image: Image.Image, height, width):
    image = np.array(image)
    alpha = image[..., 3] > 0
    image_center = np.array(Image.fromarray(image))
    image = image_center
    image = image.astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = (image * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)

    return image

def preprocess_image_white_bg(image: Image.Image, height, width):
    image = np.array(image)
    alpha = image[..., 3] > 0
    image_center = np.array(Image.fromarray(image))
    image = image_center
    image = image.astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 1.0
    image = (image * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)

    return image

def save_background_as_mask(image: Image.Image, height, width):
    if image.mode != "RGBA":
        mask = Image.new("L", image.size, 0)
    else:
        alpha = image.getchannel("A")
        mask = alpha.point(lambda p: 255 if p >= 200 else 0)
    return mask

def run_pipeline(
    pipe,
    num_views,
    text,
    image,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    seed,
    remove_bg_fn=None,
    reference_conditioning_scale=1.0,
    negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
    lora_scale=1.0,
    device="cuda",
):
    # Prepare cameras
    cameras = get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 0, 0],
        distance=[1.8] * num_views,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 45, 90, 180, 270, 315]],
        device=device,
    )

    plucker_embeds = get_plucker_embeds_from_cameras_ortho(
        cameras.c2w, [1.1] * num_views, width
    )
    control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1)

    # Prepare image
    reference_image = Image.open(image) if isinstance(image, str) else image
    if remove_bg_fn is not None:
        reference_image = remove_bg_fn(reference_image)
        reference_image = preprocess_image(reference_image, height, width)
    elif reference_image.mode == "RGBA":
        reference_image = preprocess_image(reference_image, height, width)

    pipe_kwargs = {}
    if seed != -1 and isinstance(seed, int):
        pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

    images = pipe(
        text,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_views,
        control_image=control_images,
        control_conditioning_scale=1.0,
        reference_image=reference_image,
        reference_conditioning_scale=reference_conditioning_scale,
        negative_prompt=negative_prompt,
        cross_attention_kwargs={"scale": lora_scale},
        **pipe_kwargs,
    ).images

    return images, reference_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Models
    parser.add_argument(
        "--base_model", type=str, default="../../huggingface/models--stabilityai--stable-diffusion-xl-base-1.0"
    )
    parser.add_argument(
        "--vae_model", type=str, default="../../huggingface/models--madebyollin--sdxl-vae-fp16-fix"
    )
    parser.add_argument("--unet_model", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default="../../huggingface/models--huanngzh--mv-adapter")
    parser.add_argument("--num_views", type=int, default=6)
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    # Inference
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--text", type=str, default="high quality")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--reference_conditioning_scale", type=float, default=1.0)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="watermark, ugly, deformed, noisy, blurry, low contrast",
    )
    parser.add_argument("--output", type=str, default="output.png")
    # Extra
    parser.add_argument("--remove_bg", action="store_true", help="Remove background")
    args = parser.parse_args()

    pipe = prepare_pipeline(
        base_model=args.base_model,
        vae_model=args.vae_model,
        unet_model=args.unet_model,
        lora_model=args.lora_model,
        adapter_path=args.adapter_path,
        scheduler=args.scheduler,
        num_views=args.num_views,
        device=args.device,
        dtype=torch.float16,
    )

    if args.remove_bg:
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            "../../huggingface/models--ZhengPeng7--BiRefNet", trust_remote_code=True
        )
        birefnet.to(args.device)
        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        remove_bg_fn = lambda x: remove_bg(x, birefnet, transform_image, args.device)
    else:
        remove_bg_fn = None

    images, reference_image = run_pipeline(
        pipe,
        num_views=args.num_views,
        text=args.text,
        image=args.image,
        height=768,
        width=768,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        lora_scale=args.lora_scale,
        reference_conditioning_scale=args.reference_conditioning_scale,
        negative_prompt=args.negative_prompt,
        device=args.device,
        remove_bg_fn=remove_bg_fn,
    )
    
    i = 0
    for image in images:
        image = remove_bg_fn(image)
        mask = save_background_as_mask(image, 768, 768)
        image = preprocess_image_white_bg(image, 768, 768)
        image.save(f"./results/images/{i:03}" + ".png")
        mask.save(f"./results/masks/{i:03}" + ".png")
        i += 1
        
    make_image_grid(images, rows=1).save(args.output)
    reference_image.save(args.output.rsplit(".", 1)[0] + "_reference.png")
