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
    lora_name,
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
    
    with open("./unet.txt", "w") as f:
        for name, module in pipe.unet.named_modules():
            f.write(f"{name}: {module.__class__.__name__}\n")
        

    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)
    pipe.copy_mv_linear_adapter()

    from peft import PeftModel, PeftConfig
    config = lora_name
    pipeline_unet = PeftModel.from_pretrained(pipe.unet, config)
    pipe.unet = pipeline_unet
    
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
    # H, W = alpha.shape
    # get the bounding box of alpha
    # y, x = np.where(alpha)
    # y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
    # x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
    # y0, y1 = 0, H
    # x0, x1 = 0, W
    # image_center = image[y0:y1, x0:x1]
    # resize the longer side to H * 0.9
    # H, W, _ = image_center.shape
    # if H > W:
    #     W = int(W * (height * 0.9) / H)
    #     H = int(height * 0.9)
    # else:
    #     H = int(H * (width * 0.9) / W)
    #     W = int(width * 0.9)
    image_center = np.array(Image.fromarray(image))
    # pad to H, W
    # start_h = (height - H) // 2
    # start_w = (width - W) // 2
    # image = np.zeros((height, width, 4), dtype=np.uint8)
    image = image_center
    image = image.astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = (image * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)

    return image

def preprocess_image_white_bg(image: Image.Image, height, width):
    image = np.array(image)
    alpha = image[..., 3] > 0
    # H, W = alpha.shape
    # # get the bounding box of alpha
    # # y, x = np.where(alpha)
    # # y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
    # # x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
    # y0, y1 = 0, H
    # x0, x1 = 0, W
    # image_center = image[y0:y1, x0:x1]
    # # resize the longer side to H * 0.9
    # H, W, _ = image_center.shape
    # if H > W:
    #     W = int(W * (height * 0.9) / H)
    #     H = int(height * 0.9)
    # else:
    #     H = int(H * (width * 0.9) / W)
    #     W = int(width * 0.9)
    image_center = np.array(Image.fromarray(image))
    # pad to H, W
    # start_h = (height - H) // 2
    # start_w = (width - W) // 2
    # image = np.zeros((height, width, 4), dtype=np.uint8)
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

    # import PIL
    # import os
    # # import numpy as np
    # def _load_image(path):
    #     image = PIL.Image.open(path)
    #     image = np.array(image)
    #     image = image.astype(np.float32) / 255.0
    #     image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    #     image = (image * 255).clip(0, 255).astype(np.uint8) / 255.0
    #     return image
    # transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5], [0.5])
    #     ])
    # object_dirs = "./dataset/girl"
    # images = [f for f in os.listdir(object_dirs) if f.endswith(".png")]
    # images = sorted(images)
    # imgs_in = [transform(_load_image(os.path.join(object_dirs, f)).astype(np.float32)) for f in images]
    # imgs_in = torch.stack(imgs_in, dim=0).to(device=device, dtype=torch.float16)
    # latents = pipe.vae.encode(imgs_in).latent_dist.mode() * pipe.vae.config.scaling_factor
    # # latents_save = latents.detach().view(latents.shape[0], -1)
    # # np.savetxt('inference.txt', latents_save.cpu().numpy(), fmt='%.4f')
    # noise = torch.randn_like(latents)
    # timesteps = torch.randint(2, 3, (1, ), device=latents.device).repeat_interleave(6)
    # timesteps = timesteps.long()
    # latents = pipe.scheduler.add_noise(latents, noise, timesteps)
    
    # timesteps = torch.tensor([181, 161,
    #     141, 121, 101,  81,  61,  41,  21,   1], device="cpu")
    
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
        # latents = latents,
        # timesteps = timesteps,
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
        "--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
    )
    parser.add_argument(
        "--vae_model", type=str, default="madebyollin/sdxl-vae-fp16-fix"
    )
    parser.add_argument("--unet_model", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default="huanngzh/mv-adapter")
    parser.add_argument("--num_views", type=int, default=6)
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    # Inference
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--text", type=str, default="high quality")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--lora_name", type=str, default="./lora_output/pipeckpts")
    args = parser.parse_args()

    print(args.lora_name)
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
        lora_name=args.lora_name,
    )

    if args.remove_bg:
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
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
