import argparse
import itertools
import json
import logging
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, hf_hub_download, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from torch.utils.data import Sampler
from torch.cuda.amp import autocast, GradScaler

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EDMEulerScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.utils import (
    check_min_version,
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange

from mvadapter.utils import (
    get_orthogonal_camera,
    get_plucker_embeds_from_cameras_ortho,
    make_image_grid,
)

logger = get_logger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default="madebyollin/sdxl-vae-fp16-fix",
        required=False,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora_output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=800)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=800,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--regularization_lambda",
        type=float,
        default=0.001,
        help="Regularization lambda to use for the LoRA layers.",
    )
    parser.add_argument(
        "--clip_skip",
        type=int,
        default=None,
        help="Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that "
        "the output of the pre-final layer will be used for computing the prompt embeddings.",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--promptpath",
        type=str,
        default="prompt.json",
        help=(
            "location of prompt.json"
        ),
    )
    parser.add_argument(
        "--trainids",
        type=str,
        default="train_ids.json",
        help=(
            "location of object"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

# tokenize prompt
def tokenize_prompt(tokenizer, prompt, add_special_tokens=False):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# encoder prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None, clip_skip=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        if clip_skip is None:
            prompt_embeds = prompt_embeds[-1][-2]
        else:
            # "2" because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds[-1][-(clip_skip + 2)]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def replace_copy_weights(model: torch.nn.Module, verbose=True):
    param_dict = dict(model.named_parameters())

    for name, param in param_dict.items():
        if 'copy' in name and 'base_layer' not in name and 'lora_B' not in name:
            parts = name.split('.')
            for i, part in enumerate(parts):
                if 'copy' in part:
                    parts[i] = part.split('_copy')[0]  # 去掉 _copy 及后缀
                    break
            target_name = '.'.join(parts)

            if target_name in param_dict:
                with torch.no_grad():
                    param.copy_(param_dict[target_name])
                if verbose:
                    print(f"✅ Replaced: {name} <-- {target_name}")
            else:
                if verbose:
                    print(f"⚠️ Skipped (no source found): {name}")


def main(args):
    
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                    "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
    from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
    from peft import LoraConfig, get_peft_model
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    pipe: MVAdapterI2MVSDXLPipeline
    pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(args.pretrained_model_name_or_path)
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=DDPMScheduler
    )
    pipe.init_custom_adapter(num_views=6)
    pipe.load_custom_adapter(
        "huanngzh/mv-adapter", weight_name="mvadapter_i2mv_sdxl.safetensors"
    )
    pipe.to(device=accelerator.device, dtype=weight_dtype)
    pipe.cond_encoder.to(device=accelerator.device, dtype=weight_dtype)
    unet = pipe.unet 
    tokenizer_one = pipe.tokenizer
    tokenizer_two = pipe.tokenizer_2
    text_encoder_one = pipe.text_encoder
    text_encoder_two = pipe.text_encoder_2
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_name_or_path)
        
        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    pipe.cond_encoder.requires_grad_(False)
    pipe.copy_mv_linear_adapter()
    
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
      
    def filter_layers(model):
        filtered_layers = []
        for name, module in model.named_modules():
            if ("mv" in name and "to_out_mv" not in name) or ("to_out_mv.0" in name):
                filtered_layers.append(name)
        return filtered_layers
    target_modules = filter_layers(unet)
    unet_lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    # unet_lora_config = LoraConfig(
    #     r=32,
    #     lora_alpha=16,
    #     inference_mode=False,
    #     target_modules=target_modules,
    #     lora_nums=4,
    #     lora_dropout=0.0,
    # )
    unet = get_peft_model(unet, unet_lora_config)
    replace_copy_weights(pipe.unet)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
    unet_lora_parameters_with_lr = {"params": unet_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [unet_lora_parameters_with_lr]
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
        
    from data.mv_adapter_dataset import MVAdapterDataset
    # Dataset and DataLoaders creation:
    train_dataset = MVAdapterDataset("./dataset", args.promptpath, args.trainids)
    
    print(len(train_dataset))
    if len(train_dataset) == 1:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            num_workers=8,
            pin_memory=True
        )
        args.gradient_accumulation_steps = 1
        
    else:
        class CustomSampler(Sampler):
            def __init__(self, dataset, specified_index, batch_size):
                self.dataset = dataset
                self.specified_index = specified_index
                self.batch_size = batch_size
            def __iter__(self):
                indices = list(range(len(self.dataset)))
                indices.remove(self.specified_index)
                random.shuffle(indices)
                result = []
                for i in range(len(indices)):
                    for j in range(self.batch_size - 1):
                        result.append(self.specified_index)
                    result.append(indices[i])
                for i in result:
                    yield i
            def __len__(self):
                return (len(self.dataset) - 1) * self.batch_size
        sampler = CustomSampler(train_dataset, specified_index=0, batch_size=2)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            sampler=sampler,
            num_workers=8,
            pin_memory=True
        )

    def compute_time_ids(crops_coords_top_left, original_size=None):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]
    def compute_text_embeddings(prompt, text_encoders, tokenizers, clip_skip):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, clip_skip)
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # for name, param in unet.named_parameters():
    #     if param.dtype == torch.float16:
    #         print(f"{name}: dtype={param.dtype}")
    #     if param.dtype == torch.float32:
    #         print(f"{name}: dtype={param.dtype}")
    unet, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, vae, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = ("lora mvadapter")
        tracker_config = {}
        accelerator.init_trackers(tracker_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # for name, param in unet.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: requires_grad={param.requires_grad}")
    # for name, param in pipe.cond_encoder.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: requires_grad={param.requires_grad}")
    for name, param in unet.named_parameters():
        if torch.isnan(param).any():
            print(f"🚨 NaN detected in {name}")
        if torch.isinf(param).any():
            print(f"⚠️ Inf detected in {name}")
        # if param.dtype == torch.float32:
        #     print(f"{name}: dtype={param.dtype}")
            
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                prompts = batch["prompts"]
                control_images = batch["control_images"].squeeze(0)
                # print(batch["prompts"], batch["index"])
                # with accelerator.autocast():
                with torch.no_grad():
                    imgs_in, imgs_out = batch['imgs_in'], batch['imgs_out']
                    reference_imgs = batch['reference_imgs']
                    imgs_in, imgs_out = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W"), rearrange(imgs_out, "B Nv C H W -> (B Nv) C H W")
                    imgs_in, imgs_out = imgs_in.to(weight_dtype), imgs_out.to(weight_dtype)
                    model_input = vae.encode(imgs_in).latent_dist.sample()
                    model_input = model_input * pipe.vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    noise[0] = torch.zeros_like(noise[0])
                    bsz = model_input.shape[0]

                    timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (bsz // 6,), device=model_input.device).repeat_interleave(6)
                    timesteps = timesteps.long()
                    noisy_model_input = pipe.scheduler.add_noise(model_input, noise, timesteps)
                    
                    reference_img = Image.fromarray(reference_imgs[0].cpu().permute(1, 2, 0).numpy().astype(np.uint8))
                    pipe_kwargs = {}
                    pipe_kwargs["generator"] = torch.Generator(device=accelerator.device).manual_seed(0)
                    timesteps = torch.tensor([timesteps[0]]).to(device=accelerator.device)
                
                model_pred = pipe.train_forward(
                    prompts,
                    height=768,
                    width=768,
                    num_images_per_prompt=6,
                    control_image=control_images,
                    control_conditioning_scale=1.0,
                    reference_image=reference_img,
                    latents = noisy_model_input,
                    timesteps = timesteps,
                    reference_conditioning_scale=1.0,
                    negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
                    cross_attention_kwargs={"scale": 1.0},
                    accelerate = accelerator,
                    **pipe_kwargs,
                )
                # print(f"model_pred: {model_pred}")

                if pipe.scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif pipe.scheduler.config.prediction_type == "v_prediction":
                    target = pipe.scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {pipe.scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr_timesteps = timesteps

                    snr = compute_snr(pipe.scheduler, snr_timesteps)
                    base_weight = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(snr_timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * base_weight
                    loss = loss.mean()
                    
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                
                print(f"loss : {avg_loss}")
                if torch.isnan(avg_loss):
                    print("🚨 NaN detected in loss!")
                if torch.isinf(avg_loss):
                    print("⚠️ Inf detected in loss!")
                    
                if batch["index"] == 0:
                    avg_loss = avg_loss
                else:
                    avg_loss = avg_loss * args.regularization_lambda
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(avg_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
            replace_copy_weights(pipe.unet, False)
            
            if global_step % 100 == 0 and global_step > 0:
                unet.save_pretrained(os.path.join(args.output_dir, "pipeckpts_" + str(global_step)))

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        os.makedirs(os.path.join(args.output_dir, "pipeckpts"), exist_ok=True)
        unet.save_pretrained(os.path.join(args.output_dir, "pipeckpts"))

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)