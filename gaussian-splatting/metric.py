import clip
import torch
from torch import Tensor
import os
from torch.backends import cudnn
from pathlib import Path
from PIL import Image
import math
from typing import Any
import torchvision.transforms
import lpips
from torch.nn.functional import l1_loss, mse_loss
import pytorch_fid.fid_score as fid
from transformers import AutoImageProcessor, AutoModel
INFINITY = 1e10

def mse2psnr(x: Any) -> Any:
    if isinstance(x, Tensor):
        dtype, device = x.dtype, x.device
        # fmt: off
        return (
            -10.0 * torch.log(x) / torch.log(torch.tensor([10.0], dtype=dtype, device=device))
            if x != 0.0 
            else torch.tensor([INFINITY], dtype=dtype, device=device)
        )
        # fmt: on
    else:
        return -10.0 * math.log(x) / math.log(10.0) if x != 0.0 else math.inf

# Age-old custom option for fast training :)
cudnn.benchmark = True
# Also set torch's multiprocessing start method to spawn
# refer -> https://github.com/pytorch/pytorch/issues/40403
# for more information. Some stupid PyTorch stuff to take care of

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Available CLiP models - {clip.available_models()}")
model, preprocess = clip.load("ViT-B/32", device=device)

dino_preprocess = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

# -------------------------------------------------------------------------------------
#  Command line configuration for the script                                          |
# -------------------------------------------------------------------------------------
# fmt: off
# noinspection PyUnresolvedReferences
prompt = [
        #   "The object features a robotic arm clawing a biscuits and an attached wooden barrel.",
        #   "The object features a robotic arm clawing a pizza and an attached container.",
          "The object features a cat in a car with wheels.",
        #   "The object features a pig in a car with wheels.",
        #   "The object features a pig wearing a white hat in a car with wheels.",
        #   "The object is a stylized figure with rabbit-like ears, wearing a red dress, and posed with arms extended.",
        #   "The object is a stylized figure with rabbit-like ears, wearing a pair of goggles, and posed with arms extended.",
        #   "The object is a stylized figure with rabbit-like ears, wearing a pair of blue shoes, and posed with arms extended.",
        #   "The object is a stylized figure resembling an anthropomorphic rabbit in knightly armor, holding a sword and well-decorated shield.",
        #   "The object is a stylized figure resembling an anthropomorphic rabbit in knightly armor, wearing a blue mask, holding a sword and plain shield.",
          "The object is a stylized character head with a mustache like red pepper.",
          "The object is a stylized character head with a cowboy hat and a red mustache.",
          "The object is a whimsical, animal-like device featuring a face on its screen and rabbit like ears, complete with a long, flexible tail extending from its top.",
          "The object is a whimsical, animal-like device featuring a face on its screen and ears, complete with a tail like a hammer.",
        ]
source_cases = ["rccar", "head", "head", "box", "box"]
# source_cases = ["mechine", "mechine", "rccar", "rccar", "rccar", "girl", "girl", "girl",  "knight_toy", "knight_toy", "head", "head", "box", "box"]
# edit_cases = ["mechine_barrel_1_6", "mechine_pizza_1_6", "cat_rccar_1_6", "pig_rccar_1_6", "pig_rccar_hat_1_6", "girl_red_dress_1_6", "girl_goggles_1_6", "girl_blue_shoes_1_6", "knight_toy_shield_1_6", "knight_toy_mask_1_6", "head_pepper_mustache_1_6", "head_cowboy_hat_1_6", "box_rabbit_ears_1_6", "hammer_like_tail_1_6"]
# edit_cases = ["mechine_barrel", "mechine_pizza", "cat_rccar", "pig_rccar", "pig_rccar_hat", "girl_red_dress", "girl_goggles", "girl_blue_shoes", "knight_toy_shield", "knight_toy_mask", "head_pepper_mustache", "head_cowboy_hat", "box_rabbit_ears", "hammer_like_tail"]
# edit_cases = ["mechine_barrel_3DAdapter", "mechine_pizza_3DAdapter", "cat_rccar_3DAdapter", "pig_rccar_3DAdapter", "pig_rccar_hat_3DAdapter", "girl_red_dress_3DAdapter", "girl_goggles_3DAdapter", "girl_blue_shoes_3DAdapter", "knight_toy_shield_3DAdapter", "knight_toy_mask_3DAdapter", "head_pepper_mustache_3DAdapter", "head_cowboy_hat_3DAdapter", "box_rabbit_ears_3DAdapter", "hammer_like_tail_3DAdapter"]
# edit_cases = ["mechine_barrel_MVEdit", "mechine_pizza_MVEdit", "cat_rccar_MVEdit", "pig_rccar_MVEdit", "pig_rccar_hat_MVEdit", "girl_red_dress_MVEdit", "girl_goggles_MVEdit", "girl_blue_shoes_MVEdit", "knight_toy_shield_MVEdit", "knight_toy_mask_MVEdit", "head_pepper_mustache_MVEdit", "head_cowboy_hat_MVEdit", "box_rabbit_ears_MVEdit", "hammer_like_tail_MVEdit"]
# edit_cases = ["mechine_barrel_tailor3d", "mechine_pizza_tailor3d", "cat_rccar_tailor3d", "pig_rccar_tailor3d", "pig_rccar_hat_tailor3d", "girl_red_dress_tailor3d", "girl_goggles_tailor3d", "girl_blue_shoes_tailor3d", "knight_toy_shield_tailor3d", "knight_toy_mask_tailor3d", "head_pepper_mustache_tailor3d", "head_cowboy_hat_tailor3d", "box_rabbit_ears_tailor3d", "hammer_like_tail_tailor3d"]
# edit_cases = ["mechine_barrel_norepair", "mechine_pizza_norepair", "cat_rccar_norepair", "pig_rccar_norepair", "pig_rccar_hat_norepair", "girl_red_dress_norepair", "girl_goggles_norepair", "girl_blue_shoes_norepair", "knight_toy_shield_norepair", "knight_toy_mask_norepair", "head_pepper_mustache_norepair", "head_cowboy_hat_norepair", "box_rabbit_ears_norepair", "hammer_like_tail_norepair"]
edit_cases = ["cat_rccar_fixed", "head_pepper_mustache_fixed", "head_cowboy_hat_fixed", "box_rabbit_ears_fixed", "hammer_like_tail_fixed"]

def main(**kwargs) -> None:
    # make dataframe array
    dataframes = []
    frame_titles = []

    i = 0
    for source_case, edit_case in zip(source_cases, edit_cases):
        # get ref images
        recon_path = "./output/" + source_case + "/train/ours_None/renders"
        recon_imgs = get_images(recon_path)
        
        out_path = "./output/" + edit_case + "/train/ours_None/renders"
        # out_path = "./output/" + edit_case
        lpips_score = get_LPIPS(recon_path, out_path)
        print(f"{edit_case} ours lpips_score : {lpips_score}")
        out_imgs = get_images(out_path)
        # print(len(out_imgs))
        psnr_recon = get_PSNRS(out_imgs, recon_imgs)
        print(f"{edit_case} ours psnr_recon : {psnr_recon}")
        # fid_score_recon = fid.calculate_fid_given_paths((out_path, recon_path),
        #                                            72,
        #                                            "cuda",
        #                                            2048,
        #                                            )
        out_features = get_Dino_im_features(out_imgs)
        recon_features = get_Dino_im_features(recon_imgs)
        dinoI = get_avg_DINO_img_sim(recon_features, out_features)
        print(f"{edit_case} ours dinoI : {dinoI}")
        # print(f"{edit_case} ours fid : {fid_score_recon}")
        clip_out_text_features = get_text_features(prompt[i])
        clip_out_img_features = get_CLIP_im_features(out_imgs)
        
        # record [Output - Text] CliP similarity
        out_text_similarity = get_avg_CLIP_text_sim(clip_out_img_features, clip_out_text_features)
        print(f"{edit_case} ours CLIPT : {out_text_similarity}")

        # # record directional CliP similarity
        # directional_similarity = get_avg_CLIP_directional_sim(clip_input_features, \
        #                                                         clip_recon_img_features, \
        #                                                         clip_out_text_features, \
        #                                                         clip_out_img_features)
        # print(directional_similarity)
        
        i = i + 1
        
def get_LPIPS(source_img_path, out_img_path):
    # 初始化 LPIPS 模型
    loss_fn = lpips.LPIPS(net='alex').cuda()
    
    # 图像预处理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    # 获取文件名列表并排序
    source_files = sorted(os.listdir(source_img_path))
    out_files = sorted(os.listdir(out_img_path))
    
    # 取最小长度，避免索引越界
    num_pairs = min(len(source_files), len(out_files))
    
    if num_pairs == 0:
        print("两个文件夹至少要各有一张图片。")
        return
    
    total_distance = 0.0

    for i in range(num_pairs):
        src_path = os.path.join(source_img_path, source_files[i])
        out_path = os.path.join(out_img_path, out_files[i])

        try:
            img0 = Image.open(src_path).convert('RGB')
            img1 = Image.open(out_path).convert('RGB')
        except Exception as e:
            print(f"跳过索引 {i} 的文件对，原因: {e}")
            continue

        img0_tensor = transform(img0).unsqueeze(0).cuda()
        img1_tensor = transform(img1).unsqueeze(0).cuda()

        with torch.no_grad():
            distance = loss_fn(img0_tensor, img1_tensor)
            total_distance += distance.item()

    avg_distance = total_distance / num_pairs
    print(f"平均 LPIPS 距离: {avg_distance:.4f}")
    return avg_distance


def get_PSNRS(out_imgs: tuple, ref_imgs):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Resize((400,400))])
    total_PSNR = 0.0
    with torch.no_grad():
        for out_img, ref_img in zip(out_imgs, ref_imgs):
            out_t = transform(out_img).reshape((-1, 3)).to(device)
            ref_t = transform(ref_img).reshape((-1, 3)).to(device)
            mse = mse_loss(out_t, ref_t)
            psnr = mse2psnr(mse)
            total_PSNR += psnr
    return (total_PSNR.item() / len(out_imgs))


def get_avg_CLIP_directional_sim(ref_txt_features: torch.Tensor, \
                                 ref_img_features: torch.Tensor, \
                                 out_txt_features: torch.Tensor, \
                                 out_img_features: torch.Tensor) -> float:
    total_sim = 0.0
    ref_txt_feat_normed = ref_txt_features / ref_txt_features.norm(dim=-1, keepdim=True)
    out_txt_feat_normed = out_txt_features / out_txt_features.norm(dim=-1, keepdim=True)
    text_dir = ref_txt_feat_normed - out_txt_feat_normed

    for out_im_feat, ref_im_feat in zip(out_img_features, ref_img_features):
        ref_im_feat_normed = ref_im_feat / ref_im_feat.norm(dim=-1, keepdim=True)
        out_im_feat_normed = out_im_feat / out_im_feat.norm(dim=-1, keepdim=True)
        im_dir = ref_im_feat_normed - out_im_feat_normed
        sim = (text_dir @ im_dir.T).item()
        total_sim += sim
    print(len(out_img_features))

    return total_sim


def get_avg_CLIP_text_sim(out_features: tuple, text_features: torch.Tensor) -> float:
    total_sim = 0.0
    target_feat_normed = text_features / text_features.norm(dim=-1, keepdim=True)
    for out_feat in out_features:
        out_feat_normed = out_feat / out_feat.norm(dim=-1, keepdim=True)
        sim = (out_feat_normed @ target_feat_normed.T).item()
        total_sim += sim
    print(len(out_features))
    return total_sim / len(out_features)

def get_avg_DINO_img_sim(source_img_features: tuple, out_img_features: tuple) -> float:
    total_sim = 0.0
    for source_feature, out_feature in zip(source_img_features, out_img_features):
        source_feature_normed = source_feature / source_feature.norm(dim=-1, keepdim=True)
        out_feature_normed = out_feature / out_feature.norm(dim=-1, keepdim=True)
        sim = (source_feature_normed @ out_feature_normed.T).item()
        total_sim += sim
    return total_sim / len(source_img_features)


def get_text_features(prompt: str):
    text = clip.tokenize(prompt).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features
    
    
def get_images(im_dir: Path) -> tuple:
    ims = []
    for name in os.listdir(im_dir):
        if not name.endswith('.png') and not name.endswith('.jpg'):
            continue
        im_path = os.path.join(im_dir, name)
        img = Image.open(im_path).convert("RGB") 
        ims.append(img)
    return ims


def get_CLIP_im_features(imgs: tuple) -> tuple:
    im_features = []
    for img in imgs:
        img = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(img)
        im_features.append(image_features)
    return im_features

def get_Dino_im_features(imgs: tuple) -> tuple:
    im_features = []
    for img in imgs:
        input = dino_preprocess(images=img, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            outputs = dino_model(pixel_values=input)
        image_features = outputs.last_hidden_state
        image_features = image_features.mean(dim=1)
        im_features.append(image_features)
    return im_features

# thanks chatGPT :)
def remove_word_from_filenames(folder_path, word_to_remove):
    """
    Recursively iterates over a folder and removes a given word from filenames within the folders.

    Args:
        folder_path (str): The path to the folder to iterate over.
        word_to_remove (str): The word to remove from filenames.

    Returns:
        None
    """
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if word_to_remove in filename:
                # Construct the new filename without the word to remove.
                new_filename = os.path.join(root, filename).replace(word_to_remove, "")
                # Rename the file.
                os.rename(os.path.join(root, filename), new_filename)


if __name__ == "__main__":
    main()