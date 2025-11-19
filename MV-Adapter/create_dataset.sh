#!/bin/bash

# 定义 inference_image 变量
inference_image="../../dataset/mvadapter_imagepair/datasets--shuoyueli--mvadapter-image-pair/scripts/seed_edit_api/output_img2img_1"
reference_image="../../dataset/mvadapter_dataset/datasets--huanngzh--Objaverse-Ortho10View/mnt/pfs/data/texture_ortho10view_easylight_objaverse"
train_image="./dataset/3D_model"
create_dataset="./create_dataset"
results="./results"

dirs=($(ls -d ${inference_image}/*/ 2>/dev/null))
for d in "${dirs[@]}"; do
    # rm -f "$train_image"/*
    # mkdir -p "$train_image"

    name="$(basename "$d")"
    short="${name:0:2}"
    if [[ "$short" == "0b" ]]; then
        continue
    fi
    target_path="${reference_image}/${short}/${name}"
    
    src_dir="$target_path"
    declare -A mapping=(
        ["color_0000.webp"]="00000.png"
        ["color_0004.webp"]="00001.png"
        ["color_0001.webp"]="00002.png"
        ["color_0002.webp"]="00003.png"
        ["color_0003.webp"]="00004.png"
        ["color_0005.webp"]="00005.png"
    )
    for key in "${!mapping[@]}"; do
        src_file="$src_dir/$key"
        dst_file="$train_image/${mapping[$key]}"
        if [[ -f "$src_file" ]]; then
            echo "✅ 复制: $src_file → $dst_file"
            cp "$src_file" "$dst_file"
        else
            echo "⚠️  找不到文件: $src_file"
        fi
    done

    CUDA_VISIBLE_DEVICES=2 python train_MoVELoRA.py \
                                --seed 0 \
                                --promptpath "prompt.json" \
                                --trainids "train_ids.json" \
                                --output_dir "lora_output_3D_model"
    inference_path="${inference_image}/${name}"
    for f in "$inference_path"/*; do
        CUDA_VISIBLE_DEVICES=2 python   -m scripts.inference_i2mv_sdxl_train \
                            --image $f \
                            --text  "A 3D model." \
                            --output output.png \
                            --remove_bg --scheduler ddpm --seed 0 \
                            --lora_name ./lora_output_3D_model/pipeckpts
        folder_name=$(basename "$inference_path")
        detail_foldername=$(basename "$f" .png)
        target_create_dataset="${create_dataset}/${folder_name}/${detail_foldername}"
        mkdir -p "$target_create_dataset"
        for d in "$results"/*/; do
            if [[ -d "$d" ]]; then
                echo "📁 复制目录: $d → $target_create_dataset"
                cp -r "$d" "$target_create_dataset/"
            fi
        done
    done
done
