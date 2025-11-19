#!/bin/bash

# -------------------------------
#  路径配置
# -------------------------------
inference_image="../../dataset/mvadapter_imagepair/datasets--shuoyueli--mvadapter-image-pair/scripts/seed_edit_api/output_img2img_1"
reference_image="../../dataset/mvadapter_dataset/datasets--shuoyueli--mvadapter_dataset_reference/datasets--huanngzh--Objaverse-Ortho10View/mnt/pfs/data/texture_ortho10view_easylight_objaverse"
create_dataset="./create_dataset_new"
results="./results"

# 获取所有任务目录
dirs=($(ls -d ${inference_image}/*/ 2>/dev/null))

# -------------------------------
#  自动检测 GPU 数量
# -------------------------------
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    IFS=',' read -ra gpu_list <<< "$CUDA_VISIBLE_DEVICES"
    n_gpus=${#gpu_list[@]}
else
    # 否则 fallback 到物理卡
    n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

echo "🔍 检测到 $n_gpus 张 GPU 可用。"

if [[ $n_gpus -eq 0 ]]; then
    echo "❌ 没有检测到 GPU，退出。"
    exit 1
fi

# -------------------------------
#  定义一个函数用于运行单个任务
# -------------------------------
run_task() {
    local gpu_id=$1
    local d=$2

    local physical_gpu=${gpu_list[$gpu_id]}

    export CUDA_VISIBLE_DEVICES=$physical_gpu
    echo "[运行] 逻辑 GPU $gpu_id → 物理 GPU $physical_gpu"

    train_image="./dataset/3D_model_gpu${gpu_id}"
    mkdir -p "$train_image"

    name="$(basename "$d")"
    short="${name:0:2}"
    # if [[ "$short" == "0b" ]]; then
    #     echo "跳过 $name"
    #     return
    # fi

    target_path="${reference_image}/${short}/${name}"

    # 建立文件映射
    declare -A mapping=(
        ["color_0000.webp"]="00000.png"
        ["color_0004.webp"]="00001.png"
        ["color_0001.webp"]="00002.png"
        ["color_0002.webp"]="00003.png"
        ["color_0003.webp"]="00004.png"
        ["color_0005.webp"]="00005.png"
    )

    # 拷贝参考图
    for key in "${!mapping[@]}"; do
        src_file="$target_path/$key"
        dst_file="$train_image/${mapping[$key]}"
        if [[ -f "$src_file" ]]; then
            echo "[$name] GPU $gpu_id 复制: $src_file → $dst_file"
            cp "$src_file" "$dst_file"
        else
            echo "[$name] GPU $gpu_id 警告: 找不到文件 $src_file"
        fi
    done

    # -------------------------------
    #        训练 LoRA
    # -------------------------------
    python train_MoVELoRA.py \
        --seed 0 \
        --promptpath "prompt.json" \
        --trainids "train_ids_gpu${gpu_id}.json" \
        --output_dir "lora_output_3D_model_gpu${gpu_id}"

    # -------------------------------
    #        推理多个图
    # -------------------------------
    inference_path="${inference_image}/${name}"
    target_create_dataset="${create_dataset}/${name}"

    for f in "$inference_path"/*; do
        filename=$(basename "$f")
        filename_without_ext="${filename%.png}"
        target_output_path="${target_create_dataset}/${filename_without_ext}"
        python -m scripts.inference_i2mv_sdxl_train_parrallel \
            --image "$f" \
            --text "A 3D model." \
            --output "$target_output_path" \
            --remove_bg --scheduler ddpm --seed 0 \
            --lora_name ./lora_output_3D_model_gpu${gpu_id}/pipeckpts
    done

    echo "🎉 [$name] GPU $gpu_id 任务完成"
}

# -------------------------------
#  多GPU并行分发任务
# -------------------------------
task_id=0
for d in "${dirs[@]}"; do
    # 等待直到运行中的任务数量 < GPU 数量
    while (( $(jobs -r | wc -l) >= n_gpus )); do
        sleep 1
    done

    gpu_id=$((task_id % n_gpus))

    echo "➡️  分配任务: $d 到 GPU $gpu_id"

    run_task "$gpu_id" "$d" &   # 后台运行

    ((task_id++))
done

wait  # 等待所有 GPU 任务完成

echo "🎯 全部任务结束！"
