CUDA_VISIBLE_DEVICES=4 python train_MoVELoRA.py \
                                --seed 0 \
                                --promptpath "prompt.json" \
                                --trainids "train_ids.json" \
                                --output_dir "lora_output_rccar"

CUDA_VISIBLE_DEVICES=4 python   -m scripts.inference_i2mv_sdxl_train \
                                --image assets/demo/i2mv/cat_rccar.png \
                                --text  "A cat rides a racing car." \
                                --output output.png \
                                --remove_bg --scheduler ddpm --seed 0 \
                                --lora_name ./lora_output_rccar/pipeckpts