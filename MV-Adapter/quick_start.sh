CUDA_VISIBLE_DEVICES=7 python train_lora.py --seed 0 --promptpath "prompt.json" --trainids "train_ids.json" --output_dir "lora_output_girl"

CUDA_VISIBLE_DEVICES=7 python   -m scripts.inference_i2mv_sdxl_train \
                                --image assets/demo/i2mv/A_girl_red_dress.png \
                                --text  "A little girl wears a red dress." \
                                --output output.png \
                                --remove_bg --scheduler ddpm --seed 0 \
                                --lora_name ./lora_output_girl/pipeckpts_800