#!bin/sh
name=$1
echo ./data/realcap/$name
object_name=$name
object_name_loo=${object_name}"_loo"
echo ${object_name}
echo ${object_name_loo}

python leave_one_out_stage1.py -s data/realcap/${object_name}  \
    -m output/gs_init/${object_name_loo} \
    -r 1 --sparse_view_num 6 --sh_degree 0  \
    --init_pcd_name dust3r_4     --dust3r_json data/realcap/${object_name}/dust3r_4.json  \
    --white_background --use_dust3r

python leave_one_out_stage2.py -s data/realcap/${object_name}  \
    -m output/gs_init/${object_name_loo}  \
    -r 1 --sparse_view_num 6 --sh_degree 0   \
    --init_pcd_name dust3r_4     --dust3r_json data/realcap/${object_name}/dust3r_4.json   \
    --white_background --use_dust3r

python train_lora.py --exp_name controlnet_finetune/${object_name}   \
    --prompt xxy5syt00 --sh_degree 0 --resolution 1 --sparse_num 6   \
    --data_dir data/realcap/${object_name}     --gs_dir data/realcap/${object_name}  \
    --loo_dir output/gs_init/${object_name_loo}     --bg_white --sd_locked --train_lora --use_prompt_list   \
    --add_diffusion_lora --add_control_lora --add_clip_lora --use_dust3r

python train_repair.py     --config configs/gaussian-object-colmap-free.yaml  \
   --train --gpu 0     tag="${object_name}"     system.init_dreamer="output/gs_init/${object_name_loo}"  \
    system.exp_name="output/controlnet_finetune/${object_name}"    system.refresh_size=12  \
    data.data_dir="data/realcap/${object_name}"     data.resolution=1     data.sparse_num=6  \
    data.prompt="a photo of a xxy5syt00"     data.json_path="data/realcap/${object_name}/refined_cams.json" \
    data.refresh_size=12     system.sh_degree=0