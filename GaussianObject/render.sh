CUDA_VISIBLE_DEVICES=4 python render.py \
                        -s ./data/render_keyviews/rccar \
                        -m ./data/render_keyviews/rccar \
                        --use_dust3r --sparse_view_num 6 \
                        --dust3r_json ./data/render_keyviews/rccar/dust3r_4.json --init_pcd_name dust3r_4