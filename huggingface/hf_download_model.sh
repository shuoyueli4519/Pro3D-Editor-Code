#!/bin/bash

python  ./huggingface/hf_download.py \
        --model "$1" \
        --save_dir ./huggingface