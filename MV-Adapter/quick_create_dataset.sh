#!/bin/bash

conda create -n mvadapter_dataset python=3.10
conda activate mvadapter_dataset

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

mkdir ../../dataset

python dataset_download.py --dataset shuoyueli/mvadapter-image-pair --save_dir ../../dataset/mvadapter_imagepair
python dataset_download.py --dataset shuoyueli/mvadapter_dataset_reference --save_dir ../../dataset/mvadapter_dataset

bash create_dataset_parrallel.sh