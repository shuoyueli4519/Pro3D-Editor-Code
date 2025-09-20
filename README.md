<h1 align="center">Pro3D-Editor:<br>A Progressive-Views Perspective for Consistent and Precise 3D Editing</h1>

<p align="center"><a href="https://arxiv.org/pdf/2506.00512"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://shuoyueli4519.github.io/Pro3D-Editor/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>

# Installation
Clone our repo first:
```
git clone https://github.com/huanngzh/MV-Adapter.git
```

Create a new conda env:

```
conda create -n Pro3DEditor python=3.10
conda activate Pro3DEditor
```

Install necessary packages:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
cd ./MV-Adapter
pip install -r requirements.txt
cd ../GaussianObject
pip install -r requirements.txt
cd ./models
python download_hf_models.py
```

Install necessary models:
1. [huanngzh/mv-adapter](https://huggingface.co/huanngzh/mv-adapter)
2. [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
3. [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
4. [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)
5. [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)


```
cd ../..
bash ./huggingface/hf_download_model.sh huanngzh/mv-adapter
bash ./huggingface/hf_download_model.sh stabilityai/stable-diffusion-xl-base-1.0
bash ./huggingface/hf_download_model.sh madebyollin/sdxl-vae-fp16-fix
bash ./huggingface/hf_download_model.sh ZhengPeng7/BiRefNet
bash ./huggingface/hf_download_model.sh openai/clip-vit-large-patch14
```

# Edit Demo
## Render Keyviews
In this case, our initial input is an object represented in 3DGS, specifically the file dust3r_4.ply located in ./GaussianObject/data/render_keyviews/rccar. In the same directory, camera.json contains the camera extrinsics of six key views, cfg_args is the configuration file for 3DGS initialization, and sparse_6.txt contains the indices of the views.
```
cd ./GaussianObject
bash render.sh
```
The rendering results are saved in `./data/render_keyviews/rccar/train/ours_None/renders`
<p align="center"><img src="assets/rendered_key_views.jpg" width="100%"></p>

## Edit Keyviews
In the previous step, we render six key views of the 3D object. Then, we use these key views for fine-tuning the multi-view diffusion model. After that, we will use the fine-tuned multi-view diffusion model to edit the key views.
```
cd ../MV-Adapter
cp ../GaussianObject/data/render_keyviews/rccar/train/ours_None/renders/* ./dataset/rccar
bash quick_start.sh
```
The edited key views are saved in `./results/images`
<p align="center"><img src="assets/edited_key_views.jpg" width="100%"></p>

## Edit 3D
In the previous step, we finish editing all the key views. Next, we will edit the entire 3D object.
```
cd ../GaussianObject
cp -r ../MV-Adapter/results/* ./realcap/cat_rccar
bash edit.sh cat_rccar
```
The edited 3D object is saved in ./output/gaussian_object/rccar/save


# Acknowledgement
Our code is based on [MV-Adapter](https://github.com/huanngzh/MV-Adapter) and [GaussianObject](https://github.com/chensjtu/GaussianObject). Thanks for their works.

# Citation:
If you find this work helpful, please consider citing our paper:
```
@article{zheng2025pro3d,
  title={Pro3D-Editor: A Progressive-Views Perspective for Consistent and Precise 3D Editing},
  author={Zheng, Yang and Huang, Mengqi and Chen, Nan and Mao, Zhendong},
  journal={arXiv preprint arXiv:2506.00512},
  year={2025}
}
```