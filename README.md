# Pro3D-Editor-Code🚀

# Installation

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

# Edit Demo
## Render Keyviews
```
cd ./GaussianObject
bash render.sh
```
The rendering results are saved in `./data/render_keyviews/rccar/images`

## Edit Keyviews
```
cd ../MV-Adapter
cp ../GaussianObject/data/render_keyviews/rccar/images/* ./dataset/rccar
bash quick_start.sh
```
The edited key views are saved in `./results/images`

## Edit 3D
```
cd ../GaussianObject
cp -r ../MV-Adapter/results/* ./realcap/cat_rccar
bsah edit.sh cat_rccar
```
The edited 3D object is saved in ./output/gaussian_object/rccar/save