import os
from PIL import Image

# 第一行的顺序
ORDER = ["0000", "0004", "0001", "0002", "0003", "0005"]

BASE1 = "../../dataset/mvadapter_dataset/datasets--huanngzh--Objaverse-Ortho10View/mnt/pfs/data/texture_ortho10view_easylight_objaverse"

EDIT_FOLDERS = ["add", "modify1", "modify2", "modify3", "delete"]

CREATE_DATASET_DIR = "./create_dataset"
OUTPUT_DIR = "./visulization"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = Image.open(path).convert("RGBA").resize((768, 768))

    # 创建白色背景
    white_bg = Image.new("RGB", img.size, (255, 255, 255))

    # 贴到白色背景上，使用 alpha 通道作为 mask
    white_bg.paste(img, mask=img.split()[3])

    return white_bg


def concat_horizontal(images):
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    canvas = Image.new("RGB", (sum(widths), max(heights)))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width
    return canvas


def concat_vertical(im1, im2):
    w = max(im1.width, im2.width)
    h = im1.height + im2.height
    canvas = Image.new("RGB", (w, h))
    canvas.paste(im1, (0, 0))
    canvas.paste(im2, (0, im1.height))
    return canvas


def process_one_folder(folder_name):
    print("Processing:", folder_name)

    subdir = folder_name[:2]
    base_path = os.path.join(BASE1, subdir, folder_name)

    # ---------- 第一行 ----------
    row1_images = []
    for idx in ORDER:
        img_path = os.path.join(base_path, f"color_{idx}.webp")
        row1_images.append(load_image(img_path))
    row1 = concat_horizontal(row1_images)

    # ---------- 第二行（5 个 edit folder） ----------
    base_path_2 = os.path.join(CREATE_DATASET_DIR, folder_name)
    for edit in EDIT_FOLDERS:
        edit_img_dir = os.path.join(base_path_2, edit, "images")
        if not os.path.exists(edit_img_dir):
            print("   Missing:", edit_img_dir)
            continue

        img_files = sorted([
            f for f in os.listdir(edit_img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ])

        if len(img_files) == 0:
            print("   No images in", edit_img_dir)
            continue

        row2_images = [load_image(os.path.join(edit_img_dir, f)) for f in img_files]
        row2 = concat_horizontal(row2_images)

        # ---------- 两行拼接 ----------
        final = concat_vertical(row1, row2)

        # 输出为：visulization/<folder_name>/<edit>.png
        out_dir = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{edit}.png")
        final.save(out_path)
        print("   Saved:", out_path)



def main():
    folders = sorted([
        f for f in os.listdir(CREATE_DATASET_DIR)
        if os.path.isdir(os.path.join(CREATE_DATASET_DIR, f))
    ])

    for folder in folders:
        process_one_folder(folder)


if __name__ == "__main__":
    main()
