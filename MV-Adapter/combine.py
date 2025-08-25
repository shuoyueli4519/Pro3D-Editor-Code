import argparse
from PIL import Image
import os

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='合并指定文件夹下的所有图片为一张一行六列的大图')
parser.add_argument('--folder_path', type=str, help='包含图片的文件夹路径')
parser.add_argument('--output', type=str, default='merged_image.png', help='输出图片的文件名')
args = parser.parse_args()

# 获取文件夹路径和输出文件名
folder_path = args.folder_path
parent_path = os.path.dirname(folder_path)
print(parent_path)
output_image = os.path.join(parent_path, args.output)

# 读取所有图片
image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
image_files.sort()  # 确保图片顺序一致

# 检查图片数量
if len(image_files) == 0:
    raise ValueError("文件夹中没有找到图片！")

# 打开所有图片
images = [Image.open(os.path.join(folder_path, img)).resize((768, 768)) for img in image_files]

# 计算最终合并图片的尺寸（一行六列）
num_images = len(images)
single_width, single_height = images[0].size
merged_width = single_width * num_images
merged_height = single_height

# 创建新画布
merged_image = Image.new('RGB', (merged_width, merged_height))

# 将所有图片粘贴到新画布上
for i, img in enumerate(images):
    merged_image.paste(img, (i * single_width, 0))

# 保存最终合并的图片
merged_image.save(output_image)
print(f"图片已保存为 {output_image}")
