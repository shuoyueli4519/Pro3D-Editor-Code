import os
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

image_folder = "./output/rccar/train/ours_None/renders"
# image_folder = "./output/head/train/ours_None/renders"
# image_folder = "./output/girl/train/ours_None/renders"
# image_folder = "./output/box/train/ours_None/renders"
# image_folder = "./output/knight_toy/train/ours_None/renders"
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

alpha = 5.0  # 控制 P_is 影响力
gamma = 40.0  # 控制 P_idel 影响力
w1, w2, w3 = 1.0, 1.0, 1.5  # 三个分数的权重

images = [preprocess(Image.open(os.path.join(image_folder, img))).unsqueeze(0).to(device) for img in image_files]

original_prompt = "a rubber duck toy riding a toy car with a number two on it."
edit_prompt = "a car with a tail on its back."
# edit_prompt = "a cat riding a toy car with a number two on it."
# original_prompt = "a cartoon Viking head with a black helmet, silver horns, and a red beard."
# edit_prompt = "a head with a red pepper-like mustache on its face."
# edit_prompt = "a head with two fairy ears."
# original_prompt = "a doll with big blue eyes, gray hair, upright rabbit ears, a gray vest, and pink shoes, arms outstretched."
# edit_prompt = "a doll wearing a red dress."
# edit_prompt = "a doll wearing a pair of blue shoes."
# original_prompt = "A cute cartoonish blue creature with cat-like ears, a long tail."
# edit_prompt = "A cartoonish creature with a hammer like tail."
# edit_prompt = "A cute cartoonish blue creature with rabbit-like ears."
# original_prompt = "A knight in armor, holding sword and shield, with big round eyes."
# edit_prompt = "A knight holds a beautiful shield."
# edit_prompt = "A knight wearing a blue mask."

text_tokens = clip.tokenize([original_prompt, edit_prompt]).to(device)
text_features = model.encode_text(text_tokens).float()
text_features /= text_features.norm(dim=-1, keepdim=True)  # 归一化

original_feature, edit_feature = text_features  # 分离两个文本特征

# 计算所有视图的 CLIP 图像特征
image_features = torch.cat([model.encode_image(img).float() for img in images])
image_features /= image_features.norm(dim=-1, keepdim=True)  # 归一化

# 计算 CLIP 相似度
clip_is = torch.matmul(image_features, original_feature)  # 与原 Prompt 的相似度
clip_it = torch.matmul(image_features, edit_feature)      # 与编辑 Prompt 的相似度

# 计算 softmax 归一化概率
def softmax(x, alpha):
    e_x = torch.exp(alpha * x)  # 避免溢出
    return e_x / e_x.sum()

P_is = softmax(clip_is, gamma)  # 原视图选择概率
P_it = softmax(clip_it, gamma)  # 编辑视图选择概率
P_idel = softmax((clip_it - clip_is), alpha)  # 无关视图选择概率
print(f"clip_is: {clip_is}, clip_it: {clip_it}, clip_it - clip_is: {clip_it - clip_is}")
print(f"P_it : {P_it}, P_is: {P_is}, P_idel: {P_idel}")
P_idel_roll_3 = torch.roll(P_idel, 27)  # 循环移位
P_idel_roll_5 = torch.roll(P_idel, 45)  # 循环移位

# 计算最终得分
# final_score = w1 * P_is + w1 * P_it - w2 * (P_idel_roll_3 + P_idel_roll_5)
final_score = w1 * P_is + w1 * P_it
print(f"final_score: {final_score}")
final_score_numpy = final_score.cpu().detach().numpy()
P_is_numpy = P_is.cpu().detach().numpy()
P_it_numpy = P_it.cpu().detach().numpy()
P_idel_numpy = P_idel.cpu().detach().numpy()

# 选出最佳视图索引
top3_values, top3_indices = torch.topk(final_score, k=3)
top3_indices = top3_indices.tolist()
top3_image_paths = [image_files[i] for i in top3_indices]
min_top3_value = top3_values.min().item()

# differences = np.abs(final_score_numpy[1:] - final_score_numpy[:-1])
# average_difference = differences.sum() / 36
# filtered_tensor = [final_score_numpy[0]]
# for i in range(1, len(final_score_numpy)):
#     if differences[i - 1] <= average_difference:
#         filtered_tensor.append(final_score_numpy[i])
# final_score_numpy = filtered_tensor

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# ax1.plot(P_idel_numpy, marker='x', linestyle='--', color='r', label='P_idel')
# ax1.set_title("P_idel")
# ax1.set_xlabel("Index")
# ax1.set_ylabel("Value")
# ax1.legend()  # 添加图例
# ax1.grid(True)  # 添加网格

fig, ax2 = plt.subplots(1, 1, figsize=(12, 3))
ax2.plot(final_score_numpy, marker='o', linestyle='-', color='#1f77b4', markersize=3)
# ax2.axhline(y=min_top3_value, color='r', linestyle='--', label=f'Top 3 Min Value: {min_top3_value:.2f}')
# ax2.set_xlabel("Azimuth Angle", fontsize=14, fontname='Times New Roman', labelpad=-200)
ax2.set_ylabel("Sampler Score", fontsize=20, fontname='Times New Roman')
ax2.legend()  # 添加图例
ax2.grid(True)  # 添加网格

angles = np.arange(len(final_score_numpy)) * 5  # 每个索引代表5度
ax2.set_xticks(np.arange(len(final_score_numpy)))  # 设置刻度位置
ax2.set_xticklabels([f"{angle}°" for angle in angles], fontsize=12, fontname='Times New Roman')  # 设置刻度标签并添加度数符号

ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
ax2.tick_params(axis='y', labelsize=12)
ax2.tick_params(axis='x', labelsize=20)

plt.tight_layout()
plt.savefig("combined_plots.png")
print("两个折线图已保存为 combined_plots.png")

print(f"最佳前三编辑视图: {top3_image_paths}")