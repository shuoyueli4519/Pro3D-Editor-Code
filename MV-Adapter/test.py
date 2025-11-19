from diffusers import StableDiffusionPipeline
import torch

# 1. 加载预训练模型（可换成自己微调的）
model_id = "runwayml/stable-diffusion-v1-5"

# 2. 创建推理管线（推荐用 float16 加速）
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

# 3. 定义文本提示
prompt = "a futuristic city skyline at sunset, ultra detailed, cinematic lighting"

# 4. 生成图像
image = pipe(prompt).images[0]

# 5. 保存结果
image.save("output.png")

print("✅ 图片已保存到 output.png")
