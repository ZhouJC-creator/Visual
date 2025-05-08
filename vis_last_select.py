import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms
from DataAugmentation.DA import attention_crop
# from Models.ViT.modeling import VisionTransformer, CONFIGS

from Models.ViT_DA_D.models.modeling import VisionTransformer, CONFIGS

# from Models.ViT_GATV2.models.modeling import VisionTransformer, CONFIGS

config = CONFIGS["ViT-B_16"]

model_name = "ViT_DA_D"
datasets_name = ["AgriculturalDisease", "AppleLeaf9", "CUB", "PlantPathology", "WheatData"]
datasets_class = [61, 9, 200, 12, 12]
dataset_name = datasets_name[3]
num_classes = datasets_class[3]
model_weight = "PlantPathology_26_0.92769.bin"
imgName = "rust_gray02.jpg"

model = VisionTransformer(config, img_size=448, num_classes=num_classes)

model_pkl = f"./checkpoint/{model_name}/{model_weight}"
var = torch.load(model_pkl, map_location="cpu")['model']
model.load_state_dict(torch.load(model_pkl, map_location="cpu")['model'])
asd = model.transformer.encoder.heads_select_rate
model = model.eval()

transform0 = transforms.Compose([
    transforms.Resize((448, 448))
])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

im = Image.open(f"./img/{dataset_name}/{imgName}")
img = transform0(im)
x = transform(img)
x = x.unsqueeze(0)

logits, att_mat, crop_imgs, heads_select_index, last_select_index = model(x)
print(last_select_index.shape)
grid_size = 28
img = img.convert("RGB")
img_width, img_height = img.size

# 计算每个块的宽度和高度
block_width = img_width // grid_size
block_height = img_height // grid_size

# 创建一个图层用于绘制高亮区域
overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))  # 透明图层
draw = ImageDraw.Draw(overlay)
# 遍历 select_index 并高亮对应的块
for _, index in enumerate(last_select_index):
    row = int(torch.div(index, grid_size, rounding_mode='trunc'))
    col = int(index % grid_size)
    # 计算图像块的左上角和右下角坐标
    x1 = col * block_width
    y1 = row * block_height
    x2 = x1+ block_width
    y2 = y1+ block_height

    # 确保坐标在图片范围内
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)

    # 在块上绘制半透明的高亮色块
    draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 128))  # 红色高亮，透明度128

# 将高亮图层叠加到原图上
highlighted_img = Image.alpha_composite(img.convert("RGBA"), overlay)

# 显示结果
# plt.figure(figsize=(8, 8))
# plt.imshow(highlighted_img)
# plt.axis("off")
# plt.title("Highlighted Image Blocks")
# plt.show()

# 如果需要保存高亮后的图片
highlighted_img.convert("RGB").save(f"./output/DA/{dataset_name}/{imgName}_last_selected.jpg")

print("done!")


