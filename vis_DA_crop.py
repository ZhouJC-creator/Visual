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
datasets_name = ["AgriculturalDisease", "AppleLeaf9", "CUB", "PlantPathology", "RiceLeaf", "WheatData"]
datasets_class = [61, 9, 200, 12, 9, 12]
dataset_name = datasets_name[3]
num_classes = datasets_class[3]
model_weight = "PlantPathology_26_0.92769.bin"
imgName = "rust01.jpg"

model = VisionTransformer(config, img_size=448, num_classes=num_classes)

model_pkl = f"./checkpoint/{model_name}/{model_weight}"
model.load_state_dict(torch.load(model_pkl, map_location="cpu")['model'])
model = model.eval()

transform0 = transforms.Compose([
    transforms.Resize((448, 448))
])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

im = Image.open(f"./img/{dataset_name}/{imgName}")
im = transform0(im)
x = transform(im)
x = x.unsqueeze(0)

logits, att_mat, crop_imgs, heads_select_index, last_select_index = model(x)


# head_select_rate = torch.full((1, 12), 64).squeeze(0) / (64 * 12)

# for i in range(12):
# crop_imgs, _,_  = attention_crop(att_mat[i], x, head_select_rate, (64 * 12))
crop_img = crop_imgs.squeeze(0)

# 假设归一化参数 (常用的 ImageNet 归一化参数)
mean = torch.tensor([0.485, 0.456, 0.406])  # 通道均值
std = torch.tensor([0.229, 0.224, 0.225])  # 通道标准差

# 定义反归一化函数
def denormalize(tensor, mean, std):
    mean = mean.view(1, 1, 3)
    std = std.view(1, 1, 3)
    return tensor * std + mean  # 恢复到原始范围

# 假设 crop_img 是 Tensor，先反归一化
if isinstance(crop_img, torch.Tensor):
    crop_img = crop_img.permute(1, 2, 0)  # 调整通道顺序
    crop_img = denormalize(crop_img, mean, std)  # 反归一化
    crop_img = torch.clamp(crop_img, 0, 1)  # 限制在 [0, 1] 范围
    crop_img = (crop_img.numpy() * 255).astype(np.uint8)  # 转换为 uint8 类型

# 保存图片
im.save(f"./output/DA/{dataset_name}/{imgName}_original.jpg")  # 保存原图
Image.fromarray(crop_img).save(f"./output/DA/{dataset_name}/{imgName}_DA_croped_single.jpg")  # 保存裁剪后的图片

# 创建图像展示
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 展示原始图像
axes[0].imshow(im)
axes[0].set_title("original")
axes[0].axis("off")  # 隐藏坐标轴

# 展示裁剪后的图像
axes[1].imshow(crop_img)
axes[1].set_title("crop")
axes[1].axis("off")  # 隐藏坐标轴

# 保存展示图为文件
plt.tight_layout()
plt.savefig(f"./output/DA/{dataset_name}/{imgName}_DA_croped.jpg")  # 保存对比图


print("done!")
