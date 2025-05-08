import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from Models.ViT_DMFF_36.models.modeling import VisionTransformer, CONFIGS
import os

# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用黑体字体（SimHei），支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.weight']='bold'

config = CONFIGS["ViT-B_16"]

model_name = "ViT_DMFF_36"
datasets_name = ["AgriculturalDisease", "AppleLeaf9", "CUB", "PlantPathology", "WheatData"]
datasets_class = [61, 9, 200, 12, 12]
dataset_name = datasets_name[1]
num_classes = datasets_class[1]
model_weight = "AppleLeaf9_22_0.99348.bin"
imgName = "Powdery mildew (1369).jpg"

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
img = transform0(im)
x = transform(img)
x = x.unsqueeze(0)

logits, _, selected_inx_all = model(x)
selected_inx_all = selected_inx_all[0]
# 参数设置
patch_size = 16  # ViT通常使用16x16的patch
img_size = 448
grid_size = img_size // patch_size
num_layers = len(selected_inx_all)  # 获取总层数

# 创建保存目录
save_dir = f"./output/{model_name}/layer_select/{dataset_name}/"
os.makedirs(save_dir, exist_ok=True)

# 原始图像转换为numpy数组
img_np = np.array(img)

# 为每一层创建并保存选中token的可视化
for layer_idx, selected_inx in enumerate(selected_inx_all):
    # 创建黑色背景图像
    viz_img = np.zeros_like(img_np)

    # 标记被选中的patch
    for idx in selected_inx:
        # 计算patch在图像中的位置
        row = (idx % grid_size) * patch_size
        col = (idx // grid_size) * patch_size

        # 从原图复制选中的patch
        viz_img[col:col + patch_size, row:row + patch_size] = img_np[col:col + patch_size, row:row + patch_size]

    # 创建figure
    plt.figure(figsize=(8, 8))
    plt.imshow(viz_img)
    plt.title(f'{layer_idx + 2}层: {len(selected_inx)}个token被选择', fontsize=12)
    plt.axis('off')

    # 保存图像
    save_path = os.path.join(save_dir, f'{imgName}_layer_{layer_idx + 2}_selected_tokens.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=400, pad_inches=0.1)
    plt.close()

    print(f"Saved layer {layer_idx + 2} visualization to {save_path}")
img.save(f"./output/{model_name}/layer_select/{dataset_name}/{imgName.split('.')[0]}_original.jpg")
print(f"\nAll selected tokens visualizations saved to: {os.path.abspath(save_dir)}")