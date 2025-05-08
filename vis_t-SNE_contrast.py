import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from sklearn.manifold import TSNE
import seaborn as sns
import os
from tqdm import tqdm  # 导入 tqdm 库
from torchvision import models
# from Models.ViT_Contrast.models.modeling import VisionTransformer, CONFIGS
from Models.ViT_SoftTriple.models.modeling import VisionTransformer, CONFIGS

# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用黑体字体（SimHei），支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.weight']='bold'

# # **1. 加载 ViT 模型**
# config = CONFIGS["ViT-B_16"]
# datasets_name = ["AgriculturalDisease", "AppleLeaf9", "CUB", "PlantPathology", "RiceLeaf"]
# datasets_class = [61, 9, 200, 12, 9]
# dataset_name = datasets_name[1]
# num_classes = datasets_class[1]
# model_name = "ViT_Contrast"
# model_weight = "AppleLeaf9_22_0.99279.bin"

# **1. 加载 ViT 模型**
config = CONFIGS["ViT-B_16"]
datasets_name = ["AgriculturalDisease", "AppleLeaf9", "CUB", "PlantPathology", "RiceLeaf"]
datasets_class = [61, 9, 200, 12, 9]
dataset_name = datasets_name[1]
num_classes = datasets_class[1]
model_name = "ViT_Contrast"
model_weight = "AppleLeaf9_22_0.99279.bin"

model = VisionTransformer(config, img_size=448, num_classes=num_classes)
model_pkl = f"./checkpoint/{model_name}/{model_weight}"
model.load_state_dict(torch.load(model_pkl, map_location="cpu")['model'])
model.eval()

# **2. 设置图片文件夹路径**
image_root = "./class_imgs/AppleLeaf"  # 图片文件夹路径
categories = sorted(os.listdir(image_root))  # 获取类别文件夹
image_paths = []
labels = []
label_to_class = {}  # 标签-类别名映射

# **3. 收集图片路径和类别标签**
for label, category in enumerate(categories):
    category_path = os.path.join(image_root, category)
    label_to_class[label] = category
    if not os.path.isdir(category_path):
        continue
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(img_path)
            labels.append(label)

# **4. 图片预处理**
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

features = []

# **5. 提取特征**
with torch.no_grad():
    for img_path in tqdm(image_paths):
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0)
        feature = model(img)
        features.append(feature.cpu().numpy())

features = np.concatenate(features, axis=0)

# **6. t-SNE 降维**
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
features_tsne = tsne.fit_transform(features)

# **7. 绘制 t-SNE 可视化图**
plt.figure(figsize=(12, 8))
unique_labels = np.unique(labels)
palette = sns.color_palette("tab20", len(unique_labels))


# 映射标签到颜色
label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

# 绘制散点图
scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1],
                      c=[label_to_color[label] for label in labels],
                      s=120, edgecolors='black')

# 创建图例
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=label_to_color[label],
                      markersize=10, label=label_to_class[label])
           for label in unique_labels]

plt.legend(handles=handles, title="Class Label")

# 美化图像
plt.xticks([])
plt.yticks([])
plt.title(f"Contrast Loss 在 AppleLeaf9 数据集上的t-SNE图")

# **8. 保存图像**
output_dir = f"./output/{model_name}/t-SNE/{dataset_name}"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, f"{model_name}_t-SNE_Apple.jpg"), dpi=900, bbox_inches='tight')

print("t-SNE 可视化完成，已保存至:", output_dir)