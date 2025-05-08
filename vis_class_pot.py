import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from sklearn.manifold import TSNE
import seaborn as sns
from torchvision import models
from Models.ViT.modeling import VisionTransformer, CONFIGS
from Models.ViT_SoftTriple.models.modeling import VisionTransformer, CONFIGS
# from Models.ViT_GATV2_SoftTriple.models.modeling import VisionTransformer, CONFIGS

config = CONFIGS["ViT-B_16"]
datasets_name = ["AgriculturalDisease", "AppleLeaf9", "CUB", "PlantPathology", "RiceLeaf"]
datasets_class = [61, 9, 200, 12, 9]
dataset_name = datasets_name[0]
num_classes = datasets_class[0]
model_name = "ViT_SoftTriple"
# model_name = "ViT"
model_weight = "AgriculturalDisease_13_0.87787.bin"

model = VisionTransformer(config, img_size=448, num_classes=num_classes)

model_pkl = f"./checkpoint/{model_name}/{model_weight}"
model.load_state_dict(torch.load(model_pkl, map_location="cpu")['model'])
model.eval()

# **2. 选择要输入的图片**
image_paths = [
    "./class_imgs/18/18_01.jpg",
    "./class_imgs/18/18_02.jpg",
    "./class_imgs/18/18_03.jpg",
    "./class_imgs/18/18_04.jpg",
    "./class_imgs/19/19_01.jpg",
    "./class_imgs/19/19_02.jpg",
    "./class_imgs/19/19_03.jpg",
    "./class_imgs/19/19_04.jpg",
    "./class_imgs/20/20_01.jpg",
    "./class_imgs/20/20_02.jpg",
    "./class_imgs/20/20_03.jpg",
    "./class_imgs/20/20_04.jpg",
    "./class_imgs/21/21_01.jpg",
    "./class_imgs/21/21_02.jpg",
    "./class_imgs/21/21_03.jpg",
    "./class_imgs/21/21_04.jpg",
]

# image_paths = [
#     "./class_imgs/18/18_01.jpg",
#     "./class_imgs/18/18_02.jpg",
#     "./class_imgs/18/18_03.jpg",
#     "./class_imgs/18/18_06.jpg",
#     "./class_imgs/19/19_01.jpg",
#     "./class_imgs/19/19_02.jpg",
#     "./class_imgs/19/19_03.jpg",
#     "./class_imgs/19/19_04.jpg",
#     "./class_imgs/20/20_01.jpg",
#     "./class_imgs/20/20_02.jpg",
#     "./class_imgs/20/20_03.jpg",
#     "./class_imgs/20/20_06.jpg",
#     "./class_imgs/21/21_01.jpg",
#     "./class_imgs/21/21_02.jpg",
#     "./class_imgs/21/21_03.jpg",
#     "./class_imgs/21/21_04.jpg",
# ]

labels = [0, 0, 0, 0,
          1, 1, 1, 1,
          2, 2, 2, 2,
          3, 3, 3, 3]  # 手动指定类别（类别 A, B, C）

# **3. 预处理图片**
transform = transforms.Compose([
transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

features = []

# **4. 加载图片并提取特征**
with torch.no_grad():
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")  # 确保是 RGB 图像
        img = transform(img).unsqueeze(0)  # 添加 batch 维度
        feature = model(img)  # 提取 CLS token
        # _,feature = model(img)  # 提取 CLS token
        feature = feature.cpu().numpy()
        features.append(feature)

features = np.concatenate(features, axis=0)

# **5. 进行降维（t-SNE）**
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
features_tsne = tsne.fit_transform(features)

# **6. 绘制分类图**
plt.figure(figsize=(8, 6))
sns.scatterplot(x=features_tsne[:, 0], y=features_tsne[:, 1], hue=labels, palette="coolwarm", s=100)
# plt.title("Selected Image Classification Visualization")
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# 去掉坐标轴的数字
plt.xticks([])  # 去掉 x 轴上的数字
plt.yticks([])  # 去掉 y 轴上的数字
plt.legend(title="Class Label")
plt.savefig(f"./output/{model_name}/class_pot/{dataset_name}/{model_name}_class_pot2.jpg", dpi=300)
plt.close()
print("done!")
