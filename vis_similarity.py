import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from DataAugmentation.DA import attention_crop
from Models.ViT.modeling import VisionTransformer, CONFIGS
import torch.nn.functional as F
import seaborn as sns
import os
import time

# from Models.ViT_GATV2.models.modeling import VisionTransformer, CONFIGS

config = CONFIGS["ViT-B_16"]


model_name = "ViT"
datasets_name = ["AgriculturalDisease", "AppleLeaf9", "CUB", "PlantPathology", "WheatData"]
datasets_class = [61, 9, 200, 12, 12]
dataset_name = datasets_name[0]
num_classes = datasets_class[0]
model_weight = "AgriculturalDisease_13_0.87324.bin"
imgName = "cls_49.jpg"

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

# 配置文件夹路径和输出路径
dataset_folder = f"./img/{dataset_name}/"
output_folder = "./output/attention_similarity/"
os.makedirs(output_folder, exist_ok=True)

# 初始化存储所有相似性矩阵的列表
all_similarity_matrices = []
start_time = time.time()
# 遍历文件夹中的所有图片
for imgName in os.listdir(dataset_folder):
    img_path = os.path.join(dataset_folder, imgName)
    if not imgName.lower().endswith((".png", ".jpg", ".jpeg")):
        continue  # 跳过非图像文件

    # 加载并处理图片
    im = Image.open(img_path)
    im = transform0(im)
    x = transform(im)
    x = x.unsqueeze(0)

    # 获取模型输出
    logits, attention_matrices = model(x)
    num_layers = len(attention_matrices)

    # 初始化当前图片的相似性矩阵
    similarity_matrix = np.zeros((num_layers, num_layers))

    # 计算每一层与其他层的余弦相似度
    for i in range(num_layers):
        for j in range(num_layers):
            # 展平矩阵为一维向量
            A_i_flat = attention_matrices[i].flatten()
            A_j_flat = attention_matrices[j].flatten()

            # 计算余弦相似度
            cosine_sim = F.cosine_similarity(A_i_flat.unsqueeze(0), A_j_flat.unsqueeze(0), dim=1)

            # 将结果存储到相似性矩阵中
            similarity_matrix[i, j] = cosine_sim.item()

    # 将当前图片的相似性矩阵加入列表
    all_similarity_matrices.append(similarity_matrix)

# 计算平均相似性矩阵
average_similarity_matrix = np.mean(all_similarity_matrices, axis=0)

# 绘制平均相似性矩阵的热力图
plt.figure(figsize=(10, 8))
sns.heatmap(average_similarity_matrix, annot=True, cmap="YlGnBu",
            xticklabels=range(1, num_layers + 1), yticklabels=range(1, num_layers + 1))
plt.xlabel("Layer")
plt.ylabel("Layer")
plt.title("Average Attention Similarity Between Layers")

# 保存热力图
output_path = os.path.join(output_folder, "average_attention_similarity_heatmap.png")
plt.savefig(output_path, bbox_inches='tight', dpi=600)  # bbox_inches='tight' 确保保存的图像没有白边
plt.close()  # 关闭图像，释放内存

end_time = time.time()
execution_time = end_time - start_time  # 计算执行时间
print(f"Execution time: {execution_time:.4f} seconds")
print("done!")
