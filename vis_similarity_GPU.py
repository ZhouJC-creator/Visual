import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from DataAugmentation.DA import attention_crop
from Models.ViT.modeling import VisionTransformer, CONFIGS
# from Models.ViT_DA_D.models.modeling import VisionTransformer, CONFIGS
import torch.nn.functional as F
import seaborn as sns
import os
import time
from tqdm import tqdm  # 导入 tqdm 库

# from Models.ViT_GATV2.models.modeling import VisionTransformer, CONFIGS

config = CONFIGS["ViT-B_16"]

model_name = "ViT"
datasets_name = ["AgriculturalDisease", "AppleLeaf9", "CUB", "PlantPathology", "WheatData", "RiceLeaf"]
datasets_class = [61, 9, 200, 12, 12, 9]
dataset_name = datasets_name[5]
num_classes = datasets_class[5]
model_weight = "./RiceLeaf_11_0.97857.bin"
# imgName = "cls_49.jpg"

# 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型并移至GPU（如果有）
model = VisionTransformer(config, img_size=448, num_classes=num_classes)

model_pkl = f"./checkpoint/{model_name}/{model_weight}"
model.load_state_dict(torch.load(model_pkl, map_location=device)['model'])
model = model.to(device)  # 将模型加载到GPU（如果可用）
model = model.eval()

transform0 = transforms.Compose([
    transforms.Resize((448, 448))
])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 配置文件夹路径和输出路径
# dataset_folder = f"./img/{dataset_name}/"
dataset_folder = f"E:\\MyProject\\Fine-grained\\Visual\\img\\AgriculturalDisease"
# dataset_folder = f"F:\\DataSet\\Mix-ViT_dataset\\CUB\\images"
output_folder = "./output/attention_similarity/"
os.makedirs(output_folder, exist_ok=True)

# 获取图片总数以显示进度条
img_files = [
    os.path.join(root, file)
    for root, _, files in os.walk(dataset_folder)
    for file in files
    if file.lower().endswith((".png", ".jpg", ".jpeg"))
]
# 初始化存储所有相似性矩阵的列表
all_similarity_matrices = []

start_time = time.time()
# 使用 tqdm 包装文件遍历，显示进度条
for imgName in tqdm(img_files, desc="Processing images", ncols=100, unit="image"):
    # img_path = os.path.join(dataset_folder, imgName)

    # 加载并处理图片
    im = Image.open(imgName).convert("RGB")
    im = transform0(im)
    x = transform(im)
    x = x.unsqueeze(0).to(device)  # 将图片数据转移到GPU（如果有）

    # 获取模型输出
    # logits, attention_matrices = model(x)
    logits, attention_matrices, _ = model(x)
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
            xticklabels=range(1, 12 + 1), yticklabels=range(1, 12 + 1))
plt.xlabel("Layer")
plt.ylabel("Layer")
# plt.title("Attention Similarity Between Layers on Dataset" + dataset_name)
plt.title("Attention Similarity Between Layers on Dataset PPFGVC8")

# 保存热力图
output_path = os.path.join(output_folder, dataset_name + "_attention_similarity_heatmap.png")
plt.savefig(output_path, bbox_inches='tight', dpi=800)  # bbox_inches='tight' 确保保存的图像没有白边
plt.close()  # 关闭图像，释放内存

# 选择相似度高的矩阵对进行逐元素相乘，并绘制热力图


end_time = time.time()
execution_time = end_time - start_time  # 计算执行时间
print(f"Execution time: {execution_time:.4f} seconds")
print("done!")
