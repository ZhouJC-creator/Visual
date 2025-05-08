import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchvision import transforms, datasets
from tqdm import tqdm
# from Models.ViT.modeling import VisionTransformer, CONFIGS
from Models.ViT_DA_D.models.modeling import VisionTransformer, CONFIGS
import os

plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用黑体字体（SimHei），支持中文
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 "-" 显示为方块的问题

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# 加载模型配置
config = CONFIGS["ViT-B_16"]
datasets_name = ["AgriculturalDisease", "AppleLeaf9", "CUB", "PlantPathology", "RiceLeaf"]
datasets_class = [61, 9, 200, 12, 9]
dataset_name = datasets_name[1]
num_classes = datasets_class[1]
model_name = "ViT_DA_D"
model_weight = "AppleLeaf9_28_0.99348.bin"

model = VisionTransformer(config, img_size=448, num_classes=num_classes)

# 加载预训练权重
model_pkl = f"./checkpoint/{model_name}/{model_weight}"
model.load_state_dict(torch.load(model_pkl, map_location="cpu")['model'])
model = model.to(device)
model.eval()  # 设置为评估模式

# 数据预处理（需与训练时一致）
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载测试数据集（假设目录结构为：./data/PlantPathology/test/）
test_dataset = datasets.ImageFolder(
    root=f'F:\\DataSet\\fine-grained\\AppleLeaf9\\test',  # 根据你的实际路径修改
    transform=transform
)

# 创建数据加载器
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=4,  # 根据显存调整
    shuffle=False
)

# 获取类别名称（假设目录结构按类别存放）
class_names = test_dataset.classes

# 存储所有预测结果和真实标签
all_preds = []
all_labels = []

# 禁用梯度计算以节省内存
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Processing"):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs, _, _, _, _ = model(images)

        # 获取预测结果
        _, preds = torch.max(outputs, 1)

        # 收集结果
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)

# 可视化混淆矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

# 设置图表属性
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签
plt.yticks(rotation=0)

# 调整布局并保存
plt.tight_layout()
os.makedirs(f"./output/{model_name}/confusion_matrix/", exist_ok=True)
plt.savefig(f"./output/{model_name}/confusion_matrix/{dataset_name}_confusion_matrix.png", dpi=900)
plt.close()

print("Confusion matrix saved successfully!")
