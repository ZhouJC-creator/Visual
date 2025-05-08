import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import seaborn as sns
from Models.ViT.modeling import VisionTransformer, CONFIGS
from scipy.stats import gaussian_kde

# 设置中文字体（确保系统已安装支持中文的字体）
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# from Models.ViT_GATV2.models.modeling import VisionTransformer, CONFIGS
# ![](output/ViT/heatmap/CUB/Laysan_Albatross_0102_611_heatmap_layer_12.jpg)from Models.ViT_DA_D.models.modeling import VisionTransformer, CONFIGS

config = CONFIGS["ViT-B_16"]

datasets_name = ["AgriculturalDisease", "AppleLeaf9", "CUB", "PlantPathology", "RiceLeaf"]
datasets_class = [61, 9, 200, 12, 9]
dataset_name = datasets_name[1]
num_classes = datasets_class[1]
model_name = "ViT"
model_weight = "AppleLeaf9_15_0.99279.bin"
imgName = "Rust (2161).jpg"

model = VisionTransformer(config, img_size=448, num_classes=num_classes)

model_pkl = f"./checkpoint/{model_name}/{model_weight}"
model.load_state_dict(torch.load(model_pkl, map_location="cpu")['model'])
model.eval()

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
print(x.unsqueeze(0).size())

logits, att_mat, all_attention_scores = model(x.unsqueeze(0))

print(len(att_mat))

score = all_attention_scores[10][:, :, 0, 1:].mean(dim=1).squeeze(0)
score = torch.softmax(score, dim=-1)

# 将 score 转换为 numpy 数组
score_np = score.detach().numpy()

# 创建 x 轴（token 的索引）
x = np.arange(len(score_np))

# 创建柱状图
plt.figure(figsize=(10, 6))
plt.bar(x, score_np, alpha=0.6, label='Token Importance')

# 创建光滑曲线图
# 使用 x 和 score_np 作为输入
kde = gaussian_kde(x, weights=score_np,bw_method=0.03)
x_smooth = np.linspace(x.min(), x.max(), 500)
y_smooth = kde(x_smooth)
plt.plot(x_smooth, y_smooth, color='red', label='Smooth Curve')

# 添加标签和标题
plt.xlabel('Token Index')
plt.ylabel('Importance Score')
plt.title('Token Importance Distribution')
plt.legend()

# 显示图像
# plt.show()
plt.savefig('./output/cdf/token_score.png', dpi=1200, bbox_inches='tight')