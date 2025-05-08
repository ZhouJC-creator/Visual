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

score = all_attention_scores[10][:, :, 0, 1:].mean(dim=1)

score= torch.softmax(score,dim=-1)

sorted_scores, sorted_indices = torch.sort(score, descending=True, dim=1)

# 转换数据为numpy数组
scores = score.squeeze().detach().cpu().numpy()

plt.figure(figsize=(15, 6))

# ====================== 概率分布图单独保存 ======================
plt.figure(figsize=(10, 6))
n_bins = 50

# 绘制直方图
n, bins, patches = plt.hist(scores, bins=n_bins, density=True,
                           alpha=0.7, label='直方图')

# 添加KDE曲线
kde = gaussian_kde(scores)
x = np.linspace(scores.min(), scores.max(), 500)
plt.plot(x, kde(x), 'r-', linewidth=2, label='密度曲线')

plt.title('概率分布图')
plt.xlabel('分数值')
plt.ylabel('概率密度')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存概率分布图
plt.savefig('./output/cdf/probability_distribution.png', dpi=300, bbox_inches='tight')
plt.close()  # 关闭当前图像

# ====================== CDF累计分布图单独保存 ======================
plt.figure(figsize=(10, 6))

# sorted_scores = np.sort(scores)
cdf = np.arange(1, len(scores)+1) / len(scores)

# 绘制CDF曲线
plt.plot(scores, cdf, 'b-', linewidth=2, label='累计曲线')

# 叠加累计直方图
plt.hist(scores, bins=n_bins, density=True, cumulative=True,
         histtype='step', linewidth=2,
         color='orange', label='累计直方图')

plt.title('累计分布图')
plt.xlabel('分数值')
plt.ylabel('累计概率')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存累计分布图
plt.savefig('./output/cdf/cumulative_distribution.png', dpi=300, bbox_inches='tight')
plt.close()  # 关闭当前图像

# 生成概率点（0到1之间的1000个点）
p = np.linspace(0, 1, 1000)

# 计算逆函数值（分位数）
quantiles = np.quantile(scores, p)

# ====================== 绘制逆函数图 ======================
plt.figure(figsize=(10, 6))

# 绘制阶梯状逆函数（更符合离散数据特性）
plt.step(p, quantiles, where='post', linewidth=2, label='分位函数')

plt.title('分位数函数（CDF逆函数）')
plt.xlabel('概率值 (p)')
plt.ylabel('对应分数值 (F⁻¹(p))')
plt.grid(True, alpha=0.3)
plt.legend()

# 添加参考线示例
plt.axhline(y=np.median(scores), color='r', linestyle='--',
           label=f'中位数 ({np.median(scores):.2f})')
plt.axvline(x=0.5, color='g', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('./output/cdf/cdf_inverse_function.png', dpi=300, bbox_inches='tight')
plt.close()