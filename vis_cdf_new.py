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


def inverse_cdf_sampling(weights, num_samples):
    """
    使用逆变换采样从归一化后的权重分布中采样 token 索引。
    参数：
        weights: Tensor，形状 [B, L]，每个 token 的得分
        num_samples: 每个样本采样的 token 数量
    返回：
        Tensor，形状 [B, num_samples]，采样到的 token 在序列中的索引
    """
    # 将得分归一化为概率分布
    probs = torch.softmax(weights, dim=-1)
    # 计算累积分布函数（CDF）
    cdf = torch.cumsum(probs, dim=-1)  # shape: [B, L]
    # 生成均匀分布的随机数，形状为 [B, num_samples]
    step = 1.0 / (2 * num_samples)
    quantiles = torch.linspace(step, 1 - step, num_samples)
    # rand_samples = torch.rand(probs.size(0), num_samples, device=weights.device)
    rand_samples = quantiles.unsqueeze(0).expand(probs.size(0), -1)
    # 使用 searchsorted 根据 CDF 找到每个随机数对应的 token 索引
    indices = torch.searchsorted(cdf, rand_samples, right=True)
    indices = torch.clamp(indices, max=783)
    return indices, cdf, rand_samples, probs


# 示例数据
# weights = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])  # 假设有 5 个 token
num_samples = 24  # 采样 10 个点

# 调用逆变换采样函数
indices, cdf, rand_samples, probs = inverse_cdf_sampling(score, num_samples)

# 将 CDF、随机数和概率转换为 numpy 数组以便绘图
cdf_np = cdf.detach().numpy().flatten()
rand_samples_np = rand_samples.cpu().numpy().flatten()
indices_np = indices.detach().numpy().flatten()
probs_np = probs.detach().numpy().flatten()

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制每个 token 的重要性（概率分布）
plt.bar(range(len(probs_np)), probs_np*10, alpha=0.5, label="token 重要性分布", color="orange")

# 绘制 token 重要性（概率分布）曲线
plt.plot(range(len(probs_np)), probs_np*10, label="token 重要性曲线", color="green", linewidth=1)

# 绘制 CDF 曲线
plt.plot(range(len(cdf_np)), cdf_np, label="CDF值", color="blue", linewidth=1.5)

# 标注采样点并绘制虚线连接
for i, (rand_sample, index) in enumerate(zip(rand_samples_np, indices_np)):
    # 绘制采样点
    plt.scatter(index, rand_sample, color="red", zorder=5)
    # 绘制虚线连接到 x 轴和 y 轴
    plt.plot([index, index], [0, rand_sample], 'r--', alpha=0.5)  # 连接到 y 轴
    plt.plot([0, index], [rand_sample, rand_sample], 'r--', alpha=0.5)  # 连接到 x 轴
    # 添加采样点标签
    plt.text(index, rand_sample, '', fontsize=9, ha='right')

# 添加标题和标签
plt.title("逆变换采样选取token索引")
plt.xlabel("token 索引")
plt.ylabel("CDF值")
plt.legend()
plt.grid(True)
# plt.show()


plt.tight_layout()
plt.savefig('./output/cdf/cdf_inverse_function_new.png', dpi=1200, bbox_inches='tight')
plt.close()
