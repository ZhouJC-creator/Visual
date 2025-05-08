import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
from Models.ViT_DMFF_36.models.modeling import VisionTransformer, CONFIGS


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

logits, att_mat, selected_inx_all = model(x)

print(len(att_mat))

att_tuple = []

for i in range(len(att_mat)):
    att_tuple.append(torch.tensor(att_mat[i]))
print(att_tuple[0].shape)
att_mat = torch.stack(att_tuple).squeeze(1)
print(att_mat.shape)

# Average the attention weights across all heads.
att_mat = torch.mean(att_mat, dim=1)

# To account for residual connections, we add an identity matrix to the
# attention matrix and re-normalize the weights.
residual_att = torch.eye(att_mat.size(1))
aug_att_mat = att_mat + residual_att
aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

# Recursively multiply the weight matrices
joint_attentions = torch.zeros(aug_att_mat.size())
joint_attentions[0] = aug_att_mat[0]

for n in range(1, aug_att_mat.size(0)):
    joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

# Attention from the output token to the input space.
v = joint_attentions[-1]
grid_size = int(np.sqrt(aug_att_mat.size(-1)))
mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
result = (mask * im).astype("uint8")

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

ax1.set_title('Original')
ax2.set_title('Attention Map')
_ = ax1.imshow(im)
_ = ax2.imshow(result)

for i, v in enumerate(joint_attentions):
    # Attention from the output token to the input space.
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()

    # Resize attention mask to match the input image size
    mask = cv2.resize(mask / mask.max(), im.size)  # Normalize mask to [0, 1]

    # Apply heatmap using OpenCV
    heatmap = cv2.applyColorMap((mask * 255).astype("uint8"), cv2.COLORMAP_JET)  # Generate heatmap

    # Convert PIL Image to numpy array for overlaying
    im_np = np.array(im)

    # Overlay heatmap on the original image (with a transparency factor)
    overlay = cv2.addWeighted(heatmap, 0.6, im_np, 0.4, 0)

    os.makedirs(f"./output/{model_name}/heatmap/{dataset_name}/", exist_ok=True)

    # Save the original image
    if i==0:
        im.save(f"./output/{model_name}/heatmap/{dataset_name}/{imgName.split('.')[0]}_original.jpg")

    # Save the attention heatmap overlay
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for consistency

    # Save overlay images
    plt.imsave(f"./output/{model_name}/heatmap/{dataset_name}/{imgName.split('.')[0]}_heatmap_layer_{i + 1}.jpg", overlay_rgb)  # Save overlay as RGB

    # Optionally display results using matplotlib
    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    # ax1.set_title('Original')
    # ax2.set_title(f'Attention Map - Layer {i + 1}')
    # ax1.imshow(im)
    # ax2.imshow(overlay_rgb)
    # plt.savefig(f"./output/{model_name}/{dataset_name}/{imgName.split('.')[0]}_vis_layer_{i + 1}.jpg")
    plt.close(fig)  # Close the plot to save memory if running multiple iterations

print("done!")

