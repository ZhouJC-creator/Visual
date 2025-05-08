import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用黑体字体（SimHei），支持中文
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 "-" 显示为方块的问题

# 示例数据
layers = [8, 12, 24, 36, 48]  # 层数
accuracy_dataset1 = [87.732, 87.698, 87.643, 87.875, 87.762]
accuracy_dataset2 = [99.317, 99.242, 99.207, 99.348, 99.174]
accuracy_dataset3 = [92.581, 92.785, 92.831, 92.992, 92.645]
accuracy_dataset4 = [98.929, 98.551, 98.913, 97.571, 98.551]
accuracy_dataset5 = [91.217, 91.159, 92.231, 91.312, 91.304]

# 找到每个数据集的最大值及对应层数
datasets = {
    'CDS61': accuracy_dataset1,
    'AppleLeaf9': accuracy_dataset2,
    'PPFGVC8': accuracy_dataset3,
    'Rice9': accuracy_dataset4,
    'CUB-200-2011': accuracy_dataset5
}
colors = ['b', 'g', 'r', 'c', 'm']

# 画图
plt.figure(figsize=(10, 6))
# plt.plot(layers, accuracy_dataset1, marker='o', label='CDS61')
# plt.plot(layers, accuracy_dataset2, marker='s', label='AppleLeaf9')
# plt.plot(layers, accuracy_dataset3, marker='^', label='PPFGVC8')
# plt.plot(layers, accuracy_dataset4, marker='d', label='Rice9')
# plt.plot(layers, accuracy_dataset5, marker='x', label='CUB-200-2011')

for (label, data), color in zip(datasets.items(), colors):
    plt.plot(layers, data, marker='o', label=label, color=color)

    # 找最大值并标出
    max_idx = data.index(max(data))
    max_layer = layers[max_idx]
    max_value = data[max_idx]
    plt.scatter(max_layer, max_value, color=color, s=100, edgecolors='black', zorder=5)
    plt.annotate(f'{max_value:.3f}%', (max_layer, max_value), textcoords="offset points", xytext=(0, 10), ha='center',
                 fontsize=10, color=color)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
# 添加标签和标题
plt.xlabel('token初始每层选择数量')
plt.ylabel('准确率')
plt.title('DMFF模块初始token选择数量对应准确率')
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8),)  # 将图例放在右上方
plt.grid(True)

# 保存图片
plt.savefig(f"./output/xiaorong/DDMFF.png", dpi=1200, bbox_inches='tight')  # 保存为PNG格式的图片
