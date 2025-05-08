import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用黑体字体（SimHei），支持中文
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 "-" 显示为方块的问题

# 示例数据
layers = [5, 6, 7, 8, 9, 10, 11]  # 层数
accuracy_dataset1 = [87.875, 87.861, 87.789, 87.721, 87.875, 87.819, 87.919]
accuracy_dataset2 = [99.317, 99.218, 99.313, 99.242, 99.313, 99.294, 99.348]
accuracy_dataset3 = [92.581, 92.419, 92.749, 92.876, 92.231, 92.715, 92.769]
accuracy_dataset4 = [98.929, 97.826, 98.529, 97.5, 98.571, 98.929, 99.286]
accuracy_dataset5 = [91.217, 91.319, 91.35, 90.958, 91.316, 91.082, 91.321]

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
plt.xlabel('层数')
plt.ylabel('准确率')
plt.title('DMVSC模块基于不同层注意力权重的准确率')
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8),)  # 将图例放在右上方
plt.grid(True)

# 保存图片
plt.savefig(f"./output/xiaorong/DMVSC.png",dpi=1200,bbox_inches='tight')  # 保存为PNG格式的图片
