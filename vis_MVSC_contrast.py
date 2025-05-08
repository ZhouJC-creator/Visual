import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用黑体字体（SimHei），支持中文
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 "-" 显示为方块的问题

# 示例数据
layers = ['CDS61', 'AppleLeaf9', 'PPFGVC8', 'Rice9', 'CUB']  # 层数
accuracy_dataset1 = [87.669, 99.292, 92.532, 98.551, 91.082]
accuracy_dataset2 = [87.478,	99.279,	92.715,	98.879,	91.183]
accuracy_dataset3 = [87.919,	99.348,	92.769,	98.929,	91.321]

# 画图
plt.figure(figsize=(10, 6))
plt.plot(layers, accuracy_dataset1, marker='o', label='MC')
plt.plot(layers, accuracy_dataset2, marker='s', label='MVSC')
plt.plot(layers, accuracy_dataset3, marker='^', label='DMVSC')

# 添加标签和标题
plt.xlabel('数据集')
plt.ylabel('准确率')
plt.title('不同裁剪策略的准确率')
plt.legend()
plt.grid(True)

# 保存图片
plt.savefig(f"./output/xiaorong/DVSC_contrast.png",dpi=600,bbox_inches='tight')  # 保存为PNG格式的图片
