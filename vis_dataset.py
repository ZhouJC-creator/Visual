import os
import matplotlib.pyplot as plt

# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用黑体字体（SimHei），支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.weight']='bold'

datasets_name = ["AgriculturalDisease", "AppleLeaf9", "CUB", "PlantPathology", "WheatData","RiceLeaf"]
dataset_name = datasets_name[1]
# 设置数据集根目录
dataset_root = f"F:\\DataSet\\fine-grained\\{dataset_name}"

# 用于统计类别和数量
train_class_counts = {}
test_class_counts = {}

# 遍历训练集文件夹
train_dir = os.path.join(dataset_root, 'train')
for class_folder in os.listdir(train_dir):
    class_folder_path = os.path.join(train_dir, class_folder)
    if os.path.isdir(class_folder_path):  # 如果是文件夹
        train_class_counts[class_folder] = len(os.listdir(class_folder_path))

# 遍历测试集文件夹
test_dir = os.path.join(dataset_root, 'test')
for class_folder in os.listdir(test_dir):
    class_folder_path = os.path.join(test_dir, class_folder)
    if os.path.isdir(class_folder_path):  # 如果是文件夹
        test_class_counts[class_folder] = len(os.listdir(class_folder_path))

# 绘制柱状图
categories = sorted(set(train_class_counts.keys()).union(test_class_counts.keys()))
train_counts = [train_class_counts.get(category, 0) for category in categories]
test_counts = [test_class_counts.get(category, 0) for category in categories]

# 创建柱状图
x = range(len(categories))

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
bars_train = ax.bar(x, train_counts, width=bar_width, label='训练集', align='center')
bars_test = ax.bar([i + bar_width for i in x], test_counts, width=bar_width, label='测试集', align='center')
# bars_train = ax.bar(x, train_counts, width=bar_width, label='Train', color='#1f77b4', align='center')
# bars_test = ax.bar([i + bar_width for i in x], test_counts, width=bar_width, label='Test', color='#d62728', align='center')

plt.xlabel('类别', fontsize=14, fontweight='bold')
plt.ylabel('样本数量', fontsize=14, fontweight='bold')
plt.title('训练集和测试集中每个类的样本数', fontsize=16, fontweight='bold')
plt.xticks([i + bar_width / 2 for i in x], categories, rotation=90)
plt.xticks(fontsize=12, fontweight='bold')
plt.legend()

# 为每个柱状图添加数量标签
for bar in bars_train:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 5, str(height), ha='center', va='bottom')

for bar in bars_test:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 5, str(height), ha='center', va='bottom')

plt.tight_layout()

# 保存柱状图为文件
plt.savefig(f"./output/dataset/{dataset_name}.jpeg",dpi=1200)  # 保存为PNG格式的图片

# 如果你希望保存为其他格式，比如JPEG、PDF等，只需要修改文件名扩展名即可：
# plt.savefig('class_distribution.pdf')  # 保存为PDF格式
# plt.savefig('class_distribution.jpg')  # 保存为JPEG格式

# 关闭图形，避免占用内存
plt.close()
print("done!")