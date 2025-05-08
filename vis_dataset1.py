import json
import matplotlib.pyplot as plt
from collections import Counter

# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用黑体字体（SimHei），支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

plt.rcParams['font.weight']='bold'

# 读取 train.json 和 test.json 文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# 统计类别
def count_classes(data):
    class_counts = Counter([entry['disease_class'] for entry in data])
    return class_counts


# 绘制柱状图
def plot_class_distribution(train_counts, test_counts):
    # 合并 train 和 test 的统计结果
    all_classes = set(train_counts.keys()).union(test_counts.keys())
    # dict = {0: "苹果健康", 1: "苹果黑星病一般", 2: "苹果黑星病严重", 3: "苹果灰斑病"}
    # all_classes = {dict[item] for item in all_classes}
    train_vals = [train_counts.get(cls, 0) for cls in all_classes]
    test_vals = [test_counts.get(cls, 0) for cls in all_classes]


    # 设置柱状图的位置
    x = range(len(all_classes))
    width = 0.35  # 设置每个柱子的宽度

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, train_vals, width, label='Train')
    ax.bar([i + width for i in x], test_vals, width, label='Test')

    # 添加标签
    # ax.set_xlabel('类别')
    # ax.set_ylabel('样本数量')
    # ax.set_title('训练集和测试集中每个类的样本数')
    # ax.set_xticks([i + width / 2 for i in x],rotation=90)
    # ax.set_xticklabels(all_classes)
    # ax.legend()

    plt.xlabel('类别', fontsize=14, fontweight='bold')
    plt.ylabel('样本数量', fontsize=14, fontweight='bold')
    plt.title('训练集和测试集中每个类的样本数', fontsize=16, fontweight='bold')
    plt.xticks([i + width / 2 for i in x], all_classes, rotation=90)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.legend()

    # 显示图形
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./output/dataset/AgriculturalDisease.jpeg",dpi=900)  # 保存为PNG格式的图片


# 主程序
def main():
    # 假设文件路径如下（请根据实际文件路径调整）
    train_json_path = f'F:/DataSet/fine-grained/AgriculturalDisease/train/train.json'
    test_json_path = f'F:/DataSet/fine-grained/AgriculturalDisease/valid/valid.json'

    # 加载 JSON 数据
    train_data = load_json(train_json_path)
    test_data = load_json(test_json_path)

    # 统计类别
    train_counts = count_classes(train_data)
    test_counts = count_classes(test_data)

    # 绘制柱状图
    plot_class_distribution(train_counts, test_counts)


# 执行程序
if __name__ == "__main__":
    main()
