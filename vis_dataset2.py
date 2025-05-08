import os
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import numpy as np
import imageio


class CUB:
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform

        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))

        img_name_list = [line[:-1].split(' ')[-1] for line in img_txt_file]
        label_list = [int(line[:-1].split(' ')[-1]) - 1 for line in label_txt_file]  # 0-indexed
        train_test_list = [int(line[:-1].split(' ')[-1]) for line in train_val_file]

        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
            self.train_imgname = [x for x in train_file_list[:data_len]]

        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'images', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
            self.test_imgname = [x for x in test_file_list[:data_len]]

    def __getitem__(self, index):
        if self.is_train:
            img, target, imgname = self.train_img[index], self.train_label[index], self.train_imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img, target, imgname = self.test_img[index], self.test_label[index], self.test_imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)

        return img, target


# 统计每个类别的图片数量
def count_classes(dataset):
    if dataset.is_train:
        labels = dataset.train_label
    else:
        labels = dataset.test_label

    return Counter(labels)


# 绘制柱状图
def plot_class_distribution(train_counts, test_counts):
    all_classes = set(train_counts.keys()).union(test_counts.keys())
    train_vals = [train_counts.get(cls, 0) for cls in all_classes]
    test_vals = [test_counts.get(cls, 0) for cls in all_classes]

    x = range(len(all_classes))
    width = 0.35  # 设置柱子的宽度

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, train_vals, width, label='Train')
    ax.bar([i + width for i in x], test_vals, width, label='Test')

    ax.set_xlabel('Class ID')
    ax.set_ylabel('Number of Images')
    ax.set_title('Image Distribution per Class in Train and Test Sets')
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(list(all_classes))
    ax.legend()

    plt.tight_layout()
    plt.show()


def main():
    # 设置数据集路径
    dataset_dir = 'F:\\DataSet\\fine-grained\\CUB'  # 请根据实际情况调整路径

    # 创建 CUB 数据集对象
    train_dataset = CUB(root=dataset_dir, is_train=True)
    test_dataset = CUB(root=dataset_dir, is_train=False)

    # 统计训练集和测试集每个类别的图片数量
    train_counts = count_classes(train_dataset)
    test_counts = count_classes(test_dataset)

    # 绘制柱状图
    plot_class_distribution(train_counts, test_counts)


if __name__ == "__main__":
    main()
