import os
import csv
import torch
import torchvision
from torchvision import transforms
from PIL import Image

# 数据预处理
transform = transforms.Compose([
    transforms.ToPILImage(),  # 将 Tensor 转为 PIL 图像
])

# 数据集加载
trainData = torchvision.datasets.MNIST('./data/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
testData = torchvision.datasets.MNIST('./data/', train=False, transform=torchvision.transforms.ToTensor())

# 保存图片及标签到文件夹和 CSV 文件的函数
def save_dataset_to_disk(dataset, root_folder, sub_folder, csv_file_name):
    """
    将 MNIST 数据集保存为图片文件，并生成对应的标签 CSV 文件。

    :param dataset: MNIST 数据集
    :param root_folder: 根文件夹名称
    :param sub_folder: 子文件夹名称（train 或 test）
    :param csv_file_name: 保存标签的 CSV 文件名称
    """
    # 创建根文件夹
    dataset_folder = os.path.join(root_folder, sub_folder)
    os.makedirs(dataset_folder, exist_ok=True)

    # 打开 CSV 文件进行写入
    csv_path = os.path.join(root_folder, csv_file_name)
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['ImagePath', 'Label'])  # 写入表头

        # 遍历数据集
        for idx, (img, label) in enumerate(dataset):
            # 定义文件路径
            img_folder = os.path.join(dataset_folder, str(label))  # 按标签分类到子文件夹
            os.makedirs(img_folder, exist_ok=True)
            img_path = os.path.join(img_folder, f'image_{idx}.png')

            # 保存图片
            img = transform(img)  # 转换为 PIL 图像
            img.save(img_path)

            # 写入标签到 CSV 文件
            writer.writerow([img_path, label])

            # 可选：打印进度
            if idx % 1000 == 0:
                print(f"Saved {idx}/{len(dataset)} images to {sub_folder}")

# 定义根文件夹
root_folder = './datasets'

# 保存训练数据集
save_dataset_to_disk(trainData, root_folder, 'train', 'train_labels.csv')

# 保存测试数据集
save_dataset_to_disk(testData, root_folder, 'test', 'test_labels.csv')

print("MNIST 数据集保存完成！")
