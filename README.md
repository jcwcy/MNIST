# MNIST 手写数字识别项目

## 项目概述
本项目使用卷积神经网络（CNN）与 PyTorch 框架，对 MNIST 数据集中的手写数字进行识别。项目包含数据处理、模型训练、评估以及图形用户界面（GUI）展示识别结果。用户可以在 GUI 中绘制数字并查看模型的识别结果。

## 数据集
MNIST 数据集：包含 70,000 张手写数字图像（60,000 张训练集，10,000 张测试集），每张图像大小为 28x28 像素，灰度图。数据集存放在 `datasets/` 目录中。

### 数据集文件
- [MNIST DATABASE of handwritten digits](https://yann.lecun.com/exdb/mnist/)
- 可用文件：
  - `train-images-idx3-ubyte.gz`：训练集图像（9912422 bytes）
  - `train-labels-idx1-ubyte.gz`：训练集标签（28881 bytes）
  - `t10k-images-idx3-ubyte.gz`：测试集图像（1648877 bytes）
  - `t10k-labels-idx1-ubyte.gz`：测试集标签（4542 bytes）

## 环境要求
- Python: 3.x
- 依赖库: 所有依赖库在 `requirements.txt` 中列出。

## 项目使用说明
训练模型：
python mnist.py

评估模型：
python evaluation.py

启动 GUI 应用：
python gui.py

单张图片预测：
python predict.py

可视化结果:
test_accuracy.png：显示模型在测试集上的准确率。
test_loss.png：显示模型训练过程中的损失变化。
