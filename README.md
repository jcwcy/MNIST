# MNIST手写数字识别项目

## 项目概述
本项目使用卷积神经网络（CNN）与 PyTorch 框架，对 MNIST 数据集中的手写数字进行识别。项目包含数据处理、模型训练、评估以及图形用户界面（GUI）展示识别结果。

## 项目使用说明
训练模型
python mnist.py

评估模型
python evaluation.py

启动 GUI 应用
python gui.py
用户可以在 GUI 中绘制数字并查看模型的识别结果。

单张图片预测
python predict.py

可视化结果
test_accuracy.png：显示模型在测试集上的准确率。
test_loss.png：显示模型训练过程中的损失变化。
