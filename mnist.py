import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt 

# 定义模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            # 第一层卷积，输入通道数为1（灰度图），输出通道数为16，卷积核大小为3x3，步幅为1，边缘补齐1像素
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),  # 激活函数 ReLU
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化，池化窗口大小为2x2，步幅为2

            # 第二层卷积，输入通道数为16，输出通道数为32，其他参数同上
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三层卷积，输入通道数为32，输出通道数为64，其他参数同上
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            # 展平，将多维特征图变为一维向量
            torch.nn.Flatten(),
            # 全连接层，输入特征数为7*7*64，输出特征数为128
            torch.nn.Linear(in_features=7 * 7 * 64, out_features=128),
            torch.nn.ReLU(),
            # 最后一层全连接，输入特征数为128，输出为10（对应10个类别）
            torch.nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, input):
        output = self.model(input)  # 前向传播
        return output

# 检查设备是否支持 GPU，如果支持则使用 CUDA，否则使用 CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 定义数据预处理方法
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # 将图像转换为 Tensor 格式
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])  # 对数据进行归一化，均值为0.5，标准差为0.5
])

# 数据集与数据加载器
BATCH_SIZE = 256  # 每批次的数据量
EPOCHS = 10  # 训练轮数
trainData = torchvision.datasets.MNIST('./data/', train=True, transform=transform, download=True)  # 加载训练数据集
testData = torchvision.datasets.MNIST('./data/', train=False, transform=transform)  # 加载测试数据集
trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)  # 训练数据加载器，打乱顺序
testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE)  # 测试数据加载器

# 初始化模型
net = Net().to(device)  # 将模型加载到指定设备上
print(net)  # 打印模型结构

# 定义损失函数和优化器
lossF = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失函数
learning_rate = 1e-3  # 学习率
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 优化器使用 Adam

# 训练与测试
history = {'Test Loss': [], 'Test Accuracy': []}  # 保存每轮测试损失和准确率
for epoch in range(1, EPOCHS + 1):
    processBar = tqdm(trainDataLoader, unit='step')  # tqdm 进度条，用于显示训练进度
    net.train(True)  # 启用训练模式
    for step, (trainImgs, labels) in enumerate(processBar):
        trainImgs, labels = trainImgs.to(device), labels.to(device)  # 将数据和标签加载到设备上

        # 前向传播
        outputs = net(trainImgs)  # 获取模型输出
        loss = lossF(outputs, labels)  # 计算损失
        predictions = torch.argmax(outputs, dim=1)  # 获取预测值
        accuracy = torch.sum(predictions == labels) / labels.shape[0]  # 计算准确率

        # 反向传播
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % 
                                   (epoch, EPOCHS, loss.item(), accuracy.item()))  # 更新进度条信息

    # 测试
    correct, totalLoss = 0, 0  # 初始化测试数据
    net.train(False)  # 关闭训练模式，启用评估模式
    with torch.no_grad():  # 不计算梯度
        for testImgs, labels in testDataLoader:
            testImgs, labels = testImgs.to(device), labels.to(device)  # 将测试数据加载到设备上
            outputs = net(testImgs)  # 获取模型输出
            loss = lossF(outputs, labels)  # 计算损失
            predictions = torch.argmax(outputs, dim=1)  # 获取预测值

            totalLoss += loss  # 累积测试损失
            correct += torch.sum(predictions == labels)  # 累积正确预测数

        testAccuracy = correct / len(testData)  # 测试准确率，使用测试数据总数计算
        testLoss = totalLoss / len(testDataLoader)  # 平均测试损失
        history['Test Loss'].append(testLoss.item())  # 保存测试损失
        history['Test Accuracy'].append(testAccuracy.item())  # 保存测试准确率

    print("[Epoch %d/%d] Test Loss: %.4f, Test Accuracy: %.4f" % 
          (epoch, EPOCHS, testLoss.item(), testAccuracy.item()))  # 打印测试结果

# 可视化
plt.plot(history['Test Loss'], label='Test Loss')  # 绘制测试损失曲线
plt.title('Test Loss vs Epoch')  # 添加标题
plt.legend(loc='best')  # 添加图例
plt.grid(True)  # 添加网格
plt.xlabel('Epoch')  # 横轴标签
plt.ylabel('Loss')  # 纵轴标签
plt.savefig('./test_loss.png')  # 保存图表为文件
plt.show()  # 显示图表

plt.plot(history['Test Accuracy'], color='red', label='Test Accuracy')  # 绘制测试准确率曲线
plt.title('Test Accuracy vs Epoch')  # 添加标题
plt.legend(loc='best')  # 添加图例
plt.grid(True)  # 添加网格
plt.xlabel('Epoch')  # 横轴标签
plt.ylabel('Accuracy')  # 纵轴标签
plt.savefig('./test_accuracy.png')  # 保存图表为文件
plt.show()  # 显示图表

# 保存模型
torch.save(net.state_dict(), './model.pth')  # 保存模型参数
