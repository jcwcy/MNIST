import torch
from torchvision import transforms
from PIL import Image
import os

# 定义模型结构
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(in_features=7 * 7 * 64, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=10)  # 输出为 10 个类别
        )

    def forward(self, input):
        output = self.model(input)
        return output

# 定义预测函数
def predict_image(image_path, model_path):
    """
    对单张图片进行预测。

    :param image_path: 图片的路径
    :param model_path: 训练好的模型文件的路径
    :return: 预测的标签
    """
    # 检查文件路径是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件 {image_path} 不存在！")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在！")

    # 加载训练好的模型
    model = Net()  # 使用与训练时相同的模型结构
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式

    # 数据预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 确保图片为单通道灰度图
        transforms.Resize((28, 28)),  # 调整为 MNIST 输入大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 与训练时的标准化一致
    ])

    # 加载图片并应用预处理
    img = Image.open(image_path)
    img = transform(img)
    img = img.unsqueeze(0)  # 增加一个批次维度，形状变为 [1, 1, 28, 28]

    # 预测
    with torch.no_grad():
        outputs = model(img)
        predicted_label = torch.argmax(outputs, dim=1).item()  # 获取预测的标签

    return predicted_label

# 示例用法
if __name__ == "__main__":
    # 输入图片路径和模型路径
    image_path = 'datasets\\test\8\image_177.png'
    model_path = 'model.pth'

    try:
        # 调用预测函数
        result = predict_image(image_path, model_path)
        print(f"预测结果: {result}")
    except Exception as e:
        print(f"发生错误: {e}")
