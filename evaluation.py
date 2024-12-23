import torch
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

# 定义评估函数
def evaluate_model(model_path, test_loader, device):
    """
    评估模型的性能，计算 Accuracy, Precision, Recall 和 F1 Score。

    :param model_path: 训练好的模型文件路径
    :param test_loader: 测试数据集的 DataLoader
    :param device: 设备，'cpu' 或 'cuda'
    """
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在！")

    # 加载模型
    model = Net().to(device)  # 将模型移动到设备上
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载模型时指定设备
    model.eval()  # 设置为评估模式

    # 存储预测值和真实值
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据移动到设备上
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)  # 获取预测结果

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    #precision = precision_score(y_true, y_pred, average='weighted')
    #recall = recall_score(y_true, y_pred, average='weighted')
    #f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    #print(f"Precision: {precision:.4f}")
    #print(f"Recall: {recall:.4f}")
    #print(f"F1 Score: {f1:.4f}")

    return accuracy
    #return accuracy, precision, recall, f1

# 主程序
if __name__ == "__main__":
    # 设备设置
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 与训练时一致
    ])

    # 加载测试集
    test_dataset = datasets.MNIST('./data/', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256)

    # 模型路径
    model_path = 'model.pth'

    try:
        # 调用评估函数
        evaluate_model(model_path, test_loader, device)
    except Exception as e:
        print(f"发生错误: {e}")
