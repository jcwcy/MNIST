import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
import torch.nn as nn


# 定义模型结构，与训练时一致
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=7 * 7 * 64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, input):
        return self.model(input)


# 加载训练好的模型
class HandwrittenDigitRecognizer:
    def __init__(self, model_path):
        self.model = Net()  # 使用与训练时相同的模型
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # 加载模型
        self.model.eval()  # 切换到评估模式
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, image):
        transform = transforms.Compose([
            transforms.Resize((28, 28)),  # 将图像调整为28x28
            transforms.Grayscale(num_output_channels=1),  # 确保是单通道
            transforms.ToTensor(),  # 转换为 Tensor
            transforms.Normalize((0.5,), (0.5,))  # 归一化
        ])

        image = transform(image).unsqueeze(0).to(self.device)  # 添加批次维度
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output.data, 1)
            return predicted.item()


# 创建 GUI 界面
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("手写数字识别")
        self.geometry("400x400")

        self.canvas = tk.Canvas(self, bg="white", width=280, height=280)
        self.canvas.pack(pady=20)

        self.button_clear = tk.Button(self, text="清除", command=self.clear_canvas)
        self.button_clear.pack(side=tk.LEFT, padx=10)

        self.button_predict = tk.Button(self, text="识别", command=self.predict_digit)
        self.button_predict.pack(side=tk.RIGHT, padx=10)

        self.image = Image.new("L", (280, 280), 255)  # 创建白色背景图像
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)  # 清空图像
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="black", outline="black")
        self.draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="black", outline="black")

    def predict_digit(self):
        recognizer = HandwrittenDigitRecognizer('model.pth')  # 确保路径正确
        digit = recognizer.predict(self.image)
        messagebox.showinfo("识别结果", f"识别的数字是: {digit}")


if __name__ == "__main__":
    app = App()
    app.mainloop()