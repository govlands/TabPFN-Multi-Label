import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重现
torch.manual_seed(42)

# 定义简单的CNN模型
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(start_dim=1),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
    
    # def __init__(self):
    #     super(DigitCNN, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 32, 3, 1)
    #     self.conv2 = nn.Conv2d(32, 64, 3, 1)
    #     self.dropout1 = nn.Dropout(0.25)
    #     self.dropout2 = nn.Dropout(0.5)
    #     self.fc1 = nn.Linear(9216, 128)
    #     self.fc2 = nn.Linear(128, 10)

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = F.relu(x)
    #     x = self.conv2(x)
    #     x = F.relu(x)
    #     x = F.max_pool2d(x, 2)
    #     x = self.dropout1(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc1(x)
    #     x = F.relu(x)
    #     x = self.dropout2(x)
    #     x = self.fc2(x)
    #     return F.log_softmax(x, dim=1)

# 训练函数
def train(model, device, train_loader, optimizer, epochs):
    model.train()
    for epoch in epochs:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'训练轮次: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}')

# 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\n测试集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({accuracy:.1f}%)\n')
    return accuracy

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据集
    class CustomMNISTDataset(Dataset):
        def __init__(self, root, train=True, transform=None, download=False):
            # 使用 torchvision.datasets.MNIST 来下载并获取原始张量数据，然后在此包装类中应用 transform
            mnist = datasets.MNIST(root, train=train, download=download)
            self.images = mnist.data        # uint8 tensor, shape [N, H, W]
            self.targets = mnist.targets
            self.transform = transform

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            img = self.images[idx]
            target = int(self.targets[idx])
            if self.transform is not None:
                img = to_pil_image(img)     # 将 tensor 转为 PIL Image，以便 transform 正常工作
                img = self.transform(img)
            return img, target

    train_dataset = CustomMNISTDataset('./data', train=True, transform=transform, download=True)
    test_dataset = CustomMNISTDataset('./data', train=False, transform=transform, download=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 创建模型
    model = DigitCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("开始训练...")
    # 训练模型
    epochs = 5
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # 保存模型
    torch.save(model.state_dict(), "digit_classifier.pth")
    print("模型已保存为 digit_classifier.pth")

    # 展示一些预测结果
    model.eval()
    with torch.no_grad():
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        
        # 显示前6个样本的预测结果
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        for i in range(6):
            ax = axes[i//3, i%3]
            ax.imshow(data[i].cpu().squeeze(), cmap='gray')
            ax.set_title(f'真实: {target[i].item()}, 预测: {pred[i].item()}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('predictions.png')
        plt.show()

if __name__ == '__main__':
    main()