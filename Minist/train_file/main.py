import sys
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, 2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            # TensorBoard记录训练指标
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train', loss.item(), step)
            writer.add_scalar('Learning Rate',
                              optimizer.param_groups[0]['lr'], step)

            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

            if args.dry_run:
                break


def test(model, device, test_loader, writer, epoch):
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

    # TensorBoard记录测试指标
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    return test_loss, accuracy


def main(args=None):
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    # 添加训练批大小参数，类型int，默认64，帮助信息说明用途
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    # 添加测试批大小参数
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    # 添加训练轮次参数
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')

    # 添加初始学习率参数
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')

    # 添加学习率衰减系数
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')

    # 添加禁用CUDA的选项（action='store_true'表示存在该参数时值为True）
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # 添加禁用MPS（Mac GPU支持）的选项
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')

    # 添加试运行模式（快速验证代码是否可运行）
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')

    # 添加随机种子参数（保证可重复性）
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # 添加日志间隔参数（控制训练过程输出频率）
    parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                        help='how many batches to wait before logging training status')

    # 添加模型保存选项
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    # ...参数定义保持不变（共12个参数）...
    args = parser.parse_args(args)

    # 设备设置
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
    torch.manual_seed(args.seed)
    # 设备选择逻辑：
    if use_cuda:
        # 优先使用CUDA（NVIDIA GPU）
        device = torch.device("cuda")
    elif use_mps:
        # 其次使用MPS（Apple Silicon GPU）
        device = torch.device("mps")
    else:
        # 最后使用CPU
        device = torch.device("cpu")

    # 配置训练数据加载参数
    train_kwargs = {'batch_size': args.batch_size}
    # 配置测试数据加载参数
    test_kwargs = {'batch_size': args.test_batch_size}

    # 如果使用CUDA，添加CUDA专属配置
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,  # 数据加载子进程数
                       'pin_memory': True,  # 使用锁页内存加速数据传输
                       'shuffle': True}  # 训练数据打乱顺序
        # 更新训练和测试配置
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # 创建TensorBoard记录器
    log_dir = f"runs/mnist_resnet_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transform),
        batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    # 模型初始化
    model = ResNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test_loss, accuracy = test(model, device, test_loader, writer, epoch)
        scheduler.step()

    # 模型保存和导出
    if args.save_model:
        # 保存PyTorch模型
        torch.save(model.state_dict(), "mnist_resnet.pt")

        # 导出ONNX模型
        dummy_input = torch.randn(1, 1, 28, 28).to(device)
        onnx_path = "mnist_resnet.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            verbose=True
        )
        print(f"\nONNX model saved to {onnx_path}")

    writer.close()


if __name__ == '__main__':
    main()