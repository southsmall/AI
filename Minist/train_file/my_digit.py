import argparse
# 导入PyTorch深度学习框架主库
import torch
# 导入PyTorch神经网络模块（包含层定义、损失函数等）
import torch.nn as nn
# 导入PyTorch函数式接口（包含激活函数、池化等无需参数的操作）
import torch.nn.functional as F
# 导入优化算法模块（包含SGD、Adam等优化器）
import torch.optim as optim
# 从torchvision导入计算机视觉数据集（MNIST）和图像转换工具
from torchvision import datasets, transforms
# 导入学习率调度器（用于训练过程中调整学习率）
from torch.optim.lr_scheduler import StepLR
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(1,32,3,1)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32,64,3,1)
        self.bn2=nn.BatchNorm2d(64)
        self.dropout1=nn.Dropout(0.25)
        self.dropout2=nn.Dropout(0.5)
        self.fc1=nn.Linear(9216,128)
        self.fc2=nn.Linear(128,10)
        self.bn3=nn.BatchNorm1d(128)   #全连接是bn1 卷积是bn2
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2)
        x=self.dropout1(x)
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=self.bn3(x)
        x=F.relu(x)
        x=self.dropout2(x)
        x=self.fc2(x)
        output=F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

                # 如果开启dry_run模式（试运行），处理一个批次后立即退出
                if args.dry_run:
                    break
def test(args, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target,reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
def main(args):
        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

        # 添加训练批大小参数，类型int，默认64，帮助信息说明用途
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')

        # 添加测试批大小参数
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')

        # 添加训练轮次参数
        parser.add_argument('--epochs', type=int, default=14, metavar='N',
                            help='number of epochs to train (default: 14)')

        # 添加初始学习率参数
        parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
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
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')

        # 添加模型保存选项
        parser.add_argument('--save-model', action='store_true', default=False,
                            help='For Saving the current Model')

        # 解析命令行参数，转换为命名空间对象
        args = parser.parse_args()

        # 判断是否使用CUDA：当未禁用CUDA且CUDA可用时为True
        use_cuda = not args.no_cuda and torch.cuda.is_available()

        # 判断是否使用MPS（Mac的Metal加速）
        use_mps = not args.no_mps and torch.backends.mps.is_available()

        # 设置随机种子（保证可重复性）
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

        # 定义数据预处理管道：
        transform = transforms.Compose([
            # 将PIL图像或numpy数组转换为Tensor，并自动归一化到[0,1]区间
            transforms.ToTensor(),
            # 标准化处理：(input - mean) / std
            # MNIST的均值和标准差是预设值
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 下载并加载训练数据集：
        # root='../data' 数据存储路径
        # train=True 加载训练集
        # download=True 如果本地不存在则自动下载
        # transform=transform 应用定义好的预处理流程
        dataset1 = datasets.MNIST('../data', train=True, download=True,
                                  transform=transform)

        # 加载测试数据集（train=False）
        dataset2 = datasets.MNIST('../data', train=False,
                                  transform=transform)

        # 创建训练数据加载器：
        # dataset1：训练数据集
        # **train_kwargs：解包字典参数（包含batch_size等）
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)

        # 创建测试数据加载器
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        # 实例化模型，并迁移到目标设备
        model = ResNet().to(device)

        # 定义优化器：Adadelta算法，传入模型参数和初始学习率
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay=1e-4)

        # 定义学习率调度器：每1个epoch后学习率乘以gamma（即衰减30%）
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        # 训练循环：遍历每个epoch
        for epoch in range(1, args.epochs + 1):
            # 执行训练过程
            train(args, model, device, train_loader, optimizer, epoch)
            # 执行测试评估
            test(model, device, test_loader)
            # 更新学习率（调度器步进）
            scheduler.step()

        # 如果开启保存模型选项，保存模型参数
        if args.save_model:
            # 保存模型状态字典到文件mnist_cnn.pt
            torch.save(model.state_dict(), "mnist_cnn.pt")

    # 当脚本直接执行时（而非被导入时），运行主函数


if __name__ == '__main__':
    main()