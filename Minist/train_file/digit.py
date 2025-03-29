# 导入命令行参数解析库，用于处理用户输入的训练参数
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


# 定义神经网络模型类（继承自nn.Module基类）
class Net(nn.Module):
    def __init__(self):
        # 调用父类构造函数完成基础初始化
        super(Net, self).__init__()
        # 定义第一个卷积层：
        # in_channels=1（MNIST是灰度图单通道）
        # out_channels=32（使用32个卷积核）
        # kernel_size=3（3x3的卷积核）
        # stride=1（卷积步长）
        self.conv1 = nn.Conv2d(1, 32, 3, 1)

        # 定义第二个卷积层：
        # 输入通道32（与前一层的输出通道数一致）
        # 输出通道64（提取更高阶特征）
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # 定义第一个Dropout层，丢弃概率25%
        # 用于防止过拟合，随机丢弃25%的神经元连接
        self.dropout1 = nn.Dropout(0.25)

        # 定义第二个Dropout层，丢弃概率50%
        # 通常在全连接层使用更高的丢弃率
        self.dropout2 = nn.Dropout(0.5)

        # 第一个全连接层：
        # 输入维度9216（由前一层的输出张量展平后的尺寸计算得到）
        # 输出维度128（隐含层神经元数量）
        self.fc1 = nn.Linear(9216, 128)

        # 第二个全连接层（输出层）：
        # 输入128维，输出10维（对应MNIST的0-9十个类别）
        self.fc2 = nn.Linear(128, 10)

    # 定义前向传播过程（描述数据流向）
    def forward(self, x):
        # 第一卷积层处理（此时x形状：[batch,1,28,28] -> [batch,32,26,26]）
        x = self.conv1(x)
        # 应用ReLU激活函数（引入非线性）
        x = F.relu(x)

        # 第二卷积层处理（形状变为[batch,64,24,24]）
        x = self.conv2(x)
        x = F.relu(x)

        # 应用2x2最大池化（形状减半->[batch,64,12,12]）
        x = F.max_pool2d(x, 2)
        # 应用第一个Dropout（训练模式下随机丢弃25%节点）
        x = self.dropout1(x)

        # 展平张量：保留batch维度（dim=1），其他维度展平为一维
        # [batch,64,12,12] -> [batch,64*12*12=9216]
        x = torch.flatten(x, 1)

        # 第一个全连接层处理（9216 -> 128维）
        x = self.fc1(x)
        x = F.relu(x)
        # 应用第二个Dropout（丢弃50%连接）
        x = self.dropout2(x)

        # 第二个全连接层处理（128 -> 10维）
        x = self.fc2(x)
        # 应用对数Softmax（各元素值在[-inf,0]区间）
        # dim=1表示在特征维度（即10个类别）做归一化
        output = F.log_softmax(x, dim=1)
        return output


# 定义训练函数
def train(args, model, device, train_loader, optimizer, epoch):
    # 设置模型为训练模式（启用Dropout和BatchNorm等训练专用层）
    model.train()
    # 遍历训练数据加载器，batch_idx是批次索引，(data, target)是当前批次数据
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据和标签迁移到指定设备（GPU/CPU）
        data, target = data.to(device), target.to(device)

        # 清除优化器中的历史梯度（防止梯度累积）
        optimizer.zero_grad()

        # 前向传播计算输出（调用模型的forward方法）
        output = model(data)

        # 计算负对数似然损失（NLLLoss）
        # 输入是经过log_softmax的输出，target是真实标签
        loss = F.nll_loss(output, target)

        # 反向传播计算梯度（自动微分系统计算各参数梯度）
        loss.backward()

        # 根据梯度更新模型参数（执行优化步骤）
        optimizer.step()

        # 每隔指定间隔打印训练进度
        if batch_idx % args.log_interval == 0:
            # 格式化输出信息：
            # epoch：当前训练轮次
            # batch_idx*len(data)：已处理样本数
            # len(train_loader.dataset)：总样本数（60000）
            # 100.*batch_idx/len(train_loader)：完成进度百分比
            # loss.item()：当前批次的损失值
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

            # 如果开启dry_run模式（试运行），处理一个批次后立即退出
            if args.dry_run:
                break


# 定义测试函数
def test(model, device, test_loader):
    # 设置模型为评估模式（关闭Dropout等训练专用层）
    model.eval()
    test_loss = 0
    correct = 0  # 记录正确预测的数量

    # 关闭自动求导以加速计算并减少内存占用
    with torch.no_grad():
        # 遍历测试数据集
        for data, target in test_loader:
            # 迁移数据到设备
            data, target = data.to(device), target.to(device)

            # 前向传播得到输出
            output = model(data)

            # 累加批次损失（使用sum而非平均，因为后面要除以总样本数）
            # reduction='sum'表示返回批次损失总和
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # 获取预测结果：找到概率最大的索引（dim=1在特征维度操作）
            # keepdim=True保持维度结构，便于后续比较
            pred = output.argmax(dim=1, keepdim=True)

            # 将预测结果与真实标签比较，统计正确数量
            # view_as(pred)将target形状调整为和pred一致
            # .sum()统计True的数量，.item()转换为Python数值
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 计算平均损失：总损失除以测试集样本数（10000）
    test_loss /= len(test_loader.dataset)

    # 格式化输出测试结果
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# 主函数（程序入口）
def main():
    # 创建参数解析器，并设置描述信息
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
    model = Net().to(device)

    # 定义优化器：Adadelta算法，传入模型参数和初始学习率
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # 定义学习率调度器：每1个epoch后学习率乘以gamma（即衰减30%）
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

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