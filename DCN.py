import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
import argparse


# 数据预处理

def testset(dataset):
    if dataset == "USPS":
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return datasets.USPS(root='./data', train=False, download=True, transform=transform_test)
    if dataset == "MNIST":
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    if dataset == "SVHN":
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)


def trainset(dataset):
    if dataset == "USPS":
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return datasets.USPS(root='./data', train=True, download=True, transform=transform_train)
    if dataset == "MNIST":
        transform_train = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    if dataset == "SVHN":
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)


# 加载数据集
def testloader(dataset):
    if dataset == "USPS":
        return torch.utils.data.DataLoader(testset(dataset), batch_size=128, shuffle=False, num_workers=2)
    if dataset == "MNIST":
        return torch.utils.data.DataLoader(testset(dataset), batch_size=128, shuffle=False, num_workers=2)
    if dataset == "SVHN":
        return torch.utils.data.DataLoader(testset(dataset), batch_size=128, shuffle=False, num_workers=2)


def trainloader(dataset):
    if dataset == "USPS":
        return torch.utils.data.DataLoader(trainset(dataset), batch_size=128, shuffle=True, num_workers=2)
    if dataset == "MNIST":
        return torch.utils.data.DataLoader(trainset(dataset), batch_size=128, shuffle=True, num_workers=2)
    if dataset == "SVHN":
        return torch.utils.data.DataLoader(trainset(dataset), batch_size=128, shuffle=False, num_workers=2)


# 定义1色阶DCN模型
class DCN_1(nn.Module):
    def __init__(self):
        super(DCN_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn8 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn3(self.conv3(x))
        x = self.relu(x)
        x = self.bn4(self.conv4(x))
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn5(self.conv5(x))
        x = self.relu(x)
        x = self.bn6(self.conv6(x))
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.bn7(self.fc1(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn8(self.fc2(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# 定义3色阶DCN模型
class DCN_3(nn.Module):
    def __init__(self):
        super(DCN_3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn8 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn3(self.conv3(x))
        x = self.relu(x)
        x = self.bn4(self.conv4(x))
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn5(self.conv5(x))
        x = self.relu(x)
        x = self.bn6(self.conv6(x))
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.bn7(self.fc1(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn8(self.fc2(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# 训练模型
def train(model, trainloader, testloader, criterion, optimizer, testset, device, e):
    for epoch in range(e):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()  # 将模型设置为训练模式
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('[Epoch %d, Batch %5d] Loss: %.3f | Accuracy: %.2f %%' % (
                epoch + 1, i + 1, running_loss / 100, 100 * correct / total))
            running_loss = 0.0
        # 在每个epoch结束后测试模型
        model.eval()  # 将模型设置为测试模式
        correct = 0
        labels_true = []
        labels_pred = []
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                labels_true.extend(labels.cpu().numpy())
                labels_pred.extend(predicted.cpu().numpy())
        accuracy = correct / len(testset)
        print('Accuracy of the network on the test images: %.2f %%' % (100 * accuracy))
        # 计算ARI和NMI
        ari = adjusted_rand_score(labels_true, labels_pred)
        nmi = normalized_mutual_info_score(labels_true, labels_pred)
        print('ARI: %.4f' % ari)
        print('NMI: %.4f' % nmi)


# 测试模型
def test(model, testloader, testset, device):
    model.eval()
    correct = 0
    labels_true = []
    labels_pred = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            labels_true.extend(labels.cpu().numpy())
            labels_pred.extend(predicted.cpu().numpy())

    accuracy = correct / len(testset)
    print('Accuracy of the network on the test images: %.2f %%' % (100 * accuracy))

    # 计算ARI和NMI
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)

    print('ARI: %.4f' % ari)
    print('NMI: %.4f' % nmi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="MNIST", choices=["MNIST", "USPS", "SVHN"])
    parser.add_argument('--lr', default=0.001, type=int)
    parser.add_argument('--momentum', default=0.9, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=int)
    parser.add_argument('--epoch', default=20, type=int)
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    epoch = args.epoch
    if args.dataset == "USPS":
        model = DCN_1().to(device)
    elif args.dataset == "MNIST":
        model = DCN_1().to(device)
    elif args.dataset == "SVHN":
        model = DCN_3().to(device)
    else:
        print("invalid dataset")
        exit()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    trainloader = trainloader(args.dataset)
    testloader = testloader(args.dataset)
    testset = testset(args.dataset)
    train(model, trainloader, testloader, criterion, optimizer, testset, device, epoch)
    test(model, testloader, testset, device)
