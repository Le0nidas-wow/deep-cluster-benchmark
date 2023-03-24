import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
import argparse
from sklearn.cluster import KMeans
import numpy as np


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


class modelsource(nn.Module):
    def __init__(self):
        super(modelsource, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
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
        x = x


# 定义1色阶DCN模型
class DCN_1(nn.Module):
    def __init__(self):
        super(DCN_1, self).__init__()
        self.modelsource = modelsource()

    def forward(self, x):
        x = self.modelsource.bn1(self.modelsource.conv1(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.bn2(self.modelsource.conv2(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.pool(x)
        x = self.modelsource.bn3(self.modelsource.conv3(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.bn4(self.modelsource.conv4(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.pool(x)
        x = self.modelsource.bn5(self.modelsource.conv5(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.bn6(self.modelsource.conv6(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.modelsource.bn7(self.modelsource.fc1(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.dropout(x)
        x = self.modelsource.bn8(self.modelsource.fc2(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.dropout(x)
        x = self.modelsource.fc3(x)
        return x


# 定义3色阶DCN模型
class DCN_3(nn.Module):
    def __init__(self):
        super(DCN_3, self).__init__()
        self.modelsource = modelsource()

    def forward(self, x):
        x = self.modelsource.bn1(self.modelsource.conv0(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.bn2(self.modelsource.conv2(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.pool(x)
        x = self.modelsource.bn3(self.modelsource.conv3(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.bn4(self.modelsource.conv4(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.pool(x)
        x = self.modelsource.bn5(self.modelsource.conv5(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.bn6(self.modelsource.conv6(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.modelsource.bn7(self.modelsource.fc1(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.dropout(x)
        x = self.modelsource.bn8(self.modelsource.fc2(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.dropout(x)
        x = self.modelsource.fc3(x)
        return x


# 定义3色阶DEKM模型
class DEKM_3(nn.Module):
    def __init__(self):
        super(DEKM_3, self).__init__()
        self.num_clusters = 10
        self.dcn3 = DCN_3()
        self.centers = nn.Parameter(torch.Tensor(self.num_clusters, 10))

    def forward(self, x):
        x = self.dcn3.forward(x)
        return x

    def get_loss(self, inputs, targets):
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        return loss

    def init_centers(self, trainloader):
        # 用 KMeans 算法初始化聚类中心
        kmeans = KMeans(n_clusters=self.num_clusters)
        with torch.no_grad():
            # 获取训练集上的所有特征向量
            features = []
            for data, _ in trainloader:
                data = data.to(device)
                features.append(self.dcn3(data).reshape(data.shape[0], -1).cpu().numpy())
            features = np.concatenate(features, axis=0)
            # 进行聚类
            kmeans.fit(features)
            # 将聚类中心设置为模型的参数
            self.centers.data.copy_(torch.from_numpy(kmeans.cluster_centers_).to(device))


# 定义1色阶DEKM模型
class DEKM_1(nn.Module):
    def __init__(self):
        super(DEKM_1, self).__init__()
        self.num_clusters = 10
        self.dcn1 = DCN_1()
        self.centers = nn.Parameter(torch.Tensor(self.num_clusters, 10))

    def forward(self, x):
        x = self.dcn1.forward(x)
        return x

    def get_loss(self, inputs, targets):
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        return loss

    def init_centers(self, trainloader):
        # 用 KMeans 算法初始化聚类中心
        kmeans = KMeans(n_clusters=self.num_clusters)
        with torch.no_grad():
            # 获取训练集上的所有特征向量
            features = []
            for data, _ in trainloader:
                data = data.to(device)
                features.append(self.dcn1(data).reshape(data.shape[0], -1).cpu().numpy())
            features = np.concatenate(features, axis=0)
            # 进行聚类
            kmeans.fit(features)
            # 将聚类中心设置为模型的参数
            self.centers.data.copy_(torch.from_numpy(kmeans.cluster_centers_).to(device))


class IDEC_1(nn.Module):
    def __init__(self):
        super(IDEC_1, self).__init__()
        self.num_clusters = 10
        self.alpha = 1.0
        self.dcn1 = DCN_1()  # 使用DEKM_1模型作为编码器
        self.modelsource = modelsource()
        self.encoder = nn.Sequential(
            nn.Linear(128 * 4 * 4, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 128 * 4 * 4),
            nn.Tanh()
        )
        self.centers = nn.Parameter(torch.Tensor(10, 10))
        self.loss_fn = nn.MSELoss()  # 使用均方误差作为自编码器的损失函数

    def forward(self, x):
        x = self.modelsource.bn1(self.modelsource.conv1(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.bn2(self.modelsource.conv2(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.pool(x)
        x = self.modelsource.bn3(self.modelsource.conv3(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.bn4(self.modelsource.conv4(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.pool(x)
        x = self.modelsource.bn5(self.modelsource.conv5(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.bn6(self.modelsource.conv6(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        # 计算自编码器的重构误差
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        recon_loss = self.loss_fn(decoded, x)
        # 计算聚类误差和总误差
        distances = torch.sum((encoded.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=2)
        probs = nn.functional.softmax(-distances, dim=1)
        cluster_loss = torch.mean(torch.sum(probs * torch.log(probs), dim=1))
        total_loss = recon_loss + self.alpha * cluster_loss
        # 返回聚类结果和总误差
        return encoded, probs, total_loss

    def init_centers(self, trainloader):
        # 用 KMeans 算法初始化聚类中心
        kmeans = KMeans(n_clusters=self.num_clusters)
        with torch.no_grad():
            # 获取训练集上的所有特征向量
            features = []
            for data, _ in trainloader:
                data = data.to(device)
                features.append(self.dcn1(data).reshape(data.shape[0], -1).cpu().numpy())
            features = np.concatenate(features, axis=0)
            # 进行聚类
            kmeans.fit(features)
            # 将聚类中心设置为模型的参数
            self.centers.data.copy_(torch.from_numpy(kmeans.cluster_centers_).to(device))


class IDEC_3(nn.Module):
    def __init__(self):
        super(IDEC_3, self).__init__()
        self.num_clusters = 10
        self.alpha = 0.5
        self.dcn3 = DCN_3()  # 使用DEKM_1模型作为编码器
        self.modelsource = modelsource()
        self.encoder = nn.Sequential(
            nn.Linear(128 * 4 * 4, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 128 * 4 * 4),
            nn.Tanh()
        )
        self.centers = nn.Parameter(torch.Tensor(10, 10))
        self.loss_fn = nn.MSELoss()  # 使用均方误差作为自编码器的损失函数

    def forward(self, x):
        x = self.modelsource.bn1(self.modelsource.conv0(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.bn2(self.modelsource.conv2(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.pool(x)
        x = self.modelsource.bn3(self.modelsource.conv3(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.bn4(self.modelsource.conv4(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.pool(x)
        x = self.modelsource.bn5(self.modelsource.conv5(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.bn6(self.modelsource.conv6(x))
        x = self.modelsource.relu(x)
        x = self.modelsource.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        # 计算自编码器的重构误差
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        recon_loss = self.loss_fn(decoded, x)
        # 计算聚类误差和总误差
        distances = torch.sum((encoded.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=2)
        probs = nn.functional.softmax(-distances, dim=1)
        cluster_loss = torch.mean(torch.sum(probs * torch.log(probs), dim=1))
        total_loss = recon_loss + self.alpha * cluster_loss
        # 返回聚类结果和总误差
        return encoded, probs, total_loss

    def init_centers(self, trainloader):
        # 用 KMeans 算法初始化聚类中心
        kmeans = KMeans(n_clusters=self.num_clusters)
        with torch.no_grad():
            # 获取训练集上的所有特征向量
            features = []
            for data, _ in trainloader:
                data = data.to(device)
                features.append(self.dcn3(data).reshape(data.shape[0], -1).cpu().numpy())
            features = np.concatenate(features, axis=0)
            # 进行聚类
            kmeans.fit(features)
            # 将聚类中心设置为模型的参数
            self.centers.data.copy_(torch.from_numpy(kmeans.cluster_centers_).to(device))


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
            outputs, probs, loss = model(images)
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


# 训练模型
def DCNtrain(model, trainloader, testloader, criterion, optimizer, testset, device, e):
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
        print("[Epoch %d] Evaluating model..." % (epoch + 1))
        test(model, testloader, testset, device)


def DEKMtrain(model, trainloader, testloader, optimizer, testset, device, epoch):
    model.init_centers(trainloader)
    for epoch_i in range(epoch):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = model.get_loss(inputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            print('[Epoch %d, Batch %5d] Loss: %.3f | Accuracy: %.2f %%' % (
                epoch_i + 1, batch_idx + 1, running_loss / (batch_idx + 1), 100 * correct / total))
        # Evaluate the model after every epoch
        print("[Epoch %d] Evaluating model..." % (epoch_i + 1))
        test(model, testloader, testset, device)


def IDECtrain(model, trainloader, testloader, optimizer, testset, device, epoch):
    model.init_centers(trainloader)
    model.train()
    for epoch_i in range(epoch):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (data, target) in enumerate(trainloader):
            data = data.to(device)
            target = target.to(device)
            # 正向传播计算聚类结果和总误差
            encoded, probs, loss = model(data)
            # 反向传播更新模型参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 逐步优化聚类中心
            new_centers = torch.zeros_like(model.centers)
            for j in range(model.num_clusters):
                mask = probs[:, j].unsqueeze(1)
                sum_mask = torch.sum(mask)
                if sum_mask > 0:
                    new_centers[j] = torch.sum(mask * encoded.detach(), dim=0) / sum_mask
                else:
                    new_centers[j] = model.centers[j]
            model.centers.data = new_centers.data
            # 统计训练误差和损失
            running_loss += loss.item()
            _, predicted = torch.max(encoded.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            print('[Epoch %d, Batch %5d] Loss: %.3f | Accuracy: %.2f %%' % (
                epoch_i + 1, i + 1, running_loss / (i + 1), 100 * correct / total))
        # Evaluate the model after every epoch
        print("[Epoch %d] Evaluating model..." % (epoch_i + 1))
        test(model, testloader, testset, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="IDEC", choices=["DEKM", "IDEC", "DCN"])
    parser.add_argument('--dataset', default="SVHN", choices=["MNIST", "USPS", "SVHN"])
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
    if args.dataset == "USPS" or args.dataset == "MNIST":
        if args.model == "DEKM":
            model = DEKM_1().to(device)
        elif args.model == "DCN":
            model = DCN_1().to(device)
        elif args.model == "IDEC":
            model = IDEC_1().to(device)
        else:
            print("invalid model name")
    elif args.dataset == "SVHN":
        if args.model == "DEKM":
            model = DEKM_3().to(device)
        elif args.model == "DCN":
            model = DCN_3().to(device)
        elif args.model == "IDEC":
            model = IDEC_3().to(device)
        else:
            print("invalid model name")
    else:
        print("invalid dataset name")
        exit()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    trainloader = trainloader(args.dataset)
    testloader = testloader(args.dataset)
    testset = testset(args.dataset)
    if args.model == "DEKM":
        DEKMtrain(model, trainloader, testloader, optimizer, testset, device, epoch)
    elif args.model == "DCN":
        DCNtrain(model, trainloader, testloader, criterion, optimizer, testset, device, epoch)
    elif args.model == "IDEC":
        IDECtrain(model, trainloader, testloader, optimizer, testset, device, epoch)
    test(model, testloader, testset, device)
