import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
from torchvision import transforms
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler


# Model A - two hidden layers with Relu activation function.
class ModelA(nn.Module):
    def __init__(self, image_size, epoch=10, lr=0.001):
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.epoch = epoch
        self.lr = lr

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Model B - two hidden layers with Relu activation function and dropout of 25%.
class ModelB(nn.Module):
    def __init__(self, image_size, epoch=10, lr=0.001, p=0.25):
        super(ModelB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.epoch = epoch
        self.lr = lr
        self.p = p

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.p)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Model C - two hidden layers with Relu activation function and Batch Normalization layers.
class ModelC(nn.Module):
    def __init__(self, image_size, epoch=10, lr=0.001):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.bn = nn.BatchNorm1d(100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.epoch = epoch
        self.lr = lr

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn(self.fc0(x)))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Model D - five hidden layers with Relu activation function.
class ModelD(nn.Module):
    def __init__(self, image_size, epoch=10, lr=0.001):
        super(ModelD, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.epoch = epoch
        self.lr = lr

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


# Model E - five hidden layers with sigmoid activation function.
class ModelE(nn.Module):
    def __init__(self, image_size, epoch=10, lr=0.01):
        super(ModelE, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.epoch = epoch
        self.lr = lr

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


def main():
    # set up the train and test sets
    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_from_file = np.loadtxt("test_x")
    test_from_file /= 255
    test_from_file = tr(test_from_file)[0].float()
    test_from_file_loader = torch.utils.data.DataLoader(test_from_file, batch_size=64, shuffle=False)
    torch.manual_seed(0)
    train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=tr)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    # create the model train it and write the results to the file.
    model = ModelC(784)
    train(model, train_loader)
    results = get_results_from_model(model, test_from_file_loader)
    write_results_to_file("test_y", results)


# split the data to a training set and a validation set return their loaders.
def split_dataset(train_data):
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))
    np.random.seed(0)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=64, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(train_data, batch_size=64, sampler=valid_sampler)
    return train_loader, validation_loader


# train a given model
def train(model, train_loader):
    for e in range(model.epoch):
        print("Train")
        optimizer = optim.Adam(model.parameters(), lr=model.lr)
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()


# test the model by checking it's predictions if a validation set is given use one epoch.
def test(model, test_loader, is_validation=False):
    if is_validation:
        epoch = 1
    else:
        epoch = model.epoch
    with torch.no_grad():
        for e in range(epoch):
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            for data, labels in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, labels).item()
                for idx, i in enumerate(output):
                    if torch.argmax(i) == labels[idx]:
                        correct += 1
                    total += 1
    accuracy = round(correct / total, 3)
    loss = round(test_loss / len(test_loader), 3)
    return accuracy, loss


# train the model with a validation set, return the accuracy and loss of the training and validation sets.
def train_with_validation(model, train_loader, validation_loader):
    train_acc = []
    train_loss = []
    validation_acc = []
    validation_loss = []
    for e in range(model.epoch):
        print("Train")
        current_loss = 0
        correct = 0
        total = 0
        optimizer = optim.Adam(model.parameters(), lr=model.lr)
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            for idx, i in enumerate(output):
                if torch.argmax(i) == labels[idx]:
                    correct += 1
                total += 1
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
        train_curr_acc = round(correct / total, 3)
        train_curr_loss = round(current_loss / len(train_loader), 3)
        train_acc.append(train_curr_acc)
        train_loss.append(train_curr_loss)
        with torch.no_grad():
            valid_curr_acc, valid_curr_loss = test(model, validation_loader, is_validation=True)
            validation_acc.append(valid_curr_acc)
            validation_loss.append(valid_curr_loss)
    return train_acc, train_loss, validation_acc, validation_loss


# get the model's predictions for a given test set.
def get_results_from_model(model, test_loader):
    predictions = []
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            output = model(data)
            for idx, i in enumerate(output):
                predictions.append(torch.argmax(i).item())
    predictions = np.asarray(predictions).astype(int)
    return predictions


# plot the accuracy and loss graphs.
def plot_graphs(train_data, validation_data, graph, epoch=10):
    x = list(range(1, epoch + 1))
    y1 = train_data
    plt.plot(x, y1, label="train")
    y2 = validation_data
    plt.plot(x, y2, label="validation")
    plt.xlabel('Epoch')
    if graph == "acc":
        plt.ylabel('Accuracy')
    else:
        plt.ylabel('Loss')
    if graph == "acc":
        plt.title('Train and Validation accuracy')
    else:
        plt.title('Train and Validation loss')
    plt.legend()
    plt.show()


# write the model's results to the file.
def write_results_to_file(file_name, results):
    f = open(file_name, "w")
    last_line = results.shape[0] - 1
    for idx, res in enumerate(results):
        f.write(str(res))
        if idx != last_line:
            f.write("\n")
    f.close()


if __name__ == "__main__":
    main()
