import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)  # Reshape for fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Validation loop
def validate_model(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def get_loaders(path):
    # Define the batch size for training, validation, and testing
    batch_size = 64

    # Define data transformations (you can modify these as needed)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root=path+'/data', train=True,
                                            download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(root=path+'/data', train=False,
                                        download=True, transform=transform_test)

    # Split the training dataset into training and validation sets
    val_size = int(0.2 * len(trainset))
    train_size = len(trainset) - val_size


    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

    # Create DataLoader instances for train, validation, and test datasets
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Class labels for CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader,valloader,testloader