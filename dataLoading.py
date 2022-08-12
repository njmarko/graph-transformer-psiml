import torch
import torchvision

def dataLoadingMNIST(image_size, patch_size, num_classes, channels, dim, depth, heads, mlp_dim, epochs):
    #torch.manual_seed(42)

    DOWNLOAD_PATH = '/data/mnist'
    BATCH_SIZE_TRAIN = 250
    BATCH_SIZE_VAL = 250
    BATCH_SIZE_TEST = 250

    transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_data = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True, transform=transform_mnist)
    train_set, val_set = torch.utils.data.random_split(train_data, [50000, 10000])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE_VAL, shuffle=True)
    test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True, transform=transform_mnist)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

    parameter = [image_size, patch_size, num_classes, channels, dim, depth, heads, mlp_dim, epochs]

    return train_loader, val_loader, test_loader, parameter

def dataLoaderCIFAR10(image_size, patch_size, num_classes, channels, dim, depth, heads, mlp_dim, epochs):
    DOWNLOAD_PATH = '/data/cifar10'
    BATCH_SIZE_TRAIN = 250
    BATCH_SIZE_VAL = 250
    BATCH_SIZE_TEST = 250

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    transform_cifar10 = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.transforms.Normalize(mean, std)])
    train_data = torchvision.datasets.CIFAR10(DOWNLOAD_PATH, train=True, download=True, transform=transform_cifar10)
    train_set, val_set = torch.utils.data.random_split(train_data, [45000, 5000])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE_VAL, shuffle=True)
    test_set = torchvision.datasets.CIFAR10(DOWNLOAD_PATH, train=False, download=True, transform=transform_cifar10)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

    parameter = [image_size, patch_size, num_classes, channels, dim, depth, heads, mlp_dim, epochs]

    return train_loader, val_loader, test_loader, parameter
def dataLoaderCIFAR100(image_size, patch_size, num_classes, channels, dim, depth, heads, mlp_dim, epochs):
    DOWNLOAD_PATH = '/data/cifar100'
    BATCH_SIZE_TRAIN = 10
    BATCH_SIZE_VAL = 10
    BATCH_SIZE_TEST = 10

    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    transform_cifar100 = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.transforms.Normalize(mean, std)])
    train_data = torchvision.datasets.CIFAR100(DOWNLOAD_PATH, train=True, download=True, transform=transform_cifar100)
    train_set, val_set = torch.utils.data.random_split(train_data, [45000, 5000])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE_VAL, shuffle=True)
    test_set = torchvision.datasets.CIFAR100(DOWNLOAD_PATH, train=False, download=True, transform=transform_cifar100)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

    parameter = [image_size, patch_size, num_classes, channels, dim, depth, heads, mlp_dim, epochs]

    return train_loader, val_loader, test_loader, parameter