import wandb
import time
from torch import optim
import os
import numpy as np

from GraphViT import *
from validation import *
from train_epoch import *
from dataLoading import *
from evaluation import *

db = ['CIFAR10']
for database in db:
    if (database == 'MNIST'):
        projectName = 'MNIST'
        train_loader, val_loader, test_loader, parameters = dataLoadingMNIST(image_size=28, patch_size=7, num_classes=10,
                                                                             channels=1, dim=64, depth=6, heads=8,
                                                                             mlp_dim=128, epochs=20)
        image_size, patch_size, num_classes, channels, dim, depth, heads, mlp_dim, epochs = parameters
        #default: image_size=32, patch_size=8, num_classes=100, channels=3, dim=64, depth=6, heads=8, mlp_dim=128
    elif (database == 'CIFAR10'):
        projectName = 'CIFAR10'
        train_loader, val_loader, test_loader, parameters = dataLoaderCIFAR10(image_size=32, patch_size=8, num_classes=10,
                                                                             channels=3, dim=128, depth=8, heads=8,
                                                                             mlp_dim=256, epochs=20)
        # image_size = 32, patch_size = 8, num_classes = 10,
        # channels = 3, dim = 128, depth = 4, heads = 8,
        # mlp_dim = 1024, epochs = 50
        image_size, patch_size, num_classes, channels, dim, depth, heads, mlp_dim, epochs = parameters
        # image_size = 32, patch_size = 4, num_classes = 10,
        # channels = 3, dim = 256, depth = 8, heads = 12,
        # mlp_dim = 1024, epochs = 50
    elif (database == 'CIFAR100'):
        projectName = 'CIFAR100'
        train_loader, val_loader, test_loader, parameters = dataLoaderCIFAR100(image_size=32, patch_size=4, num_classes=100,
                                                                             channels=3, dim=256, depth=8, heads=12,
                                                                             mlp_dim=1024, epochs=50)
        image_size, patch_size, num_classes, channels, dim, depth, heads, mlp_dim, epochs = parameters
        # image_size = 32, patch_size = 4, num_classes = 100,
        # channels = 3, dim = 256, depth = 8, heads = 12,
        # mlp_dim = 1024, epochs = 50



    wandb.init(entity='njmarko', project=projectName)
    N_EPOCHS = epochs

    start_time = time.time()
    model = GraphViT(image_size, patch_size, num_classes,  dim, depth, heads, mlp_dim, channels)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    early_stop_tolerance = 10e-4
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, cycle_momentum=False, step_size_up = 1000)

    train_loss_history, val_loss_history, test_loss_history = [], [], []
    best_model_acc = -1
    best_epoch = -1
    for epoch in range(1, N_EPOCHS + 1):
        early_stopping = 0
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_loader, train_loss_history, scheduler)
        acc = validate(model, val_loader, val_loss_history)

        if acc > best_model_acc:
            best_model_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join('models/cifar10-gat', f'train-{projectName}-{epoch}-acc{acc}'))

        if len(train_loss_history) > 2 and np.isclose(train_loss_history[-2], train_loss_history[-1], atol=early_stop_tolerance):
            early_stopping += 1
            if (early_stopping == 5):
                print(f"Early stop with tolerance {early_stop_tolerance} for losses {train_loss_history[-2]} and {train_loss_history[-1]}")
                break
        else:
            early_stopping = 0

    # Testiranje modela
    path = f'models/cifar10-gat/train-{projectName}-{best_epoch}-acc{best_model_acc}'
    print(path)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    evaluate(model, test_loader, test_loss_history)

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')