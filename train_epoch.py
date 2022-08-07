import torch
from sklearn import metrics
import torch.nn.functional as F
import wandb
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_epoch(model, optimizer, data_loader, loss_history, scheduler):
    total_samples = len(data_loader.dataset)
    model.train()
    running_loss = 0.0
    old_time = time.time()

    correct_samples = 0
    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        target = target.to(device)
        output = output.to(device)

        _, pred = torch.max(output, dim=1)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        correct_samples += pred.eq(target).sum()
        target = target.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()

        #         print(output)
        #         print('trarget')
        #         print(target)
        #         print('pred')
        #         print(pred)

        f1_score = metrics.f1_score(target, pred, average='micro')

        if i % 100 == 0:
            new_time = time.time()
            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())
            wandb.log({
                'train_loss': loss.item(),
                'train_f1_score': f1_score
            })
            print(f'Execution time: {new_time - old_time}')
            old_time = time.time()

    acc = 100.0 * correct_samples / total_samples
    wandb.log({
        'train_accuracy': acc
    })
    print(f'Accuracy: ' + '{:4.2f}'.format(acc) + '%')
    return acc