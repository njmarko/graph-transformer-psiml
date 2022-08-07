import torch
from sklearn import metrics
import torch.nn.functional as F
import wandb
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate(model, data_loader, loss_history):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    global_target = np.array([])
    global_pred = np.array([])

    with torch.no_grad():
        for data, target in data_loader:
            res = model(data)
            res = res.to(device)
            output = F.log_softmax(res, dim=1)
            target = target.to(device)
            output = output.to(device)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

            target = target.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()

            global_target = np.concatenate((global_target, target))
            global_pred = np.concatenate((global_pred, pred))

    avg_loss = total_loss / total_samples
    acc = 100.0 * correct_samples / total_samples
    loss_history.append(avg_loss)

    f1_score = metrics.f1_score(global_target, global_pred, average='micro')
    precision = metrics.precision_score(global_target, global_pred, average='micro')
    recall = metrics.recall_score(global_target, global_pred, average='micro')

    # wandb.log({
    #     'test_loss': loss.item(),
    #     'accuracy': acc,
    #     'test_f1_score': f1_score,
    #     'precision': precision,
    #     'recall': recall
    # })
    print('\nTest loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(acc) + '%)  Precision: ' + '{:4.2f}'.format(precision) +
          '  Recall: ' + '{:4.2f}'.format(recall) + '\n')

    cm = metrics.confusion_matrix(global_target, global_pred)
    print(f'Confusion matrix:\n {cm}')