#!/usr/bin/env python
# coding: utf-8

# ## Custom GNN Transformer architecture

# In[50]:


import torch
import torch.nn as nn
import time
from torch import optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
import torchvision
from einops import rearrange
import wandb
from sklearn import metrics 
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)
    
class QuickFix(nn.Module):
    def __init__(self, dim, heads, fn):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.linear = nn.Linear(dim * heads, dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.linear(self.fn(x, **kwargs))


# In[52]:


# Out implementation of GAT Transformer

class GATTransformer(torch.nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, QuickFix(dim, heads, GATv2Conv(in_channels=dim, out_channels=dim, heads=heads, add_self_loops=False)))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))


    def forward(self, data):
        x, edge_index = data

        for attn, ff in self.layers:
            x = attn(x=x, edge_index=edge_index)
            x = ff(x)

        return x


# In[53]:


class GraphViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.dim = dim

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = GATTransformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = x.to(device)
        x = self.patch_to_embedding(x)
        
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        cls_tokens = cls_tokens.to(device)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        
    
        x_shape = x.shape
        edge_index = torch.ones((2, x.shape[1]**2*x.shape[0]), dtype=torch.long)
        for i in range(x.shape[0]*x.shape[1]):
            for j in range(x.shape[1]):
                edge_index[0,i*x.shape[1] + j] = i
                edge_index[1,i*x.shape[1] + j] = x.shape[1]*(i // x.shape[1]) + j
        x = x.view((x.shape[0]*x.shape[1], x.shape[2]))
        edge_index = edge_index.to(device)

        
        x = self.transformer([x, edge_index])
        x = x.view(x_shape)
        
        x = self.to_cls_token(x[:,0])
        x = x.to(device)
        
        out = self.mlp_head(x)
        
        return out


# In[54]:


torch.manual_seed(42)
# wandb.init(entity = 'njmarko', project = 'gat')

DOWNLOAD_PATH = '/data/mnist'
BATCH_SIZE_TRAIN = 200
BATCH_SIZE_TEST = 1000

transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))])
train_set = torchvision.datasets.CIFAR100(DOWNLOAD_PATH, train=True, download=True,
                                       transform=transform_mnist)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

test_set = torchvision.datasets.CIFAR100(DOWNLOAD_PATH, train=False, download=True,
                                      transform=transform_mnist)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)


# In[66]:


def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()
    running_loss = 0.0

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
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())
            wandb.log({
                'train_loss': loss.item(),
                'train_f1_score': f1_score
            })
            
    acc = 100.0 * correct_samples / total_samples
    print(f'Accuracy: ' + '{:4.2f}'.format(acc) + '%')


# In[67]:


def evaluate(model, data_loader, loss_history):
    model.eval()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

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

    avg_loss = total_loss / total_samples
    acc = 100.0 * correct_samples / total_samples
    loss_history.append(avg_loss)
    
    target = target.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    f1_score = metrics.f1_score(target, pred, average='micro')
    
    wandb.log({
        'test_loss': loss.item(),
        'test_f1_score': f1_score
    })
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(acc) + '%)\n')


dim = [96, 96, 128]
lr = [0.03, 0.003, 0.003]
for i in range(3):
    wandb.init(entity='njmarko', project=f'gat{i}')
    N_EPOCHS = 40

    start_time = time.time()
    model = GraphViT(image_size=32, patch_size=8, num_classes=100, channels=3,
                     dim=dim[i], depth=6, heads=8, mlp_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=lr[i])
    model = model.to(device)

    # learning rate decay
    train_loss_history, test_loss_history = [], []
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_loader, train_loss_history)
        evaluate(model, test_loader, test_loss_history)
        torch.save(model.state_dict(), os.path.join('models', f'train-{i}-{epoch}'))
        # To do: early stopping
        # if len(train_loss_history) > 2 and np.isclose(train_loss_history[-2], train_loss_history[-1], atol=10e-3): break

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')




