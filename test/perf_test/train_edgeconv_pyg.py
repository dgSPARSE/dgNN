import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import KNNGraph

# from dgNN.layers.edgeconv_layer import EdgeConv
import time
import numpy as np
import GPUtil

edgeconv_time=[]
from typing import Callable, Optional, Union
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor

class EdgeConv_pyg(MessagePassing):
    def __init__(self, in_channels,out_channels):
        super().__init__(aggr='max')
        self.theta = nn.Linear(in_channels,out_channels,bias=False)
        self.phi=nn.Linear(in_channels,out_channels,bias=False)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)

        rst_pyg=self.propagate(edge_index, x=x, size=None)

        return rst_pyg

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.theta(x_j-x_i)+self.phi(x_i)

class Model(nn.Module):
    def __init__(self, k, feature_dims, emb_dims, output_classes, input_dims=3,
                 dropout_prob=0.5):
        super(Model, self).__init__()
        self.k=k
        self.nng = KNNGraph(k)
        self.conv = nn.ModuleList()

        self.num_layers = len(feature_dims)
        for i in range(self.num_layers):
            self.conv.append(EdgeConv_pyg(
                feature_dims[i - 1] if i > 0 else input_dims,
                feature_dims[i]))

        self.proj = nn.Linear(sum(feature_dims), emb_dims[0])

        self.embs = nn.ModuleList()
        self.bn_embs = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.num_embs = len(emb_dims) - 1
        for i in range(1, self.num_embs + 1):
            self.embs.append(nn.Linear(
                # * 2 because of concatenation of max- and mean-pooling
                emb_dims[i - 1] if i > 1 else (emb_dims[i - 1] * 2),
                emb_dims[i]))
            self.bn_embs.append(nn.BatchNorm1d(emb_dims[i]))
            self.dropouts.append(nn.Dropout(dropout_prob))

        self.proj_output = nn.Linear(emb_dims[-1], output_classes)

    def forward(self, x):
        hs = []
        batch_size, n_points, x_dims = x.shape
        h = x

        for i in range(self.num_layers):
            g = self.nng(h)
            h = h.view(batch_size * n_points, -1)
            src,dst=g.edges(order='srcdst')
            edge_idx=torch.stack((src.long(),dst.long())).to(h.device)
            # print(edge_idx.shape)
            # print(torch.max(edge_idx))
            # print(torch.min(edge_idx))
            # print(h.shape)
            h = self.conv[i](h,edge_idx)
            h = F.leaky_relu(h, 0.2)
            h = h.view(batch_size, n_points, -1)
            hs.append(h)

        h = torch.cat(hs, 2)
        h = self.proj(h)
        h_max, _ = torch.max(h, 1)
        h_avg = torch.mean(h, 1)
        h = torch.cat([h_max, h_avg], 1)

        for i in range(self.num_embs):
            h = self.embs[i](h)
            h = self.bn_embs[i](h)
            h = F.leaky_relu(h, 0.2)
            h = self.dropouts[i](h)

        h = self.proj_output(h)
        return h


def compute_loss(logits, y, eps=0.2):
    num_classes = logits.shape[1]
    one_hot = torch.zeros_like(logits).scatter_(1, y.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
    log_prob = F.log_softmax(logits, 1)
    loss = -(one_hot * log_prob).sum(1).mean()
    return loss


import numpy as np
from torch.utils.data import Dataset

class ModelNet(object):
    def __init__(self, path, num_points):
        import h5py
        self.f = h5py.File(path)
        self.num_points = num_points

        self.n_train = self.f['train/data'].shape[0]
        self.n_valid = int(self.n_train / 5)
        self.n_train -= self.n_valid
        self.n_test = self.f['test/data'].shape[0]

    def train(self):
        return ModelNetDataset(self, 'train')

    def valid(self):
        return ModelNetDataset(self, 'valid')

    def test(self):
        return ModelNetDataset(self, 'test')

class ModelNetDataset(Dataset):
    def __init__(self, modelnet, mode):
        super(ModelNetDataset, self).__init__()
        self.num_points = modelnet.num_points
        self.mode = mode

        if mode == 'train':
            self.data = modelnet.f['train/data'][:modelnet.n_train]
            self.label = modelnet.f['train/label'][:modelnet.n_train]
        elif mode == 'valid':
            self.data = modelnet.f['train/data'][modelnet.n_train:]
            self.label = modelnet.f['train/label'][modelnet.n_train:]
        elif mode == 'test':
            self.data = modelnet.f['test/data'].value
            self.label = modelnet.f['test/label'].value

    def translate(self, x, scale=(2/3, 3/2), shift=(-0.2, 0.2)):
        xyz1 = np.random.uniform(low=scale[0], high=scale[1], size=[3])
        xyz2 = np.random.uniform(low=shift[0], high=shift[1], size=[3])
        x = np.add(np.multiply(x, xyz1), xyz2).astype('float32')
        return x

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        x = self.data[i][:self.num_points]
        y = self.label[i]
        if self.mode == 'train':
            x = self.translate(x)
            np.random.shuffle(x)
        return x, y

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dgl.data.utils import download, get_download_dir

from functools import partial
import tqdm
import urllib
import os
import argparse

from torch.autograd.profiler import profile

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, default='')
parser.add_argument('--load-model-path', type=str, default='')
parser.add_argument('--save-model-path', type=str, default='')
parser.add_argument('--num-epochs', type=int, default=10)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--k',type=int,default=40)
parser.add_argument('--output',type=str,default=None,
                        help="output file")
args = parser.parse_args()

print(args)
num_workers = args.num_workers
batch_size = args.batch_size
data_filename = 'modelnet40-sampled-2048.h5'
local_path = args.dataset_path or os.path.join(get_download_dir(), data_filename)

if not os.path.exists(local_path):
    download('https://data.dgl.ai/dataset/modelnet40-sampled-2048.h5', local_path)

CustomDataLoader = partial(
        DataLoader,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

def train(model, opt, scheduler, train_loader, dev):
    scheduler.step()

    model.train()

    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    with tqdm.tqdm(train_loader, ascii=True) as tq:
        for data, label in tq:
            num_examples = label.shape[0]
            data, label = data.to(dev), label.to(dev).squeeze().long()

            opt.zero_grad()
            logits = model(data)
            loss = compute_loss(logits, label)
            loss.backward()
            opt.step()


            _, preds = logits.max(1)

            num_batches += 1
            count += num_examples
            loss = loss.item()
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct

            tq.set_postfix({
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

def evaluate(model, test_loader, dev):
    model.eval()

    total_correct = 0
    count = 0

    with torch.no_grad():
        with tqdm.tqdm(test_loader, ascii=True) as tq:
            for data, label in tq:
                num_examples = label.shape[0]
                data, label = data.to(dev), label.to(dev).squeeze().long()
                logits = model(data)
                _, preds = logits.max(1)

                correct = (preds == label).sum().item()
                total_correct += correct
                count += num_examples

                tq.set_postfix({
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count)})

    return total_correct / count


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(args.k, [64, 64, 128, 256], [512, 512, 256], 40)
model = model.to(dev)
if args.load_model_path:
    model.load_state_dict(torch.load(args.load_model_path, map_location=dev))

opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, args.num_epochs, eta_min=0.001)

modelnet = ModelNet(local_path, 1024)

train_loader = CustomDataLoader(modelnet.train())
valid_loader = CustomDataLoader(modelnet.valid())
# test_loader = CustomDataLoader(modelnet.test())

maxMemory=0
for epoch in range(3):
    train(model, opt, scheduler, train_loader, dev)
    GPUs = GPUtil.getGPUs()
    maxMemory = max(GPUs[0].memoryUsed, maxMemory) 

torch.cuda.synchronize()
start=time.time()
for epoch in range(args.num_epochs):
    train(model, opt, scheduler, train_loader, dev)
torch.cuda.synchronize()
end=time.time()
train_time=(end-start)/args.num_epochs

torch.cuda.synchronize()
start=time.time()
for epoch in range(args.num_epochs):
    acc=evaluate(model,valid_loader,dev)
torch.cuda.synchronize()
end=time.time()
inference_time=(end-start)/args.num_epochs

print("Test Accuracy {:.4f}".format(acc))
print(f'max memory:{maxMemory}MB')
print("train time:",train_time)
print("inference time:",inference_time)

if args.output!=None:
    with open("{}".format(args.output),'a') as f:
        print(f"train_edgeconv_pyg,{args.batch_size} {args.k},{train_time}s,{inference_time}s,{maxMemory}MB,{acc}",file=f)
