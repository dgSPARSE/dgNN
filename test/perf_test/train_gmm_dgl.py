import argparse
import time
import torch
import torch.nn as nn
from scipy import sparse
from dgl.data import load_data
import GPUtil
from dgl.nn.pytorch.conv import GMMConv
import scipy.sparse as sp

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

class MoNet(nn.Module):
    def __init__(self,
                 graph,
                 in_feats,
                 n_hidden,
                 out_feats,
                 n_layers,
                 dim,
                 n_kernels,
                 dropout):
        super(MoNet, self).__init__()
        self.graph=graph
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Input layer
        self.layers.append(
            GMMConv(in_feats, n_hidden, dim, n_kernels))
        self.pseudo_proj.append(
            nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        # Hidden layer
        for _ in range(n_layers - 1):
            self.layers.append(GMMConv(n_hidden, n_hidden, dim, n_kernels))
            self.pseudo_proj.append(
                nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        # Output layer
        self.layers.append(GMMConv(n_hidden, out_feats, dim, n_kernels))
        self.pseudo_proj.append(
            nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat, pseudo):
        h = feat
        for i in range(len(self.layers)):
            if i != 0:
                h = self.dropout(h)
            h = self.layers[i](self.graph,h,self.pseudo_proj[i](pseudo))
        return h

def evaluate(model, features, pseudo, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features, pseudo)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def preprocess(g, args):
    rowptr, colind, _ = g.adj_sparse('csc') #reverse
    numlist = torch.arange(colind.size(0), device=rowptr.device, dtype=torch.int32)
    csr_sp = sparse.csr_matrix((numlist, colind, rowptr), shape=[rowptr.size(0)-1, rowptr.size(0)-1])
    csc_sp = csr_sp.tocsc()
    permute = torch.from_numpy(csc_sp.data)
    colptr = torch.from_numpy(csc_sp.indptr)
    rowind = torch.from_numpy(csc_sp.indices)
    permute = permute.int().to(args.gpu)
    rowptr = rowptr.int().to(args.gpu)
    colind = colind.int().to(args.gpu)
    colptr = colptr.int().to(args.gpu)
    rowind = rowind.int().to(args.gpu)
    return rowptr, colind, colptr, rowind, permute

def main(args):
    # load and preprocess dataset
    data = load_data(args)
    g = data[0]

    features = g.ndata['feat'].to(args.gpu)
    labels = g.ndata['label'].to(args.gpu)
    train_mask = g.ndata['train_mask'].to(args.gpu)
    val_mask = g.ndata['val_mask'].to(args.gpu)
    test_mask = g.ndata['test_mask'].to(args.gpu)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    # graph preprocess and calculate normalization factor
    g = g.remove_self_loop().add_self_loop().to(args.gpu)
    n_edges = g.number_of_edges()
    # rowptr, colind, colptr, rowind, permute = preprocess(g, args)

    us, vs = g.edges(order='eid')
    udeg, vdeg = 1 / torch.sqrt(g.in_degrees(us).float()), 1 / torch.sqrt(g.in_degrees(vs).float())
    pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1).to(args.gpu)

    # create GraphSAGE model
    model = MoNet(g,in_feats,args.n_hidden,n_classes,args.n_layers,args.pseudo_dim,args.n_kernels,args.dropout).to(args.gpu)

    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print('warm up')
    maxMemory = 0
    for _ in range(10):
        model.train()
        logits = model(features, pseudo)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
        GPUs = GPUtil.getGPUs()
        maxMemory = max(GPUs[args.gpu].memoryUsed, maxMemory) 
  
    print('profile training')
    model.train()
    torch.cuda.synchronize()
    start=time.time()
    for _ in range(args.n_epochs):
        logits=model(features, pseudo)
        loss=loss_fcn(logits[train_mask],labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
        print(loss.item())
    torch.cuda.synchronize()
    end=time.time()
    train_time=(end-start)/args.n_epochs

    print('profile inference')
    model.eval()
    torch.cuda.synchronize()
    start=time.time()
    for epoch in range(args.n_epochs):  
        with torch.no_grad():
            logits=model(features, pseudo)
    torch.cuda.synchronize()
    end=time.time()
    inference_time=(end-start)/args.n_epochs
    
    logits = logits[test_mask]  
    acc=accuracy(logits, labels[test_mask])
    print("Test Accuracy {:.4f}".format(acc))
    print(f'max memory:{maxMemory}MB')
    print("train time:",train_time)
    print("inference time:",inference_time)

    if args.output!=None:
        with open("{}".format(args.output),'a') as f:
            print("train_GMM_dgl,{} pseudo_dim={} n_kernels={} hidden_dim={},{:f}s,{:f}s,{}MB".format(args.dataset,args.pseudo_dim,args.n_kernels,args.n_hidden,train_time,inference_time,maxMemory),file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoNet on citation network')
    parser.add_argument("--dataset",type=str,default="cora")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")                    
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--pseudo-dim", type=int, default=2,
                        help="Pseudo coordinate dimensions in GMMConv, 2 for cora and 3 for pubmed")
    parser.add_argument("--n-kernels", type=int, default=3,
                        help="Number of kernels in GMMConv layer")
    parser.add_argument('--output',type=str,default=None,
                        help="output file")
    args = parser.parse_args()
    print(args)

    main(args)