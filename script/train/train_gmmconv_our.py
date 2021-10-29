import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from scipy import sparse
from dgl.data import register_data_args, load_data
import sys
# import GPUtil
sys.path.append('../..')
from layers.gmmconv_layer import GMMConv
# from util.indicator import *
import scipy.sparse as sp
from torch.utils.cpp_extension import load

class MoNet(nn.Module):
    def __init__(self,
                 rowptr,
                 colind,
                 colptr,
                 rowind,
                 permute,
                 in_feats,
                 n_hidden,
                 out_feats,
                 n_layers,
                 dim,
                 n_kernels,
                 dropout):
        super(MoNet, self).__init__()
        self.rowptr = rowptr
        self.colind = colind
        self.colptr = colptr
        self.rowind = rowind
        self.permute = permute
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
            h = self.layers[i](
                self.rowptr, self.colind, self.colptr, self.rowind, self.permute, h, self.pseudo_proj[i](pseudo))
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
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        # g = g.to(args.gpu)
    features = g.ndata['feat'].to(args.gpu)
    labels = g.ndata['label'].to(args.gpu)
    train_mask = g.ndata['train_mask'].to(args.gpu)
    val_mask = g.ndata['val_mask'].to(args.gpu)
    test_mask = g.ndata['test_mask'].to(args.gpu)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.sum().item(),
           val_mask.sum().item(),
           test_mask.sum().item()))

    # graph preprocess and calculate normalization factor
    g = g.remove_self_loop().add_self_loop()
    n_edges = g.number_of_edges()
    rowptr, colind, colptr, rowind, permute = preprocess(g, args)

    us, vs = g.edges(order='eid')
    udeg, vdeg = 1 / torch.sqrt(g.in_degrees(us).float()), 1 / torch.sqrt(g.in_degrees(vs).float())
    pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1).to(args.gpu)

    # create GraphSAGE model
    model = MoNet(rowptr,
                  colind,
                  colptr,
                  rowind,
                  permute,
                  in_feats,
                  args.n_hidden,
                  n_classes,
                  args.n_layers,
                  args.pseudo_dim,
                  args.n_kernels,
                  args.dropout
                  )

    if cuda:
        model.to(args.gpu)
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    for _ in range(5):
        model(features, pseudo)

    torch.cuda.synchronize()
    t1 = time.time()
    for epoch in range(args.n_epochs):
        model.train()
        logits = model(features, pseudo)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = evaluate(model, features, pseudo, labels, val_mask)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | ".format(epoch, loss.item(),
                                            acc))
    torch.cuda.synchronize()
    t2 = time.time()
    print("one epoch time: {:.4f} s".format((t2 - t1)/args.n_epochs))
    acc = evaluate(model, features, pseudo, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoNet on citation network')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=2,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=0,
                        help="number of hidden gcn layers")
    parser.add_argument("--pseudo-dim", type=int, default=2,
                        help="Pseudo coordinate dimensions in GMMConv, 2 for cora and 3 for pubmed")
    parser.add_argument("--n-kernels", type=int, default=3,
                        help="Number of kernels in GMMConv layer")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--profileio", type=int, default=0,
                    help="1 for profile io")
    args = parser.parse_args()
    print(args)

    main(args)