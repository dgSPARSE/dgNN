from scipy import sparse
import argparse
import time
# import numpy as np
# import networkx as nx
import torch
# import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import scipy._build_utils.system_info
import sys
# import GPUtil
sys.path.append('../..')
# from util.indicator import *
from layers.gmmconv_layer import GMMConv


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
    n_nodes = g.number_of_nodes()
    features = torch.rand(n_nodes, args.in_feat).to(args.gpu)
    n_edges = data.graph.number_of_edges()

    # graph preprocess and calculate normalization factor
    g = g.remove_self_loop().add_self_loop()
    n_edges = g.number_of_edges()
    rowptr, colind, colptr, rowind, permute = preprocess(g, args)

    pseudo = torch.rand(n_edges, args.pseudo_dim).to(args.gpu)
    model = GMMConv(args.in_feat, args.out_feat, args.pseudo_dim, args.n_kernels)

    if cuda:
        model.cuda()
    

    # warmup 
    # maxMemory = 0
    for _ in range(5):
        model(rowptr, colind, colptr, rowind, permute, features, pseudo)


    torch.cuda.synchronize()
    start=time.time()

    for epoch in range(args.n_epochs):
        model(rowptr, colind, colptr, rowind, permute, features, pseudo)

    torch.cuda.synchronize()
    end=time.time()

    # print(maxMemory)
    print("gmmconv forward time:", (end-start)/args.n_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoNet on citation network')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--in-feat", type=int, default=16,
                        help="size of in feature")
    parser.add_argument("--out-feat", type=int, default=16,
                        help="size of out feature")
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