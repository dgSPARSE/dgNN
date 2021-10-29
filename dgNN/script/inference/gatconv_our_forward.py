import argparse
import time
import torch
import torch.nn.functional as F
import dgl
# import dgl.data

import torch.nn as nn
import sys
import GPUtil
sys.path.append('../..')
from layers.gatconv_layer import GATConv
from torch.autograd.profiler import profile
# from util.indicator import *


import scipy.sparse as sp
def load_dataset(args):
    if args.dataset == 'cora':
        data = dgl.data.CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = dgl.data.CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = dgl.data.PubmedGraphDataset()
    elif args.dataset == 'reddit':
        data = dgl.data.RedditDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    
    col,row=g.edges(order='srcdst')
    adj_csr = sp.csr_matrix((torch.ones(row.shape), (row, col)), shape=(g.num_nodes(), g.num_nodes()))
    
    row_ptr=torch.from_numpy(adj_csr.indptr)
    col_ind=torch.from_numpy(adj_csr.indices)

    adj_coo=adj_csr.tocoo()
    row_ind=torch.from_numpy(adj_coo.row)
    print('dataset verified:',torch.equal(col_ind,torch.from_numpy(adj_coo.col)))

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    n_classes=data.num_labels
    num_feats = features.shape[1]
    return row_ind,row_ptr,col_ind,features,labels,train_mask,val_mask,test_mask,n_classes,num_feats

def preprocess_csr2csc(rowptr,colind,args):
    # numlist = torch.arange(colind.size(0), device=args.gpu, dtype=torch.int32)
    numlist=torch.arange(colind.size(0))

    adj_csr=sp.csr_matrix((numlist.numpy(),colind.cpu().numpy(),rowptr.cpu().numpy()))
    adj_csc=adj_csr.tocsc()
    # permute=adj_csc.data
    # print(permute)
    # print(torch.max(torch.from_numpy(permute)))
    colptr=adj_csc.indptr
    rowind=adj_csc.indices
    # print(colptr.shape)
    # colptr, rowind, permute = spmm.csr2csc(rowptr, colind, numlist.float())
    # permute = permute.int()
    return torch.from_numpy(colptr).to(args.gpu),torch.from_numpy(rowind).to(args.gpu)

def main(args):
    #load dataset
    row_ind,row_ptr,col_ind,features,labels,train_mask,val_mask,test_mask,n_classes,num_feats=load_dataset(args)
    n_edges = row_ind.shape[0]

    # row_ind=row_ind.to(args.gpu).int()
    row_ptr=row_ptr.to(args.gpu).int()
    col_ind=col_ind.to(args.gpu).int()


    col_ptr,row_ind=preprocess_csr2csc(row_ptr,col_ind,args)
    
    model=GATConv(args.in_feats,args.out_feats,args.num_heads).to(args.gpu)
    features=torch.rand(row_ptr.shape[0]-1,args.in_feats,device=args.gpu)
    

    for _ in range(5):
        model(row_ptr,col_ind,col_ptr,row_ind,features)


    torch.cuda.synchronize()
    start=time.time()
    
    for epoch in range(args.epochs):
        model(row_ptr,col_ind,col_ptr,row_ind,features)   
        
    torch.cuda.synchronize()
    end=time.time()

    print("gatconv_our forward time:",(end-start)/args.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--in_feats",type=int,default=16)
    parser.add_argument("--out_feats",type=int,default=6)
    parser.add_argument("--dataset",type=str,default="cora")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=400,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=1,
                        help="number of hidden attention heads")
    parser.add_argument('--profileio',type=int,default=0)
                    
    args = parser.parse_args()

    main(args)


