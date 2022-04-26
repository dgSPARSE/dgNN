from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from dgl.data import register_data_args, load_data
from scipy import sparse

from dgNN.operators import GmmConvFuse

class GMMConv_pyg(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, dim: int, kernel_size: int,
                 separate_gaussians: bool = False, aggr: str = 'mean',
                 root_weight: bool = True, bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.separate_gaussians = separate_gaussians
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.rel_in_channels = in_channels[0]

        if in_channels[0] > 0:
            self.g = Parameter(
                Tensor(in_channels[0], out_channels * kernel_size))

            if not self.separate_gaussians:
                self.mu = Parameter(Tensor(kernel_size, dim))
                self.sigma = Parameter(Tensor(kernel_size, dim))
            if self.separate_gaussians:
                self.mu = Parameter(
                    Tensor(in_channels[0], out_channels, kernel_size, dim))
                self.sigma = Parameter(
                    Tensor(in_channels[0], out_channels, kernel_size, dim))
        else:
            self.g = torch.nn.parameter.UninitializedParameter()
            self.mu = torch.nn.parameter.UninitializedParameter()
            self.sigma = torch.nn.parameter.UninitializedParameter()
            self._hook = self.register_forward_pre_hook(
                self.initialize_parameters)

        if root_weight:
            self.root = Linear(in_channels[1], out_channels, bias=False,
                               weight_initializer='glorot')

        # if bias:
        #     self.bias = Parameter(torch.Tensor(out_channels))
        # else:
        #     self.register_parameter('bias', None)

        # self.reset_parameters()



    def forward(self, row_ptr,col_ind,col_ptr,row_ind,permute,
    x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None):


        x=torch.matmul(x,self.g)
        inv_sigma=1/(self.sigma+1e-15)
        out_dgnn=GmmConvFuse(row_ptr,col_ind,col_ptr,row_ind,permute,x.view(-1, self.kernel_size, self.out_channels),edge_attr,self.mu,inv_sigma).sum(1)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                                 size=size)
        
        print(torch.allclose(out,out_dgnn,1e-3,1e-5))
        print(torch.max(torch.absolute(out-out_dgnn)))
            
        return out


    def message(self, x_j: Tensor, edge_attr: Tensor):
        EPS = 1e-15
        F, M = self.rel_in_channels, self.out_channels
        (E, D), K = edge_attr.size(), self.kernel_size

        if not self.separate_gaussians:
            gaussian = -0.5 * (edge_attr.view(E, 1, D) -
                               self.mu.view(1, K, D)).pow(2)
            gaussian = gaussian / (EPS + self.sigma.view(1, K, D).pow(2))
            gaussian = torch.exp(gaussian.sum(dim=-1))  # [E, K]

            return (x_j.view(E, K, M) * gaussian.view(E, K, 1)).sum(dim=-2)

        else:
            gaussian = -0.5 * (edge_attr.view(E, 1, 1, 1, D) -
                               self.mu.view(1, F, M, K, D)).pow(2)
            gaussian = gaussian / (EPS + self.sigma.view(1, F, M, K, D).pow(2))
            gaussian = torch.exp(gaussian.sum(dim=-1))  # [E, F, M, K]

            gaussian = gaussian * self.g.view(1, F, M, K)
            gaussian = gaussian.sum(dim=-1)  # [E, F, M]

            return (x_j.view(E, F, 1) * gaussian).sum(dim=-2)  # [E, M]

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, dim={self.dim})')


import argparse
import torch
import dgl



from dgNN.operators import GATConvFuse

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

    adj_csc=adj_csr.tocsc()

    col_ptr=torch.from_numpy(adj_csc.indptr)
    row_ind=torch.from_numpy(adj_csc.indices)

    return row_ptr,col_ind,col_ptr,row_ind,g

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

    pseudo = torch.rand(n_edges, args.dim).to(args.gpu)
    src,dst=g.edges(order='srcdst')
    edge_idx=torch.stack((src,dst)).to(args.gpu)

    model=GMMConv_pyg(args.in_feats,args.out_feats,args.dim,args.kernel_size).to(args.gpu)
    features=torch.rand(rowptr.shape[0]-1,args.in_feats,device=args.gpu)
    
    for _ in range(args.epochs):
        model(rowptr,colind,colptr,rowind,permute,features,edge_idx,pseudo)   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--in_feats",type=int,default=16)
    parser.add_argument("--out_feats",type=int,default=16)
    parser.add_argument("--dataset",type=str,default="cora")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--dim", type=int, default=2,
                        help="number of hidden attention heads")
    parser.add_argument('--kernel_size',type=int,default=3)
                    
    args = parser.parse_args()

    main(args)