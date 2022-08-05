import torch
from dgl.nn.pytorch import KNNGraph
import time
import GPUtil

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Callable, Optional, Union
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor

from dgNN.operators.fused_edgeconv import EdgeConvFuse


class EdgeConv_test_pyg(MessagePassing):
    def __init__(self, in_channels,out_channels):
        super().__init__(aggr='max')
        self.theta = nn.Linear(in_channels,out_channels,bias=False)
        self.phi=nn.Linear(in_channels,out_channels,bias=False)

    def forward(self,k,src_ind, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)

        h_theta=self.theta(x[0])
        h_phi=self.phi(x[0])
        h_src=h_theta
        h_dst=h_phi-h_theta
        rst_dgnn=EdgeConvFuse(k,src_ind,h_src,h_dst)

        rst_pyg=self.propagate(edge_index, x=x, size=None)

        torch.cuda.synchronize()
        print(torch.allclose(rst_pyg,rst_dgnn,1e-4,1e-5))
        print(torch.max(torch.absolute(rst_pyg-rst_dgnn)))
        return rst_pyg

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.theta(x_j-x_i)+self.phi(x_i)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, default='')
parser.add_argument('--load-model-path', type=str, default='')
parser.add_argument('--save-model-path', type=str, default='')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--k',type=int,default=40)
parser.add_argument('--out-feat',type=int,default=64)
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data=torch.randn(args.batch_size*1024,3)
nng = KNNGraph(args.k)
model=EdgeConv_test_pyg(3,args.out_feat).to(dev)
g = nng(data)
src,dst=g.edges()
src=src.to(dev)
dst=dst.to(dev)
data=data.to(dev)
edge_idx=torch.stack((src,dst)).to(dev)

maxMemory = 0

for _ in range(5):   
    model(args.k,src.int(),data,edge_idx)
    GPUs = GPUtil.getGPUs()
    maxMemory = max(GPUs[args.gpu].memoryUsed, maxMemory)
