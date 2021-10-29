import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import KNNGraph
import sys
sys.path.append('../..')
from layers.edgeconv_layer import EdgeConv
import time



import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

from torch.autograd.profiler import profile

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
parser.add_argument('--profileio',type=int,default=0)

args = parser.parse_args()



dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data=torch.randn(args.batch_size*1024,3)
nng = KNNGraph(args.k)
model=EdgeConv(3,args.out_feat).to(dev)
g = nng(data)
src,dst=g.edges()
src=src.to(dev)
data=data.to(dev)


for _ in range(5):   
    model(args.k,src.int(),data)

torch.cuda.synchronize()
start=time.time()

for _ in range(args.epochs):
    model(args.k,src.int(),data)
    
    
torch.cuda.synchronize()
end=time.time()

print("edgeconv_our forward time:",(end-start)/args.epochs)