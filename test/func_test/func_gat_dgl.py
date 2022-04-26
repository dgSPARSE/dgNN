import argparse
import torch
import dgl

import torch.nn as nn

import dgl.function as fn
from dgl.ops.edge_softmax import edge_softmax

from dgNN.operators import GATConvFuse

class GATConv_test_dgl(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 negative_slope=0.2):
        super(GATConv_test_dgl, self).__init__()
        self._num_heads = num_heads
        self._in_feats= in_feats
        self._out_feats = out_feats

        self.fc = nn.Linear(
                self._in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.negative_slope=negative_slope
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)



    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, row_ptr,col_ind,col_ptr,row_ind, graph, feat):
        with graph.local_scope():             
            feat_src = feat_dst = self.fc(feat).view(-1, self._num_heads, self._out_feats)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = edge_softmax(graph, e)
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst_dgl = graph.dstdata['ft']
            rst_dgnn=GATConvFuse(er,el,row_ptr,col_ind,col_ptr,row_ind,self.negative_slope,feat_src,0.0)
            
            # print(rst_dgnn.shape,rst_dgl.shape)
            torch.cuda.synchronize()
            print(torch.allclose(rst_dgl,rst_dgnn,1e-4,1e-5))
            print(torch.max(torch.absolute(rst_dgl-rst_dgnn)))
            return rst_dgnn

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

def main(args):
    #load dataset
    row_ptr,col_ind,col_ptr,row_ind,g=load_dataset(args)

    row_ptr=row_ptr.to(args.gpu).int()
    col_ind=col_ind.to(args.gpu).int()
    col_ptr=col_ptr.to(args.gpu).int()
    row_ind=row_ind.to(args.gpu).int()
    g=g.to(args.gpu)

    
    model=GATConv_test_dgl(args.in_feats,args.out_feats,args.num_heads).to(args.gpu)
    features=torch.rand(row_ptr.shape[0]-1,args.in_feats,device=args.gpu)
    
    for _ in range(args.epochs):
        model(row_ptr,col_ind,col_ptr,row_ind,g,features)   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--in_feats",type=int,default=16)
    parser.add_argument("--out_feats",type=int,default=16)
    parser.add_argument("--dataset",type=str,default="cora")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=1,
                        help="number of hidden attention heads")
                    
    args = parser.parse_args()
    print(args)
    main(args)

