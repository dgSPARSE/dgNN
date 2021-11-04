from torch import nn
import torch

from ..operators.fused_gatconv import GATConvFuse

class GATConv(nn.Module): # our gat layer
    def __init__(self,
                in_feats,
                out_feats,
                num_heads,
                negative_slope=0.2,

                ):
        super(GATConv,self).__init__()
        self.in_feats=in_feats
        self.out_feats=out_feats
        self.num_heads=num_heads
        self.W = nn.Parameter(torch.FloatTensor(in_feats, out_feats * num_heads))
        self.attn_l = nn.Parameter(torch.zeros(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.zeros(size=(1, num_heads, out_feats)))
        self.negative_slope=negative_slope
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        

    def forward(self,row_ptr,col_ind,col_ptr,row_ind,feat):

        h=torch.matmul(feat,self.W).view(-1,self.num_heads,self.out_feats)

        attn_row = (self.attn_l * h).sum(dim=-1)
        attn_col = (self.attn_r * h).sum(dim=-1)
        
        out=GATConvFuse(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,self.negative_slope,h)
            
        return out

