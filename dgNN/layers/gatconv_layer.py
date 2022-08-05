from torch import nn
import torch
from torch.nn.modules.linear import Identity

from ..operators.fused_gatconv import GATConvFuse, GATConvFuse_inference

class GATConv(nn.Module): # our gat layer
    def __init__(self,
                in_feats,
                out_feats,
                num_heads,
                feat_drop=0.,
                attn_drop=0.,
                negative_slope=0.2,
                residual=False,
                activation=None,
                bias=True
                ):
        super(GATConv,self).__init__()
        self.in_feats=in_feats
        self.out_feats=out_feats
        self.num_heads=num_heads
        self.W = nn.Parameter(torch.FloatTensor(in_feats, out_feats * num_heads))
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.negative_slope=negative_slope
        self.attn_drop=attn_drop
        self.feat_drop=nn.Dropout(feat_drop)
        if bias:
            self.bias=nn.Parameter(torch.FloatTensor(size=(num_heads*out_feats,)))
        else:
            self.register_buffer('bias',None)
        
        if residual:
            if in_feats!=out_feats*num_heads:
                self.res_fc=nn.Linear(in_feats,out_feats*num_heads,bias=False)
            else:
                self.res_fc=Identity()
        else:
            self.register_buffer('res_fc',None)
        
        self.reset_parameters()
        self.activation=activation
    
    
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

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        

    def forward(self,row_ptr,col_ind,col_ptr,row_ind,permute,feat):

        h=torch.matmul(feat,self.W).view(-1,self.num_heads,self.out_feats)
        h=self.feat_drop(h)

        attn_row = (self.attn_l * h).sum(dim=-1)
        attn_col = (self.attn_r * h).sum(dim=-1)
        
        if not self.training:
            rst=GATConvFuse_inference(attn_row,attn_col,row_ptr,col_ind,self.negative_slope,h)
        else:
            rst=GATConvFuse(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,permute,self.negative_slope,h,self.attn_drop)
        
        if self.res_fc is not None:
            resval=self.res_fc(h).view(-1,self.num_heads,self.out_feats)
            rst=rst+resval
        
        if self.bias is not None:
            rst=rst+self.bias.view(-1,self.num_heads,self.out_feats)

        if self.activation:
            rst=self.activation(rst)

        return rst

