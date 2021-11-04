"""Torch Module for GMM Conv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init

from ..operators.fused_gmmconv import GmmConvFuse

class GMMConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 dim,
                 n_kernels,
                 aggregator_type='sum',
                 residual=False,
                 bias=True,
                 allow_zero_in_degree=False):
        super(GMMConv, self).__init__()
        self._out_feats = out_feats
        self._dim = dim
        self._n_kernels = n_kernels
        self._allow_zero_in_degree = allow_zero_in_degree


        self.mu = nn.Parameter(th.Tensor(n_kernels, dim))
        self.inv_sigma = nn.Parameter(th.Tensor(n_kernels, dim))
        self.fc = nn.Linear(in_feats, n_kernels * out_feats, bias=False)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc.weight, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            init.xavier_normal_(self.res_fc.weight, gain=gain)
        init.normal_(self.mu.data, 0, 0.1)
        init.constant_(self.inv_sigma.data, 1)
        if self.bias is not None:
            init.zeros_(self.bias.data)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, rowptr, colind, colptr, rowind, permute, feat, pseudo):
            node_feat = self.fc(feat).view(-1, self._n_kernels, self._out_feats)
            rst = GmmConvFuse(rowptr, colind, colptr, rowind, permute, node_feat, pseudo, self.mu, self.inv_sigma).sum(1)
            # residual connection
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat)
            # bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst

