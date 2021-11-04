from torch import nn
from ..operators.fused_edgeconv import EdgeConvFuse

class EdgeConv(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 batch_norm=False,
                 allow_zero_in_degree=False):
        super(EdgeConv, self).__init__()
        self.batch_norm = batch_norm
        self._allow_zero_in_degree = allow_zero_in_degree

        self.theta = nn.Linear(in_feat, out_feat,bias=False)
        self.phi = nn.Linear(in_feat, out_feat,bias=False)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, k,src_ind,feat):
        h_theta=self.theta(feat)
        h_phi=self.phi(feat)
        h_src=h_theta
        h_dst=h_phi-h_theta
        result=EdgeConvFuse(k,src_ind,h_src,h_dst)
        
        return result

