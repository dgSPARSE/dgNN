import os
import torch

import fused_gmmconv as gmm
import mhsddmm
import mhtranspose
# import time

def GmmConvFuse(rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma):
    return GmmFuseFunction.apply(rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma)

class GmmFuseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma):
        out = gmm.gmmconv(rowptr, colind, node_feat, pseudo, mu, sigma) # (E, K, F)
        ctx.save_for_backward(rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma)
        return out

    @staticmethod
    def backward(ctx, grad_out): # (E, F)
        rowptr, colind, colptr, rowind, permute, node_feat, pseudo, mu, sigma = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        guassian_grad = mhsddmm.mhsddmm(rowptr, colind, grad_out, node_feat)
        pseudo_tran = mhtranspose.mhtranspose(permute, pseudo) #
        node_feat_grad = gmm.gmmconv(colptr, rowind, grad_out, pseudo_tran, mu, sigma)
        pseudo_out, mu_out, sigma_out = gmm.GaussianBp(pseudo, mu, sigma, guassian_grad)
        return None, None, None, None, None, node_feat_grad, pseudo_out, mu_out, sigma_out




